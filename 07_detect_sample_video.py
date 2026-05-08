"""
Run object detection on a sample video using the latest trained model.

Default input:
  {repo_root}/pintu-cold-storage-datasets/Video Pintu Cold Storage Terbuka Tertutup ada Tirainya.mp4

Default output:
  {repo_root}/pintu-cold-storage-detection-result/{video-stem}-detection-result.mp4

Model discovery:
  - Scans {repo_root}/pintu-cold-storage-models recursively
  - Picks the newest supported checkpoint by mtime
  - Supports:
      y: YOLO (.pt)
      r: RF-DETR (.pth / .ckpt / .pt)

HUD / alerts:
  - Door open vs closed from celah-pintu vs daun-pintu horizontal widths (--celah-daun-ratio).
  - Top-right screen indicator ROI: hijau/merah (HSV).
  - Open duration; alert when open longer than --alert-open-seconds (blink + optional sound).

Future messaging (not implemented; hook later from the same long-open event as --alert-sound):
  - Telegram: HTTPS api.telegram.org/bot<token>/sendMessage; store TELEGRAM_BOT_TOKEN and chat id in env.
  - WhatsApp: Meta Cloud API or a gateway; keep secrets in env, rate-limit and dedupe like sound.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    mtime_ns: int


@dataclass(frozen=True)
class Detection:
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]


@dataclass
class DoorHudState:
    open_start_t: float | None = None
    long_alert_latched: bool = False
    last_repeat_sound_t: float = -1e9
    alert_anim_until_t: float = -1e9


_SV_PALETTE = sv.ColorPalette.from_hex(["#FF3B30", "#34C759", "#00D4FF"])
_SV_BOX_ANNOTATOR = sv.BoxAnnotator(color=_SV_PALETTE, thickness=2)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_video_path() -> Path:
    return repo_root() / "pintu-cold-storage-datasets" / "Video Pintu Cold Storage Terbuka Tertutup ada Tirainya.mp4"


def default_models_dir() -> Path:
    return repo_root() / "pintu-cold-storage-models"


def default_output_path(video_path: Path, model_family: str) -> Path:
    out_dir = repo_root() / "pintu-cold-storage-detection-result"
    return out_dir / f"{video_path.stem}-{model_family}-detection-result.mp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect objects on sample video using latest model from models folder.")
    p.add_argument(
        "family",
        nargs="?",
        choices=("y", "r"),
        default=None,
        help="Optional positional model family shorthand: y=YOLO, r=RF-DETR",
    )
    p.add_argument(
        "batch_size_positional",
        nargs="?",
        type=int,
        default=None,
        help="Optional positional batch size shorthand, e.g. `... y 64`",
    )
    p.add_argument("--video", type=Path, default=default_video_path(), help="Input video path")
    p.add_argument("--models-dir", type=Path, default=default_models_dir(), help="Model directory to scan recursively")
    p.add_argument("--output", type=Path, default=None, help="Output video path (default: auto in detection-result folder)")
    p.add_argument(
        "--model-family",
        "-m",
        choices=("y", "r"),
        default="y",
        help="Model family: y=YOLO, r=RF-DETR (default: y)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size target (auto-fallback on error: 64,32,16,8,4,2,1)",
    )
    p.add_argument(
        "--celah-class",
        type=str,
        default="celah-pintu",
        help="Class name for celah (gap) boxes",
    )
    p.add_argument(
        "--daun-class",
        type=str,
        default="daun-pintu",
        help="Class name for daun (door leaf) boxes",
    )
    p.add_argument(
        "--celah-daun-ratio",
        type=float,
        default=0.15,
        help="Terbuka iff W_celah > ratio * W_daun (default 0.15)",
    )
    p.add_argument(
        "--indicator-roi",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(0.78, 0.0, 1.0, 0.18),
        help="Normalized ROI x0 y0 x1 y1 for screen indicator (top-right)",
    )
    p.add_argument(
        "--indicator-roi-alpha",
        type=float,
        default=0.4,
        help="Opacity of ROI tint (0.4 = 60%% transparent overlay)",
    )
    p.add_argument(
        "--ui-theme",
        choices=("dark", "light"),
        default="dark",
        help="UI theme for HUD/panels: dark or light",
    )
    p.add_argument("--debug-roi", action="store_true", help="Draw indicator ROI rectangle outline at full opacity")
    p.add_argument(
        "--alert-open-seconds",
        type=float,
        default=10.0,
        help="Show long-open alert when door open longer than this (seconds)",
    )
    p.add_argument(
        "--alert-animation-seconds",
        type=float,
        default=2.0,
        help="Duration of red alert animation after threshold crossing (seconds)",
    )
    p.add_argument("--alert-sound", action="store_true", help="Play sound when long-open threshold is first crossed")
    p.add_argument(
        "--alert-sound-repeat-sec",
        type=float,
        default=0.0,
        help="If >0, repeat alert sound every this many seconds while still in long-open state",
    )
    p.add_argument(
        "--alert-sound-file",
        type=Path,
        default=None,
        help="Optional WAV file for paplay/aplay; else terminal bell",
    )
    args = p.parse_args()
    if args.family is not None:
        args.model_family = args.family
    if args.batch_size_positional is not None:
        args.batch_size = args.batch_size_positional
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not (0.0 <= args.indicator_roi_alpha <= 1.0):
        raise SystemExit("--indicator-roi-alpha must be in [0, 1]")
    return args


def find_latest_yolo_checkpoint(models_dir: Path) -> Path:
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    preferred_names = ("best.pt", "last.pt")
    candidates: list[ModelCandidate] = []

    for weight in models_dir.rglob("*.pt"):
        if not weight.is_file():
            continue
        try:
            mtime_ns = weight.stat().st_mtime_ns
        except OSError:
            continue
        candidates.append(ModelCandidate(path=weight, mtime_ns=mtime_ns))

    if not candidates:
        raise FileNotFoundError(
            "No supported model checkpoints found in "
            f"{models_dir}. Expected at least one '.pt' file (e.g. weights/best.pt)."
        )

    def rank(c: ModelCandidate) -> tuple[int, int]:
        preferred = 1 if c.path.name in preferred_names else 0
        return (preferred, c.mtime_ns)

    latest = max(candidates, key=rank)
    return latest.path


def find_latest_rfdetr_checkpoint(models_dir: Path) -> Path:
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    preferred_names = ("checkpoint_best_regular.pth", "best.pth", "last.pth", "checkpoint.pth")
    candidates: list[ModelCandidate] = []

    for ext in ("*.pth", "*.ckpt", "*.pt"):
        for weight in models_dir.rglob(ext):
            if not weight.is_file():
                continue
            try:
                mtime_ns = weight.stat().st_mtime_ns
            except OSError:
                continue
            candidates.append(ModelCandidate(path=weight, mtime_ns=mtime_ns))

    if not candidates:
        raise FileNotFoundError(
            "No supported RF-DETR checkpoints found in "
            f"{models_dir}. Expected one of: .pth, .ckpt, .pt"
        )

    def rank(c: ModelCandidate) -> tuple[int, int]:
        preferred = 1 if c.path.name in preferred_names else 0
        return (preferred, c.mtime_ns)

    latest = max(candidates, key=rank)
    return latest.path


def build_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_str in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Failed to open output video writer with codecs: mp4v, avc1, H264")


def _fallback_sizes(target: int) -> list[int]:
    ladder = [64, 32, 16, 8, 4, 2, 1]
    sizes: list[int] = [target]
    for size in ladder:
        if size < target:
            sizes.append(size)
    if 1 not in sizes:
        sizes.append(1)
    seen: set[int] = set()
    uniq: list[int] = []
    for s in sizes:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def box_width(xyxy: tuple[int, int, int, int]) -> float:
    x1, _y1, x2, _y2 = xyxy
    return float(max(0, x2 - x1))


def door_is_open(
    detections: list[Detection],
    celah_name: str,
    daun_name: str,
    ratio: float,
) -> tuple[bool, float, float]:
    """Return (terbuka, W_celah, W_daun) from horizontal (x-axis) widths."""
    w_celah = sum(box_width(d.xyxy) for d in detections if d.name == celah_name)
    w_daun = sum(box_width(d.xyxy) for d in detections if d.name == daun_name)
    if w_celah == 0:
        return False, w_celah, w_daun
    if w_daun == 0:
        return True, w_celah, w_daun
    return (w_celah > ratio * w_daun), w_celah, w_daun


def yolo_results_to_detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names or {}
    out: list[Detection] = []
    for i in range(len(boxes)):
        xyxy_t = boxes.xyxy[i].detach().cpu().numpy()
        cls_id = int(boxes.cls[i].detach().cpu().numpy())
        conf = float(boxes.conf[i].detach().cpu().numpy())
        name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy_t]
        out.append(Detection(name=str(name), conf=conf, xyxy=(x1, y1, x2, y2)))
    return out


def rfdetr_det_to_detections(detections: Any) -> list[Detection]:
    class_ids = getattr(detections, "class_id", None)
    confidences = getattr(detections, "confidence", None)
    xyxy = getattr(detections, "xyxy", None)
    data = getattr(detections, "data", {}) or {}
    class_names = data.get("class_name", None)

    if xyxy is None:
        return []

    out: list[Detection] = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = "obj"
        if class_names is not None and i < len(class_names):
            cls_name = str(class_names[i])
        elif class_ids is not None and i < len(class_ids):
            cls_name = f"id-{int(class_ids[i])}"

        conf_v = 1.0
        if confidences is not None and i < len(confidences):
            conf_v = float(confidences[i])
        out.append(Detection(name=cls_name, conf=conf_v, xyxy=(x1, y1, x2, y2)))
    return out


def norm_roi_to_px(
    w: int, h: int, nx0: float, ny0: float, nx1: float, ny1: float
) -> tuple[int, int, int, int]:
    x0 = int(round(nx0 * w))
    y0 = int(round(ny0 * h))
    x1 = int(round(nx1 * w))
    y1 = int(round(ny1 * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, min(1, w), min(1, h)
    return x0, y0, x1, y1


def classify_indicator_color(
    frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> str:
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return "tidak_dikenal"
    small = roi
    if max(small.shape[:2]) > 160:
        scale = 160 / max(small.shape[:2])
        small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask_g = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_r1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255]))
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)

    g = int(cv2.countNonZero(mask_g))
    r = int(cv2.countNonZero(mask_r))
    n = max(1, mask_g.size)
    if g + r < max(20, n // 200):
        return "tidak_dikenal"
    if g > r * 1.15 and g > 30:
        return "hijau"
    if r > g * 1.15 and r > 30:
        return "merah"
    return "tidak_dikenal"


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    *,
    celah_name: str,
    daun_name: str,
) -> None:
    if not detections:
        return
    xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
    conf = np.array([d.conf for d in detections], dtype=np.float32)
    class_ids: list[int] = []
    labels: list[str] = []
    for d in detections:
        if d.name == celah_name:
            cid = 0  # merah
        elif d.name == daun_name:
            cid = 1  # hijau
        else:
            cid = 2
        class_ids.append(cid)
        labels.append(f"{d.name} {d.conf:.2f}")
    det_sv = sv.Detections(
        xyxy=xyxy,
        confidence=conf,
        class_id=np.array(class_ids, dtype=np.int32),
    )
    annotated = _SV_BOX_ANNOTATOR.annotate(scene=frame_bgr, detections=det_sv)
    frame_bgr[:, :] = annotated


def draw_roi_tint(
    frame_bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha: float,
    tint_bgr: tuple[int, int, int],
) -> None:
    """Fill ROI with semi-transparent tint (alpha = opacity of tint layer)."""
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return
    layer = np.full_like(roi, tint_bgr)
    blended = cv2.addWeighted(roi, 1.0 - alpha, layer, alpha, 0)
    frame_bgr[y0:y1, x0:x1] = blended


def draw_panel(
    frame_bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    fill_bgr: tuple[int, int, int],
    alpha: float,
    border_bgr: tuple[int, int, int] | None = None,
    border_thickness: int = 1,
) -> None:
    if x1 <= x0 or y1 <= y0:
        return
    panel = frame_bgr[y0:y1, x0:x1].copy()
    cv2.rectangle(panel, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), fill_bgr, -1)
    cv2.addWeighted(panel, alpha, frame_bgr[y0:y1, x0:x1], 1.0 - alpha, 0, dst=frame_bgr[y0:y1, x0:x1])
    if border_bgr is not None and border_thickness > 0:
        cv2.rectangle(frame_bgr, (x0, y0), (x1 - 1, y1 - 1), border_bgr, border_thickness)


def ui_theme_colors(theme: str) -> dict[str, tuple[int, int, int]]:
    if theme == "light":
        return {
            "hud_fill": (245, 245, 245),
            "hud_border": (90, 90, 90),
            "hud_text": (20, 20, 20),
            "hud_shadow": (240, 240, 240),
            "pill_fill": (235, 235, 235),
            "pill_border": (80, 80, 80),
            "bar_base": (140, 220, 140),
            "bar_fill": (80, 80, 230),
            "bar_border": (70, 70, 70),
            "warn_fill": (80, 80, 230),
            "warn_border": (255, 255, 255),
            "warn_text": (255, 255, 255),
        }
    return {
        "hud_fill": (20, 20, 20),
        "hud_border": (220, 220, 220),
        "hud_text": (255, 255, 255),
        "hud_shadow": (40, 40, 40),
        "pill_fill": (30, 30, 30),
        "pill_border": (180, 180, 180),
        "bar_base": (0, 120, 0),
        "bar_fill": (0, 0, 220),
        "bar_border": (255, 255, 255),
        "warn_fill": (0, 0, 220),
        "warn_border": (255, 255, 255),
        "warn_text": (255, 255, 255),
    }


def try_play_alert_sound(wav_path: Path | None) -> None:
    if wav_path is not None and wav_path.is_file():
        for cmd in (["paplay", str(wav_path)], ["aplay", str(wav_path)]):
            try:
                subprocess.run(cmd, check=False, timeout=8, capture_output=True)
                return
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
    print("\a", end="", flush=True)


def play_tet_tet_tet(wav_path: Path | None) -> None:
    for _ in range(3):
        try_play_alert_sound(wav_path)
        time.sleep(0.18)


def play_tet_tet_tet_non_blocking(wav_path: Path | None) -> None:
    # Run alert tone sequence in a daemon thread so frame processing does not pause.
    t = threading.Thread(target=play_tet_tet_tet, args=(wav_path,), daemon=True)
    t.start()


def render_frame(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    *,
    args: argparse.Namespace,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
    state: DoorHudState,
) -> np.ndarray:
    out = frame_bgr.copy()
    ui = ui_theme_colors(args.ui_theme)
    draw_detections(out, detections, celah_name=args.celah_class, daun_name=args.daun_class)

    is_open, w_celah, w_daun = door_is_open(
        detections,
        args.celah_class,
        args.daun_class,
        args.celah_daun_ratio,
    )
    open_pct = 100.0 if w_daun <= 0 and w_celah > 0 else (0.0 if w_daun <= 0 else (w_celah / w_daun) * 100.0)
    open_pct = max(0.0, min(100.0, open_pct))
    t = frame_index / fps if fps > 0 else 0.0

    if is_open:
        if state.open_start_t is None:
            state.open_start_t = t
        open_sec = t - state.open_start_t
    else:
        state.open_start_t = None
        open_sec = 0.0
        state.long_alert_latched = False
        state.alert_anim_until_t = -1e9

    ix0, iy0, ix1, iy1 = norm_roi_to_px(width, height, *args.indicator_roi)
    ind_label = classify_indicator_color(out, ix0, iy0, ix1, iy1)
    ind_display = {"hijau": "HIJAU", "merah": "MERAH", "tidak_dikenal": "?"}.get(ind_label, "?")
    if ind_label == "hijau":
        tint = (0, 180, 0)
    elif ind_label == "merah":
        tint = (0, 0, 220)
    else:
        tint = (80, 80, 80)

    draw_roi_tint(out, ix0, iy0, ix1, iy1, args.indicator_roi_alpha, tint)
    if args.debug_roi:
        cv2.rectangle(out, (ix0, iy0), (ix1 - 1, iy1 - 1), (255, 255, 255), 2)

    long_open = is_open and open_sec > args.alert_open_seconds
    toggle = max(1, int(round(fps / 4))) if fps > 0 else 1
    blink_on = ((frame_index // toggle) % 2) == 0
    anim_active = t <= state.alert_anim_until_t

    if (long_open and blink_on) or (anim_active and blink_on):
        msg = f"PERINGATAN: PINTU TERBUKA > {args.alert_open_seconds:g}s"
        warn_scale = 0.95
        warn_thickness = 2
        max_text_width = int(width * 0.82)
        (tw, th), baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, warn_scale, warn_thickness)
        while tw > max_text_width and warn_scale > 0.45:
            warn_scale -= 0.05
            warn_thickness = 1 if warn_scale < 0.75 else 2
            (tw, th), baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, warn_scale, warn_thickness)
        pad_x, pad_y = 22, 16
        box_w = tw + 2 * pad_x
        box_h = th + baseline + 2 * pad_y
        x0 = max(0, (width - box_w) // 2)
        y0 = max(0, (height - box_h) // 2)
        x1 = min(width, x0 + box_w)
        y1 = min(height, y0 + box_h)

        draw_panel(
            out,
            x0,
            y0,
            x1,
            y1,
            fill_bgr=ui["warn_fill"],
            alpha=0.45,
            border_bgr=ui["warn_border"],
            border_thickness=2,
        )

        tx = x0 + pad_x
        ty = y0 + pad_y + th
        cv2.putText(
            out,
            msg,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            warn_scale,
            ui["warn_text"],
            warn_thickness,
            cv2.LINE_AA,
        )

    door_txt = "TERBUKA" if is_open else "TERTUTUP"
    line1 = f"Pintu: {door_txt}"
    ms = int(round((open_sec % 1.0) * 1000))
    s_int = int(open_sec)
    line2 = f"Terbuka: {s_int}.{ms:03d} s" if is_open else "Terbuka: —"
    line3 = f"Terbuka: {open_pct:.1f}%  |  Indikator: {ind_display}"

    ui_scale = max(0.85, min(1.5, width / 1280.0))
    hud_font_scale = 0.62 * ui_scale
    hud_step = int(round(22 * ui_scale))
    hud_pad_x = int(round(12 * ui_scale))
    hud_pad_y = int(round(10 * ui_scale))
    hud_text_y0 = int(round(24 * ui_scale))

    # Transparent HUD background (adaptive width to text).
    max_line_w = 0
    for txt in (line1, line2, line3):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, 1)
        max_line_w = max(max_line_w, tw)
    panel_x0, panel_y0 = int(round(6 * ui_scale)), int(round(8 * ui_scale))
    panel_w = max_line_w + hud_pad_x * 2
    panel_h = hud_pad_y * 2 + hud_step * 3
    panel_x1, panel_y1 = panel_x0 + panel_w, panel_y0 + panel_h
    draw_panel(
        out,
        panel_x0,
        panel_y0,
        panel_x1,
        panel_y1,
        fill_bgr=ui["hud_fill"],
        alpha=0.45,
        border_bgr=ui["hud_border"],
        border_thickness=1,
    )

    y0 = panel_y0 + hud_text_y0
    for i, txt in enumerate((line1, line2, line3)):
        y = y0 + i * hud_step
        x = panel_x0 + hud_pad_x
        shadow_thickness = 2 if ui_scale > 1.1 else 1
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, ui["hud_shadow"], shadow_thickness, cv2.LINE_AA)
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, ui["hud_text"], 1, cv2.LINE_AA)

    # Openness scale bar (0-100%): base hijau, fill merah sesuai celah.
    bar_x0 = panel_x0 + hud_pad_x
    bar_y0 = panel_y1 + int(round(8 * ui_scale))
    bar_w = int(round(300 * ui_scale))
    bar_h = max(10, int(round(14 * ui_scale)))
    cv2.rectangle(out, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), ui["bar_base"], -1)
    fill_w = int(round((open_pct / 100.0) * bar_w))
    if fill_w > 0:
        cv2.rectangle(out, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y0 + bar_h), ui["bar_fill"], -1)
    cv2.rectangle(out, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), ui["bar_border"], 1)
    cv2.putText(out, f"Skala terbuka: {open_pct:.1f}%", (bar_x0 + bar_w + int(round(10 * ui_scale)), bar_y0 + bar_h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui_scale, ui["hud_text"], 1, cv2.LINE_AA)

    pill_w = int(round(220 * ui_scale))
    pill_h = int(round(30 * ui_scale))
    pill_x = width - pill_w - int(round(12 * ui_scale))
    pill_y = int(round(10 * ui_scale))
    draw_panel(
        out,
        pill_x,
        pill_y,
        pill_x + pill_w,
        pill_y + pill_h,
        fill_bgr=ui["pill_fill"],
        alpha=0.6,
        border_bgr=ui["pill_border"],
        border_thickness=1,
    )
    cv2.putText(
        out,
        f"Indikator {ind_display}",
        (pill_x + int(round(10 * ui_scale)), pill_y + int(round(20 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * ui_scale,
        ui["hud_text"],
        1,
        cv2.LINE_AA,
    )

    if long_open:
        if not state.long_alert_latched:
            state.long_alert_latched = True
            state.alert_anim_until_t = t + max(0.0, args.alert_animation_seconds)
            if args.alert_sound:
                play_tet_tet_tet_non_blocking(args.alert_sound_file)
            state.last_repeat_sound_t = t
        elif args.alert_sound_repeat_sec > 0 and args.alert_sound:
            if t - state.last_repeat_sound_t >= args.alert_sound_repeat_sec:
                play_tet_tet_tet_non_blocking(args.alert_sound_file)
                state.last_repeat_sound_t = t

    return out


def _predict_yolo_batch_raw(model: Any, frames_bgr: list[np.ndarray], conf: float, iou: float) -> Any:
    return model.predict(source=frames_bgr, conf=conf, iou=iou, verbose=False)


def _predict_rfdetr_batch_raw(model: Any, frames_bgr: list[np.ndarray], conf: float) -> Any:
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    return model.predict(frames_rgb, threshold=conf)


def run() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    models_dir = args.models_dir.resolve()
    output_path = (args.output.resolve() if args.output else default_output_path(video_path, args.model_family))

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if args.model_family == "y":
        model_path = find_latest_yolo_checkpoint(models_dir)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise SystemExit(
                "Missing dependency for YOLO: ultralytics. Install with: pip install -U ultralytics torch torchvision"
            ) from e
        model = YOLO(str(model_path))
    else:
        model_path = find_latest_rfdetr_checkpoint(models_dir)
        try:
            from rfdetr import RFDETRNano
        except ImportError as e:
            raise SystemExit("Missing dependency for RF-DETR: rfdetr. Install with: pip install -U rfdetr") from e
        model = RFDETRNano(pretrain_weights=str(model_path))

    print(f"[video]  {video_path}")
    print(f"[family] {args.model_family}")
    print(f"[batch]  target={args.batch_size}")
    print(f"[model]  {model_path}")
    print(f"[output] {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = build_writer(output_path, width, height, fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    total = frame_count if frame_count > 0 else None
    fallback_sizes = _fallback_sizes(args.batch_size)
    current_bs_idx = 0

    idx = 0
    hud_state = DoorHudState()
    try:
        with tqdm(total=total, desc="Detecting", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                chunk_size = fallback_sizes[current_bs_idx]
                frames: list[np.ndarray] = []
                for _ in range(chunk_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)

                if not frames:
                    break

                pending = frames
                while pending:
                    bs = fallback_sizes[current_bs_idx]
                    take = min(bs, len(pending))
                    sub = pending[:take]
                    try:
                        if args.model_family == "y":
                            raw = _predict_yolo_batch_raw(model, sub, conf=args.conf, iou=args.iou)
                            for frame, res in zip(sub, raw):
                                dets = yolo_results_to_detections(res)
                                out = render_frame(
                                    frame,
                                    dets,
                                    args=args,
                                    frame_index=idx,
                                    fps=fps,
                                    width=width,
                                    height=height,
                                    state=hud_state,
                                )
                                writer.write(out)
                                idx += 1
                                pbar.update(1)
                        else:
                            raw = _predict_rfdetr_batch_raw(model, sub, conf=args.conf)
                            if isinstance(raw, list):
                                det_list = raw
                            elif len(sub) == 1:
                                det_list = [raw]
                            else:
                                raise RuntimeError(
                                    "RF-DETR returned one detection object for a multi-frame batch. "
                                    "Use --batch-size 1 or switch to YOLO for larger batches."
                                )
                            for frame, det in zip(sub, det_list):
                                dets = rfdetr_det_to_detections(det)
                                out = render_frame(
                                    frame,
                                    dets,
                                    args=args,
                                    frame_index=idx,
                                    fps=fps,
                                    width=width,
                                    height=height,
                                    state=hud_state,
                                )
                                writer.write(out)
                                idx += 1
                                pbar.update(1)
                    except Exception as e:
                        if current_bs_idx + 1 >= len(fallback_sizes):
                            raise RuntimeError(
                                f"Inference failed even at batch size {bs}. Last error: {type(e).__name__}: {e}"
                            ) from e
                        new_bs = fallback_sizes[current_bs_idx + 1]
                        print(f"[warn] batch size {bs} failed ({type(e).__name__}); retrying with {new_bs}")
                        current_bs_idx += 1
                        continue

                    pending = pending[take:]
    finally:
        cap.release()
        writer.release()

    print(f"[done] wrote {idx} frames to: {output_path}")


if __name__ == "__main__":
    run()
