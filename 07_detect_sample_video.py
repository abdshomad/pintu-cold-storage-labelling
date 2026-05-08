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
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    mtime_ns: int


def repo_root() -> Path:
    # This script lives in .../pintu-cold-storage-labelling/
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
    args = p.parse_args()
    if args.family is not None:
        args.model_family = args.family
    if args.batch_size_positional is not None:
        args.batch_size = args.batch_size_positional
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
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

    # Prefer common training artifact names (best.pt, last.pt), then newest mtime.
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
    # Try widely-compatible codecs in order.
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


def _annotate_rfdetr(frame_bgr: np.ndarray, detections) -> np.ndarray:
    out = frame_bgr.copy()
    class_ids = getattr(detections, "class_id", None)
    confidences = getattr(detections, "confidence", None)
    xyxy = getattr(detections, "xyxy", None)
    data = getattr(detections, "data", {}) or {}
    class_names = data.get("class_name", None)

    if xyxy is None:
        return out

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = "obj"
        if class_names is not None and i < len(class_names):
            cls_name = str(class_names[i])
        elif class_ids is not None and i < len(class_ids):
            cls_name = f"id-{int(class_ids[i])}"

        conf_txt = ""
        if confidences is not None and i < len(confidences):
            conf_txt = f" {float(confidences[i]):.2f}"

        label = f"{cls_name}{conf_txt}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def _fallback_sizes(target: int) -> list[int]:
    ladder = [64, 32, 16, 8, 4, 2, 1]
    sizes: list[int] = [target]
    for size in ladder:
        if size < target:
            sizes.append(size)
    if 1 not in sizes:
        sizes.append(1)
    # Keep order and uniqueness.
    seen: set[int] = set()
    uniq: list[int] = []
    for s in sizes:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _predict_yolo_batch(model: Any, frames_bgr: list[np.ndarray], conf: float, iou: float) -> list[np.ndarray]:
    results = model.predict(source=frames_bgr, conf=conf, iou=iou, verbose=False)
    return [r.plot() for r in results]


def _predict_rfdetr_batch(model: Any, frames_bgr: list[np.ndarray], conf: float) -> list[np.ndarray]:
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    detections = model.predict(frames_rgb, threshold=conf)
    if isinstance(detections, list):
        return [_annotate_rfdetr(frame, det) for frame, det in zip(frames_bgr, detections)]
    # Single detection object fallback.
    return [_annotate_rfdetr(frames_bgr[0], detections)]


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
        # For fine-tuned inference, RF-DETR accepts local pretrained/checkpoint path.
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
    try:
        with tqdm(total=total, desc="Detecting", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                # Read one chunk using the currently selected batch size.
                chunk_size = fallback_sizes[current_bs_idx]
                frames: list[np.ndarray] = []
                for _ in range(chunk_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)

                if not frames:
                    break

                # Try current batch size; if inference fails, reduce and retry this same chunk.
                pending = frames
                while pending:
                    bs = fallback_sizes[current_bs_idx]
                    take = min(bs, len(pending))
                    sub = pending[:take]
                    try:
                        if args.model_family == "y":
                            plotted_frames = _predict_yolo_batch(model, sub, conf=args.conf, iou=args.iou)
                        else:
                            plotted_frames = _predict_rfdetr_batch(model, sub, conf=args.conf)
                    except Exception as e:
                        if current_bs_idx + 1 >= len(fallback_sizes):
                            raise RuntimeError(
                                f"Inference failed even at batch size {bs}. Last error: {type(e).__name__}: {e}"
                            ) from e
                        new_bs = fallback_sizes[current_bs_idx + 1]
                        print(f"[warn] batch size {bs} failed ({type(e).__name__}); retrying with {new_bs}")
                        current_bs_idx += 1
                        continue

                    for plotted in plotted_frames:
                        writer.write(plotted)
                        idx += 1
                        pbar.update(1)

                    pending = pending[take:]
    finally:
        cap.release()
        writer.release()

    print(f"[done] wrote {idx} frames to: {output_path}")


if __name__ == "__main__":
    run()
