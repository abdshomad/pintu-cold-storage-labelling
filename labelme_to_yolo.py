#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations in `train/` to YOLO detection format.

Output layout (Ultralytics-compatible; default under pintu-cold-storage-datasets/):
  converted_yolo_format/
    images/train/<image files>
    labels/train/<same stem>.txt
    data.yaml
    classes.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "pintu-cold-storage-datasets"


def shape_to_xyxy(shape: dict, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Return pixel-space xmin, ymin, xmax, ymax from a LabelMe shape."""
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None

    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # rectangle (2 corners), polygon, or other types: use axis-aligned bbox of vertices
    xmin = max(0.0, min(xmin, float(img_w)))
    xmax = max(0.0, min(xmax, float(img_w)))
    ymin = max(0.0, min(ymin, float(img_h)))
    ymax = max(0.0, min(ymax, float(img_h)))

    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def xyxy_to_yolo_line(
    cls_id: int, xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> str:
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    # clip
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def collect_class_names(json_paths: list[Path]) -> list[str]:
    names: set[str] = set()
    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for sh in data.get("shapes") or []:
            lab = sh.get("label")
            if lab is not None and str(lab).strip():
                names.add(str(lab).strip())
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe (train folder) → YOLO format")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DATASETS_DIR / "train",
        help="Folder with images and matching .json files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATASETS_DIR / "converted_yolo_format",
        help="Output root (images/train, labels/train, data.yaml)",
    )
    args = parser.parse_args()
    train_dir: Path = args.train_dir.resolve()
    out_root: Path = args.out_dir.resolve()

    if not train_dir.is_dir():
        print(f"Error: train directory not found: {train_dir}", file=sys.stderr)
        return 1

    json_paths = sorted(train_dir.glob("*.json"))
    if not json_paths:
        print(f"Error: no JSON files in {train_dir}", file=sys.stderr)
        return 1

    class_names = collect_class_names(json_paths)
    if not class_names:
        print("Warning: no labels found in JSON files; class list will be empty.", file=sys.stderr)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_img = 0
    wrote_labels = 0
    errors = 0

    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skip (bad JSON): {jp.name} — {e}", file=sys.stderr)
            errors += 1
            continue

        stem = jp.stem
        image_path = data.get("imagePath") or f"{stem}.png"
        src_image = train_dir / Path(image_path).name
        if not src_image.is_file():
            # try stem with common extensions
            found = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                cand = train_dir / f"{stem}{ext}"
                if cand.is_file():
                    found = cand
                    break
            src_image = found or src_image

        if not src_image.is_file():
            print(f"Skip (no image): {stem}", file=sys.stderr)
            skipped_img += 1
            continue

        dest_image = img_dir / src_image.name
        shutil.copy2(src_image, dest_image)
        copied += 1

        img_w = int(data.get("imageWidth") or 0)
        img_h = int(data.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            print(f"Warning: invalid imageWidth/Height in {jp.name}, skipping labels", file=sys.stderr)
            errors += 1
            continue

        lines: list[str] = []
        for sh in data.get("shapes") or []:
            label = sh.get("label")
            if label is None or not str(label).strip():
                continue
            label = str(label).strip()
            if label not in name_to_id:
                continue
            box = shape_to_xyxy(sh, img_w, img_h)
            if box is None:
                continue
            xmin, ymin, xmax, ymax = box
            cid = name_to_id[label]
            lines.append(xyxy_to_yolo_line(cid, xmin, ymin, xmax, ymax, img_w, img_h))

        label_file = lbl_dir / f"{dest_image.stem}.txt"
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        wrote_labels += 1

    # classes.txt (one name per line, line index = class id)
    classes_txt = out_root / "classes.txt"
    classes_txt.write_text("\n".join(class_names) + ("\n" if class_names else ""), encoding="utf-8")

    # data.yaml (Ultralytics-style)
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_body = f"""# Generated by labelme_to_yolo.py
path: {out_root.as_posix()}
train: images/train
val: images/train

nc: {len(class_names)}

names:
{names_yaml}
"""
    (out_root / "data.yaml").write_text(yaml_body, encoding="utf-8")

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Images copied: {copied}")
    print(f"Label files written: {wrote_labels}")
    if skipped_img:
        print(f"Skipped (missing image): {skipped_img}")
    if errors:
        print(f"Issues (bad JSON / dimensions): {errors}")
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
