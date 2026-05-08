#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations (per split folder) to a single COCO JSON per split.

If only train/ exists under data-root (no valid/ or test/), all LabelMe samples in
train/ are shuffled and partitioned 70% / 20% / 10% into output train/, valid/, test/.

Output layout (Roboflow / common tooling style; default under pintu-cold-storage-datasets/):
  converted_coco_format/
    train/
      _annotations.coco.json
      <image files>
    valid/
      ...
    test/
      ...
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "pintu-cold-storage-datasets"

# Default split subfolders under --data-root (and mirrored under --out-dir).
DEFAULT_SPLITS = ("train", "valid", "test")

# When only data-root/train exists, output sizes use integer parts of n (sum to n).
SPLIT_RATIOS_PCT = (70, 20, 10)


def collect_labelme_json_paths(split_dir: Path) -> list[Path]:
    """Sorted LabelMe JSON paths under split_dir (excludes COCO export)."""
    out: list[Path] = []
    for jp in sorted(split_dir.glob("*.json")):
        if jp.name == "_annotations.coco.json":
            continue
        out.append(jp)
    return out


def partition_counts(n: int, ratios_pct: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return (n_train, n_valid, n_test) summing to n."""
    r0, r1, r2 = ratios_pct
    total_r = r0 + r1 + r2
    n0 = (n * r0) // total_r
    n1 = (n * r1) // total_r
    n2 = n - n0 - n1
    return n0, n1, n2


def shape_to_xyxy(shape: dict, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None

    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xmin = max(0.0, min(xmin, float(img_w)))
    xmax = max(0.0, min(xmax, float(img_w)))
    ymin = max(0.0, min(ymin, float(img_h)))
    ymax = max(0.0, min(ymax, float(img_h)))

    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def shape_to_segmentation(shape: dict, img_w: int, img_h: int) -> list[list[float]] | None:
    """COCO polygon: one ring as [x1,y1,x2,y2,...]. None if not polygonal."""
    st = (shape.get("shape_type") or "").lower()
    pts = shape.get("points") or []
    if st != "polygon" or len(pts) < 3:
        return None
    flat: list[float] = []
    for p in pts:
        x = max(0.0, min(float(p[0]), float(img_w)))
        y = max(0.0, min(float(p[1]), float(img_h)))
        flat.extend([x, y])
    return [flat]


def resolve_source_image(json_path: Path, data: dict) -> Path | None:
    split_dir = json_path.parent
    stem = json_path.stem
    image_path = data.get("imagePath") or f"{stem}.png"
    src = split_dir / Path(str(image_path)).name
    if src.is_file():
        return src
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        cand = split_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def collect_class_names(split_dirs: list[Path]) -> list[str]:
    names: set[str] = set()
    for d in split_dirs:
        for jp in collect_labelme_json_paths(d):
            try:
                raw = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for sh in raw.get("shapes") or []:
                lab = sh.get("label")
                if lab is not None and str(lab).strip():
                    names.add(str(lab).strip())
    return sorted(names)


def build_coco_from_json_paths(
    json_paths: list[Path],
    out_split_dir: Path,
    name_to_cat_id: dict[str, int],
) -> tuple[int, int, int]:
    """
    Returns (images_written, annotations_written, skipped_json).
    """
    out_split_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    skipped = 0

    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        src_image = resolve_source_image(jp, data)
        if src_image is None:
            skipped += 1
            continue

        img_w = int(data.get("imageWidth") or 0)
        img_h = int(data.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            skipped += 1
            continue

        dest_name = src_image.name
        dest_path = out_split_dir / dest_name
        shutil.copy2(src_image, dest_path)

        images.append(
            {
                "id": image_id,
                "file_name": dest_name,
                "width": img_w,
                "height": img_h,
            }
        )

        for sh in data.get("shapes") or []:
            label = sh.get("label")
            if label is None or not str(label).strip():
                continue
            label = str(label).strip()
            if label not in name_to_cat_id:
                continue
            box = shape_to_xyxy(sh, img_w, img_h)
            if box is None:
                continue
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            area = float(w * h)
            seg = shape_to_segmentation(sh, img_w, img_h)

            ann: dict = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": name_to_cat_id[label],
                "bbox": [round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            }
            if seg is not None:
                ann["segmentation"] = seg
            else:
                ann["segmentation"] = []

            annotations.append(ann)
            ann_id += 1

        image_id += 1

    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for name, cid in sorted(name_to_cat_id.items(), key=lambda x: x[1])
    ]

    coco = {
        "info": {
            "description": "LabelMe to COCO (converted by 04_labelme_to_coco.py)",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json = out_split_dir / "_annotations.coco.json"
    out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return len(images), len(annotations), skipped


def build_coco_for_split(
    split_dir: Path,
    out_split_dir: Path,
    name_to_cat_id: dict[str, int],
) -> tuple[int, int, int]:
    """Process every LabelMe JSON in split_dir."""
    json_paths = collect_labelme_json_paths(split_dir)
    return build_coco_from_json_paths(json_paths, out_split_dir, name_to_cat_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe folders (train/valid/test) → COCO format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATASETS_DIR,
        help="Root containing train/, valid/, and/or test/ with LabelMe .json + images",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATASETS_DIR / "converted_coco_format",
        help="Output root (creates train/, valid/, test/ as needed)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        metavar="NAME",
        help=f"Split folder names under data-root (default: {' '.join(DEFAULT_SPLITS)}). "
        "Existing folders are processed; missing splits are skipped.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when only train/ exists (70/20/10 shuffle split). Ignored otherwise.",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    out_root: Path = args.out_dir.resolve()

    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"
    only_train_on_disk = train_dir.is_dir() and not valid_dir.is_dir() and not test_dir.is_dir()

    if only_train_on_disk:
        json_paths = collect_labelme_json_paths(train_dir)
        if not json_paths:
            print(f"Error: no LabelMe JSON files under {train_dir}", file=sys.stderr)
            return 1

        rng = random.Random(args.seed)
        shuffled = json_paths.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr, n_va, _n_te = partition_counts(n, SPLIT_RATIOS_PCT)
        i1 = n_tr
        i2 = i1 + n_va
        train_paths = shuffled[:i1]
        valid_paths = shuffled[i1:i2]
        test_paths = shuffled[i2:]

        class_names = collect_class_names([train_dir])
        if not class_names:
            print("Warning: no labels found in JSON files; categories will be empty.", file=sys.stderr)
        name_to_cat_id = {name: i + 1 for i, name in enumerate(class_names)}

        r0, r1, r2 = SPLIT_RATIOS_PCT
        print(
            f"Only train/ found: shuffling {n} samples into train/valid/test "
            f"({r0}/{r1}/{r2} %, seed={args.seed})",
        )
        print(f"Categories ({len(class_names)}): {class_names}")
        for split_name, paths in (
            ("train", train_paths),
            ("valid", valid_paths),
            ("test", test_paths),
        ):
            out_split = out_root / split_name
            n_img, n_ann, skip = build_coco_from_json_paths(paths, out_split, name_to_cat_id)
            print(f"{split_name}: images={n_img}, annotations={n_ann}, skipped_json={skip} -> {out_split}")

        print(f"Done. Output root: {out_root}")
        return 0

    # argparse: `--splits` with no values yields [] and would otherwise process nothing.
    split_names: list[str] = list(args.splits) if args.splits else list(DEFAULT_SPLITS)

    split_dirs: list[Path] = []
    for name in split_names:
        d = data_root / name
        if d.is_dir():
            split_dirs.append(d)

    if not split_dirs:
        print(f"Error: none of the split folders exist under {data_root}", file=sys.stderr)
        return 1

    class_names = collect_class_names(split_dirs)
    if not class_names:
        print("Warning: no labels found in JSON files; categories will be empty.", file=sys.stderr)
    # COCO category ids are typically 1..K
    name_to_cat_id = {n: i + 1 for i, n in enumerate(class_names)}

    print(f"Categories ({len(class_names)}): {class_names}")
    for d in split_dirs:
        name = d.name
        out_split = out_root / name
        n_img, n_ann, skip = build_coco_for_split(d, out_split, name_to_cat_id)
        print(f"{name}: images={n_img}, annotations={n_ann}, skipped_json={skip} -> {out_split}")

    print(f"Done. Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
