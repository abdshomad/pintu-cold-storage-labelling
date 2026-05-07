#!/usr/bin/env python3
"""
Convert MIT LabelMe / Datumaro-style XML annotations to wkentaro labelme JSON.

Input XML: <annotation> with <filename>, <imagesize><nrows>/<ncols>, and
<object> entries with <name>, <deleted>, <type>, <polygon><pt><x>/<y>.

Output: same structure as labelme LabelFile.write_label_file (see labelme repo).
https://github.com/wkentaro/labelme
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Sibling folder: raw video, frames, XML, and converted exports live here.
DATASETS_DIR = Path(__file__).resolve().parent.parent / "pintu-cold-storage-datasets"


def _parse_rotation(attributes_text: str | None) -> float:
    if not attributes_text:
        return 0.0
    m = re.search(r"rotation\s*=\s*([-\d.]+)", attributes_text.strip())
    if not m:
        return 0.0
    return float(m.group(1))


def _bbox_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _labelme_version() -> str:
    try:
        import labelme

        return str(labelme.__version__)
    except Exception:
        return "5.0.0"


def xml_to_labelme_dict(
    xml_path: Path,
    *,
    image_root: Path | None = None,
    embed_image: bool = False,
) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_el = root.find("filename")
    if filename_el is None or not (filename_el.text or "").strip():
        raise ValueError(f"missing <filename> in {xml_path}")
    image_path = (filename_el.text or "").strip()

    nrows_el = root.find("./imagesize/nrows")
    ncols_el = root.find("./imagesize/ncols")
    if nrows_el is None or ncols_el is None:
        raise ValueError(f"missing <imagesize> in {xml_path}")
    image_height = int(float((nrows_el.text or "0").strip()))
    image_width = int(float((ncols_el.text or "0").strip()))

    shapes: list[dict] = []
    for obj in root.findall("object"):
        deleted = (obj.findtext("deleted") or "0").strip()
        if deleted in ("1", "true", "True"):
            continue

        name = (obj.findtext("name") or "").strip()
        if not name:
            continue

        oid = obj.findtext("id")
        group_id: int | None
        if oid is not None and str(oid).strip().isdigit():
            group_id = int(str(oid).strip())
        else:
            group_id = None

        obj_type = (obj.findtext("type") or "").strip()
        poly = obj.find("polygon")
        if poly is None:
            continue
        points: list[list[float]] = []
        for pt in poly.findall("pt"):
            xe = pt.find("x")
            ye = pt.find("y")
            if xe is None or ye is None:
                continue
            points.append([float((xe.text or "0").strip()), float((ye.text or "0").strip())])

        if len(points) < 1:
            continue

        rotation = _parse_rotation(obj.findtext("attributes"))
        shape_type: str
        out_points: list[list[float]]

        if obj_type == "bounding_box" and len(points) == 4 and abs(rotation) < 1e-6:
            xmin, ymin, xmax, ymax = _bbox_from_points(points)
            shape_type = "rectangle"
            out_points = [[xmin, ymin], [xmax, ymax]]
        else:
            shape_type = "polygon"
            out_points = points

        shapes.append(
            {
                "label": name,
                "points": out_points,
                "group_id": group_id,
                "shape_type": shape_type,
                "flags": {},
                "description": "",
                "mask": None,
            }
        )

    image_data: str | None = None
    if embed_image:
        search = [xml_path.parent]
        if image_root is not None:
            search.append(image_root)
        img_bytes: bytes | None = None
        for base in search:
            cand = base / image_path
            if cand.is_file():
                img_bytes = cand.read_bytes()
                break
        if img_bytes is None:
            raise FileNotFoundError(
                f"embed-image: could not find image {image_path!r} under {search}"
            )
        image_data = base64.b64encode(img_bytes).decode("ascii")

    return {
        "version": _labelme_version(),
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.replace("\\", "/"),
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def convert_file(
    xml_path: Path,
    out_path: Path,
    *,
    image_root: Path | None,
    embed_image: bool,
) -> None:
    data = xml_to_labelme_dict(xml_path, image_root=image_root, embed_image=embed_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    default_xml_dir = DATASETS_DIR / "voc_xml"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_xml_dir,
        help=f"XML file or directory of .xml (default: {default_xml_dir})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path: directory for batch, or folder for single-file JSON. "
        f"Default: {DATASETS_DIR / 'train'} (LabelMe JSON next to frames).",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Base directory to resolve imagePath when using --embed-image",
    )
    p.add_argument(
        "--embed-image",
        action="store_true",
        help="Set imageData to base64 of the image (searches XML dir then --image-root)",
    )
    args = p.parse_args()

    inp: Path = args.input
    if not inp.exists():
        print(f"not found: {inp}", file=sys.stderr)
        return 1

    xml_files: list[Path]
    if inp.is_file():
        if inp.suffix.lower() != ".xml":
            print("input file must be .xml", file=sys.stderr)
            return 1
        xml_files = [inp]
        out_base = args.output
        if out_base is None:
            out_base = DATASETS_DIR / "train" / (inp.stem + ".json")
        else:
            out_base.mkdir(parents=True, exist_ok=True)
            out_base = out_base / (inp.stem + ".json")
        convert_file(
            inp,
            out_base,
            image_root=args.image_root,
            embed_image=args.embed_image,
        )
        print(out_base)
        return 0

    xml_files = sorted(inp.rglob("*.xml"))
    if not xml_files:
        print(f"no .xml under {inp}", file=sys.stderr)
        return 1

    out_dir = args.output
    if out_dir is None:
        out_dir = DATASETS_DIR / "train"

    for xf in xml_files:
        rel = xf.relative_to(inp)
        out_path = out_dir / rel.with_suffix(".json")
        convert_file(
            xf,
            out_path,
            image_root=args.image_root,
            embed_image=args.embed_image,
        )
    print(f"wrote {len(xml_files)} json file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
