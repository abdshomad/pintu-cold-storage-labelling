"""
Fine-tune Ultralytics YOLO26n on the pintu cold storage YOLO dataset.

Requires a recent ultralytics release that ships YOLO26 (e.g. pip install -U ultralytics).

Outputs go to:
  ../pintu-cold-storage-models/yolo-YYYY-MM-DD-vN/
where N auto-increments per day. Each run dir also contains train.log
(teed stdout+stderr) and run_meta.json (args, timing, env, status).

Note: data.yaml currently sets val to the same folder as train; for honest metrics,
split some images into images/val and update val: in data.yaml.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _run_utils import make_run_dir, models_root, tee_to, write_run_meta


def default_data_yaml() -> Path:
    # Script: .../pintu-cold-storage-labelling/05_train_yolo26_pintu.py → dataset is sibling of this folder
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "pintu-cold-storage-datasets" / "converted_yolo_format" / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26n on pintu cold storage (YOLO format).")
    p.add_argument(
        "--data",
        type=Path,
        default=default_data_yaml(),
        help="Path to data.yaml (default: ../pintu-cold-storage-datasets/converted_yolo_format/data.yaml from repo root)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Ultralytics pretrained weights or YAML (default: yolo26n.pt)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Use -1 for auto batch")
    p.add_argument("--device", type=str, default=None, help="e.g. 0, cpu, 0,1 (default: ultralytics auto)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--models-dir",
        type=Path,
        default=models_root(),
        help="Root for run dirs (default: ../pintu-cold-storage-models)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder name under --models-dir (default: auto yolo-YYYY-MM-DD-vN)",
    )
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Resume last interrupted run in models-dir/run-name")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Automatic mixed precision")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Install ultralytics first: pip install -U ultralytics torch torchvision\n"
            "YOLO26 needs a recent ultralytics version that includes yolo26n.pt."
        ) from e

    models_dir = args.models_dir.resolve()
    if args.run_name:
        run_dir = (models_dir / args.run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir("yolo", base=models_dir)

    train_kw: dict = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(models_dir),
        "name": run_dir.name,
        "exist_ok": True,
        "patience": args.patience,
        "seed": args.seed,
        "resume": args.resume,
        "amp": args.amp,
    }
    if args.device is not None:
        train_kw["device"] = args.device

    start = datetime.now()
    status = "ok"
    try:
        with tee_to(run_dir / "train.log"):
            print(f"[run_dir] {run_dir}")
            model = YOLO(args.model)
            model.train(**train_kw)
    except BaseException as e:
        status = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        end = datetime.now()
        write_run_meta(
            run_dir,
            kind="yolo",
            args=vars(args),
            start=start,
            end=end,
            status=status,
            extra={"model_weights": args.model, "data": str(data)},
        )


if __name__ == "__main__":
    main()
