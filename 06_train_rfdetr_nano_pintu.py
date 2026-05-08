"""
Fine-tune RF-DETR Nano on the pintu cold storage COCO (Roboflow-style) export.

Expected layout (see https://rfdetr.roboflow.com/learn/train/#dataset-structure):
  converted_coco_format/
    train/_annotations.coco.json  (+ images)
    valid/_annotations.coco.json  (+ images)
    test/_annotations.coco.json   (+ images)  # optional if run_test is disabled

Outputs go to:
  ../pintu-cold-storage-models/rf-detr-YYYY-MM-DD-vN/
where N auto-increments per day. Each run dir also contains train.log
(teed stdout+stderr) and run_meta.json (args, timing, env, status).

Install deps in this folder with uv (adds to pyproject.toml / uv.lock):
  uv add rfdetr

Train (uses the project venv):
  uv run python 06_train_rfdetr_nano_pintu.py

Optional TensorBoard / W&B extras:
  uv add "rfdetr[metrics]"
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

from _run_utils import make_run_dir, models_root, tee_to, write_run_meta


def default_dataset_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "pintu-cold-storage-datasets" / "converted_coco_format"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF-DETR Nano on pintu COCO (Roboflow layout).")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir(),
        help="Root with train/valid[/test] each containing _annotations.coco.json",
    )
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
        help="Subfolder name under --models-dir (default: auto rf-detr-YYYY-MM-DD-vN)",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4, help="Per-step batch (pair with --grad-accum-steps for effective batch)")
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or cuda:0")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pth to resume")
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument(
        "--run-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate on test split each epoch (requires test/_annotations.coco.json)",
    )
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    return p.parse_args()


def assert_coco_splits(root: Path) -> None:
    train_ann = root / "train" / "_annotations.coco.json"
    valid_ann = root / "valid" / "_annotations.coco.json"
    if not train_ann.is_file():
        raise FileNotFoundError(
            f"Missing {train_ann}. Export COCO with 04_labelme_to_coco.py (train split)."
        )
    if not valid_ann.is_file():
        raise FileNotFoundError(
            f"Missing {valid_ann}. RF-DETR expects a **valid** folder (not val). "
            "Re-run: python 04_labelme_to_coco.py ... --splits train valid test "
            "so converted_coco_format/valid/_annotations.coco.json exists."
        )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    assert_coco_splits(dataset_dir)

    test_ann = dataset_dir / "test" / "_annotations.coco.json"
    run_test = bool(args.run_test)
    if run_test and not test_ann.is_file():
        warnings.warn(
            f"No {test_ann}; disabling run_test (validation still runs). "
            "Add a test split or pass --no-run-test explicitly.",
            stacklevel=1,
        )
        run_test = False

    try:
        from rfdetr import RFDETRNano
    except ImportError as e:
        raise SystemExit(
            "Missing package rfdetr. From pintu-cold-storage-labelling/: uv add rfdetr && uv sync"
        ) from e

    models_dir = args.models_dir.resolve()
    if args.run_name:
        run_dir = (models_dir / args.run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir("rf-detr", base=models_dir)

    train_kw: dict = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_dir),
        "dataset_file": "roboflow",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "device": args.device,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "run_test": run_test,
        "tensorboard": args.tensorboard,
        "wandb": args.wandb,
        "checkpoint_interval": args.checkpoint_interval,
    }
    if args.wandb:
        train_kw["project"] = args.wandb_project
        train_kw["run"] = args.wandb_run

    start = datetime.now()
    status = "ok"
    try:
        with tee_to(run_dir / "train.log"):
            print(f"[run_dir] {run_dir}")
            model = RFDETRNano()
            model.train(**train_kw)
    except BaseException as e:
        status = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        end = datetime.now()
        write_run_meta(
            run_dir,
            kind="rf-detr",
            args=vars(args),
            start=start,
            end=end,
            status=status,
            extra={"dataset_dir": str(dataset_dir)},
        )


if __name__ == "__main__":
    main()
