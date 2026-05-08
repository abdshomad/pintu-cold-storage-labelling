# AI agent instructions (this package)

Follow these rules when changing code, adding scripts, or running commands. They mirror how this project is already structured and documented in [`README.md`](README.md).

## Repository layout (do not break)

The **logical repo root** is the directory that contains these **three sibling folders**:

| Folder | Role |
|--------|------|
| `pintu-cold-storage-labelling/` | Python tooling, `pyproject.toml`, `uv.lock`, numbered pipeline scripts (this directory) |
| `pintu-cold-storage-datasets/` | Video, frames, VOC XML, LabelMe JSON, `converted_yolo_format/`, `converted_coco_format/` |
| `pintu-cold-storage-models/` | Training runs (`yolo-YYYY-MM-DD-vN/`, `rf-detr-YYYY-MM-DD-vN/`, …) |

Python code resolves paths from the script file: **`Path(__file__).resolve().parent.parent`** is this repo root (parent of `pintu-cold-storage-labelling/`). New scripts must keep that invariant unless you intentionally redesign layout (then update README and all path helpers).

Do not move datasets or weights **into** `pintu-cold-storage-labelling/`; they stay in the sibling folders above.

## Python environment: **uv only**

- Manage dependencies with **[uv](https://docs.astral.sh/uv/)** from **this directory** (where `pyproject.toml` lives):
  - Add or upgrade: `uv add <package>`
  - Install locked deps: `uv sync`
- Do **not** hand-edit dependency entries in `pyproject.toml` except coordinated with `uv add` / `uv remove`.
- Do **not** use ad-hoc `pip install` for project dependencies in docs or automation; use `uv sync` / `uv add`.
- Run project Python with the synced env:

  ```bash
  cd pintu-cold-storage-labelling   # this package
  uv run python <script>.py [args...]
  ```

- Training scripts (**05**, **06**) and anything that needs locked deps should be documented and run as `uv run python …`.

## Numbered pipeline scripts

Scripts in **this directory** follow a **fixed pipeline order** via **two-digit prefixes** (`01_` … `06_`):

| Order | File | Notes |
|------:|------|--------|
| 01 | `01_extract_frames.sh` | Bash + `ffmpeg`; only shell step |
| 02 | `02_convert_xml_to_labelme_json.py` | VOC/Datumaro XML → LabelMe JSON |
| 03 | `03_labelme_to_yolo.py` | LabelMe → YOLO (Ultralytics layout) |
| 04 | `04_labelme_to_coco.py` | LabelMe → COCO |
| 05 | `05_train_yolo26_pintu.py` | YOLO training → `pintu-cold-storage-models/` |
| 06 | `06_train_rfdetr_nano_pintu.py` | RF-DETR training → `pintu-cold-storage-models/` |

**Conventions for new steps:**

- Use the **next** free number (`07_`, `08_`, …).
- Keep the **`NN_descriptive_name.py`** (or `.sh`) pattern; name should state what the step does.
- Prefer **flat** placement next to existing scripts (not a deep `src/` tree unless the project is migrated as a whole).
- Update **`README.md`** workflow table, Mermaid diagram, and “Scripts” table when you add or rename steps.

## Internal shared code

- **`_run_utils.py`**: shared helpers for training scripts (**05**, **06**) and similar (run dirs, tee logging, `run_meta.json`). Prefix **`_`** means **internal** to this package; do not treat as a public API for external imports.
- Avoid duplicating `repo_root` / `models_root` / run-folder logic; extend `_run_utils.py` if both training paths need the same behavior.

## Dataset and naming conventions

- Default **datasets root**: `pintu-cold-storage-datasets/` (sibling of `pintu-cold-storage-labelling/`).
- Frames: `train/frame_%06d.png` (six digits), aligned with VOC `<filename>`.
- VOC exports: default input dir `voc_xml/` under datasets (overridable via CLI).
- YOLO export default: `converted_yolo_format/`; COCO: `converted_coco_format/`.
- **Class IDs**: YOLO 0-based from sorted label names; COCO categories `1 … K` (see README).

## Training runs (`pintu-cold-storage-models/`)

- Run folders: `{yolo|rf-detr}-YYYY-MM-DD-vN` with **auto-increment** `vN` per day; do not silently overwrite prior runs.
- Each run should keep **`train.log`** and **`run_meta.json`** in the run directory (existing scripts already do this via `_run_utils.py`).

## Shell vs Python

- **01** is **Bash**. From this directory: `bash 01_extract_frames.sh`. From the logical repo root (parent of this folder): `bash pintu-cold-storage-labelling/01_extract_frames.sh`. On Windows use Git Bash / MSYS2 / WSL; do not assume PowerShell can run it natively.
- Prefer **Python 3** with standard library + `uv`-managed deps for portable steps.
- For long-running loops (especially per-frame video processing/inference), use **`tqdm`** progress bars instead of ad-hoc periodic `print` counters so terminal progress is clear and consistent.

## `main.py`

`main.py` in this directory is a **placeholder** stub, not the data pipeline entrypoint. The pipeline is the **numbered scripts**; do not route real workflow through `main.py` unless the project is explicitly refactored to that model and README is updated.

## Documentation

- User-facing workflow and commands live in **`README.md`**. When behavior or flags change, update README in the same change.
- Keep **AGENTS.md** (this file) accurate when you change repo-wide conventions.
