"""
Internal helpers shared by the training scripts (05_train_*, 06_train_*).

Provides:
  - repo_root() / models_root(): canonical project paths.
  - make_run_dir(kind, base, date): create the next per-day versioned run dir
    (e.g. pintu-cold-storage-models/yolo-2026-05-08-v1).
  - Tee / tee_to(path): mirror sys.stdout + sys.stderr to a log file while
    still printing to the terminal.
  - write_run_meta(...): dump run_meta.json with args, timing, env info.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_root() -> Path:
    return repo_root() / "pintu-cold-storage-models"


def _next_version(base: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}v(\d+)$")
    max_seen = 0
    if base.is_dir():
        for entry in base.iterdir():
            m = pattern.match(entry.name)
            if m:
                n = int(m.group(1))
                if n > max_seen:
                    max_seen = n
    return max_seen + 1


def make_run_dir(
    kind: str,
    base: Path | None = None,
    date: str | None = None,
) -> Path:
    """Create and return `<base>/<kind>-<date>-v<N>` with N = max existing + 1."""
    base = (base or models_root()).resolve()
    base.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now().strftime("%Y-%m-%d")
    prefix = f"{kind}-{date}-"
    n = _next_version(base, prefix)
    run_dir = base / f"{kind}-{date}-v{n}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class Tee:
    """File-like object that forwards writes to multiple underlying streams."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for s in self._streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                pass
        return False

    def fileno(self) -> int:
        for s in self._streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("Tee has no underlying fileno")


@contextlib.contextmanager
def tee_to(path: Path) -> Iterator[Path]:
    """Append-tee stdout+stderr to `path`. Restores streams on exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(path, "a", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, log_fp)
    sys.stderr = Tee(orig_err, log_fp)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            log_fp.flush()
        finally:
            log_fp.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def write_run_meta(
    run_dir: Path,
    *,
    kind: str,
    args: dict,
    start: datetime,
    end: datetime,
    status: str,
    extra: dict | None = None,
) -> Path:
    """Write `run_meta.json` inside `run_dir` and return its path."""
    meta = {
        "kind": kind,
        "args": _jsonable(args),
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "duration_seconds": int((end - start).total_seconds()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "run_dir": str(run_dir),
        "status": status,
    }
    if extra:
        meta["extra"] = _jsonable(extra)
    out = run_dir / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
