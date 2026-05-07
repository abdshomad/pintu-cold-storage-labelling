#!/usr/bin/env bash
# Extract every video frame as PNG matching train/frame_######.png (6 digits).
# Matches <filename> in existing VOC annotations (frame_000000.png, …).
#
# Prerequisites: ffmpeg in PATH. Run from Git Bash, MSYS2, or WSL (any cwd).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASETS_DIR="${REPO_ROOT}/pintu-cold-storage-datasets"
VIDEO="${DATASETS_DIR}/Video Pintu Cold Storage Terbuka Tertutup ada Tirainya.mp4"
OUT_DIR="${DATASETS_DIR}/train"

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: Video not found:" >&2
  echo "  ${VIDEO}" >&2
  echo "Place the video under pintu-cold-storage-datasets/ with this exact filename." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# -vsync passthrough avoids dropping duplicated frames vs default vfr cap
ffmpeg -nostdin -y -i "$VIDEO" \
  -vsync passthrough \
  -f image2 -start_number 0 \
  -c:v png \
  "${OUT_DIR}/frame_%06d.png"

echo "Frames written under: ${OUT_DIR}/frame_<6-digit>.png"
