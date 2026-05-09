#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SETUP="${CITYWALKER_CONDA_SETUP:-/media/isee324/2a90eb70-2d62-4af4-b04a-0fcdde4122a5/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CITYWALKER_CONDA_ENV:-dino}"

if [ -f "$CONDA_SETUP" ]; then
  source "$CONDA_SETUP"
  conda activate "$CONDA_ENV"
else
  echo "Conda setup file not found: $CONDA_SETUP" >&2
  exit 1
fi

export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3.8/dist-packages:${PYTHONPATH}"
exec python "$SCRIPT_DIR/inference_node.py"
