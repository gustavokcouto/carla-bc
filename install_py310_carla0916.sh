#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON:-$PROJECT_DIR/.venv_gpu/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found at $PYTHON_BIN" >&2
    echo "Set PYTHON=/path/to/python or create .venv_gpu with Python 3.10." >&2
    exit 1
fi

echo "Using: $("$PYTHON_BIN" --version)"
"$PYTHON_BIN" -m pip install --upgrade pip

# Gym 0.21 does not build with recent setuptools/wheel defaults.
# Keep the legacy Gym API used by stable-baselines3 1.6.x and this codebase.
"$PYTHON_BIN" -m pip install "setuptools==65.5.0" "wheel==0.38.4"
"$PYTHON_BIN" -m pip install --no-build-isolation "gym==0.21.0"

# requirements.txt intentionally does not pin torch/torchvision, so a working
# CUDA-enabled PyTorch installation is preserved.
"$PYTHON_BIN" -m pip install -r requirements.txt

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as m
import carla
import gym
import torch

print("CARLA:", m.version("carla"))
print("Gym:", gym.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
