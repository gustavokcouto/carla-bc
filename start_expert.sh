#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$PROJECT_DIR/.venv_gpu/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv_gpu/bin/python"
else
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python virtual environment not found." >&2
    echo "Expected .venv_gpu/bin/python or .venv/bin/python, or set PYTHON=/path/to/python." >&2
    exit 1
fi

"$PYTHON_BIN" -c "import carla, importlib.metadata as m; print('Python:', __import__('sys').version.split()[0]); print('CARLA Python client:', m.version('carla'))"

exec screen -L -S carla_expert "$PYTHON_BIN" data_collect.py
