#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual environment not found at $PYTHON" >&2
    exit 1
fi

"$PYTHON" -c "import carla, importlib.metadata as m; print('CARLA Python client:', m.version('carla'))"

exec screen -L -S carla_expert "$PYTHON" data_collect.py
