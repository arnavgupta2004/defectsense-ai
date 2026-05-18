#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/defect-detection"

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print(sys.version_info[:2] >= (3, 11))')"
    if [ "$ver" = "True" ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.11+ is required. Install with: brew install python@3.12"
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env

if [ ! -f artifacts/patchcore_memory_bank.pt ]; then
  echo "Bootstrapping training data and building memory bank (first run)..."
  python scripts/bootstrap_data.py
  python -m training.train_patchcore
fi

echo "Starting API on http://127.0.0.1:8000"
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
