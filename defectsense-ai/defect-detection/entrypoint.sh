#!/bin/sh
set -e

python scripts/bootstrap_data.py 2>/dev/null || true

if [ ! -f "${MODEL_MEMORY_BANK_PATH:-/app/artifacts/patchcore_memory_bank.pt}" ]; then
  echo "No memory bank found — training on bootstrap images..."
  python -m training.train_patchcore || echo "Training skipped (no images yet)."
fi

exec "$@"
