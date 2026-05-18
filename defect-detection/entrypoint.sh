#!/bin/sh
set -e

# Render sets RENDER=true; LOW_MEMORY may come from Dockerfile or dashboard.
# A copied .env with IMAGE_SIZE=224 does not affect getenv here, but if the
# container is missing IMAGE_SIZE entirely, default to 128 for bootstrap.
_renderish() {
  case "${RENDER:-}" in 1|true|TRUE|yes|YES) return 0 ;; esac
  case "${LOW_MEMORY:-}" in 1|true|TRUE|yes|YES) return 0 ;; esac
  return 1
}
if _renderish; then
  export IMAGE_SIZE="${IMAGE_SIZE:-128}"
  export BOOTSTRAP_TRAIN_IMAGES="${BOOTSTRAP_TRAIN_IMAGES:-6}"
fi

python scripts/bootstrap_data.py 2>/dev/null || true

if [ ! -f "${MODEL_MEMORY_BANK_PATH:-/app/artifacts/patchcore_memory_bank.pt}" ]; then
  echo "No memory bank found — training on bootstrap images..."
  python -m training.train_patchcore || echo "Training skipped (no images yet)."
fi

exec "$@"
