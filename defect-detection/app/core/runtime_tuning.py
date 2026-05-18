from __future__ import annotations

import gc
import os


def configure_runtime_memory() -> None:
    """Apply CPU/thread settings suitable for 512 MB containers (e.g. Render free)."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    gc.collect()
