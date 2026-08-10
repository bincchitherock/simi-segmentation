"""
Makes a run repeatable.

Training uses random numbers in a lot of places: how weights start out, what order
the data arrives in, which pixel gets picked as a click. Fixing the seed pins all of
them, so re-running gives the same answer and a change in the score comes from the
change you made rather than from luck.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

MAX_SEED = 2 ** 32


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, numpy and torch at once. Call this first, before anything else."""
    if not 0 <= seed < MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}), got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_generator(seed: int) -> torch.Generator:
    """A random number source for shuffling, and for choices a dataset makes itself."""
    return torch.Generator().manual_seed(int(seed) % MAX_SEED)


def seed_worker(worker_id: int) -> None:
    """Seed a data-loading worker process. Each one gets its own, so they do not repeat."""
    del worker_id
    s = torch.initial_seed() % MAX_SEED
    random.seed(s)
    np.random.seed(s)


def derive_seed(*parts: int | str) -> int:
    """Build one stable seed out of several labels, such as a run seed and an item index."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % MAX_SEED


__all__ = ["MAX_SEED", "seed_everything", "make_generator", "seed_worker", "derive_seed"]
