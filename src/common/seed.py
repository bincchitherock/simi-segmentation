from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

MAX_SEED = 2 ** 32


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
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
    """A CPU generator for DataLoader shuffling and dataset-owned randomness."""
    return torch.Generator().manual_seed(int(seed) % MAX_SEED)


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`"""
    del worker_id
    s = torch.initial_seed() % MAX_SEED
    random.seed(s)
    np.random.seed(s)


def derive_seed(*parts: int | str) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % MAX_SEED


__all__ = ["MAX_SEED", "seed_everything", "make_generator", "seed_worker", "derive_seed"]
