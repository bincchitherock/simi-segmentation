"""Seeding helpers: global seeding, standalone generators, and per-item seeds."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common.seed import derive_seed, make_generator, seed_everything, seed_worker


def draw():
    return (torch.rand(4).tolist(), np.random.rand(4).tolist())


def test_seed_everything_makes_torch_and_numpy_reproducible():
    seed_everything(1234)
    first = draw()
    seed_everything(1234)
    assert draw() == first


def test_seed_everything_rejects_out_of_range():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_make_generator_is_reproducible_and_independent_of_global_state():
    a = torch.randn(8, generator=make_generator(7))
    torch.manual_seed(999)  # perturb the global stream
    b = torch.randn(8, generator=make_generator(7))
    assert torch.equal(a, b)


def test_derive_seed_is_stable_and_distinct_per_part():
    """derive_seed is stable across processes, unlike hash()."""
    assert derive_seed(0, "val", 3) == derive_seed(0, "val", 3)
    assert derive_seed(0, "val", 3) != derive_seed(0, "train", 3)
    assert derive_seed(0, "val", 3) != derive_seed(0, "val", 4)
    assert 0 <= derive_seed(0, "val", 3) < 2 ** 32


def test_seed_worker_reseeds_python_and_numpy_from_the_torch_seed():
    torch.manual_seed(42)
    seed_worker(0)
    first = np.random.rand(3).tolist()
    torch.manual_seed(42)
    seed_worker(0)
    assert np.random.rand(3).tolist() == first
