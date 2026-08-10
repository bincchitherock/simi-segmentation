"""
Picks what hardware to run on.

Set `PSAI_FORCE_CPU=1` to pin everything to the CPU. That is slower, but two runs
with the same seed then give the same answer, which is what the tests rely on.
"""

from __future__ import annotations

import os

import torch


def _is_available(kind: str) -> bool:
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return mps is not None and mps.is_available()
    return False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the fastest hardware available, or the one named in `prefer`."""
    if os.environ.get("PSAI_FORCE_CPU") == "1":
        return torch.device("cpu")
    if prefer is not None:
        dev = torch.device(prefer)
        if not _is_available(dev.type):
            raise RuntimeError(f"requested device {prefer!r} is not available on this machine")
        return dev
    for kind in ("cuda", "mps"):
        if _is_available(kind):
            return torch.device(kind)
    return torch.device("cpu")


def device_report(device: torch.device) -> str:
    """One line telling you what you are running on and what to expect from it."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA: {name} ({mem:.1f} GB). Full speed, use this for real training."
    if device.type == "mps":
        return ("MPS (Apple Silicon): fine for development and quick checks. Some steps "
                "fall back to the CPU, training SAM 2 is slow, and repeat runs do not "
                "match exactly. Prototype here, then train and report on CUDA.")
    return "CPU: everything works, slowly, and two runs match exactly. The tests use this."


__all__ = ["get_device", "device_report"]
