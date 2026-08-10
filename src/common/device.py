"""Device selection"""

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
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA: {name} ({mem:.1f} GB) — full speed, use this for real fine-tuning."
    if device.type == "mps":
        return ("MPS (Apple Silicon): good for dev + smoke tests. Some ops fall back to CPU, "
                "SAM2 image-encoder fine-tuning is slow, and Metal kernels are not "
                "bit-reproducible — prototype here, train and report on CUDA.")
    return "CPU: everything works but slowly, and reproducibly. Used by the tests."


__all__ = ["get_device", "device_report"]
