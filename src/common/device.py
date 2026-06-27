from __future__ import annotations

# I know lab runs across Linux, but I only have MPS. 

import os
import torch

def get_device(prefer: str | None = None) -> torch.device:
    if os.environ.get("PSAI_FORCE_CPU") == "1":
        return torch.device("cpu")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def device_report(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA: {name} ({mem:.1f} GB) — full speed, use this for real fine-tuning."
    if device.type == "mps":
        return ("MPS (Apple Silicon): good for dev + smoke tests. Some ops fall back "
                "to CPU and SAM2 image-encoder fine-tuning will be slow — prototype "
                "here, train the real thing on CUDA.")
    return "CPU: everything works but slowly. Fine for the synthetic smoke test only."
