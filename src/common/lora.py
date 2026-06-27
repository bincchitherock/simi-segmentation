from __future__ import annotations

import math
import re
from typing import Iterable

import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

def inject_lora(
    module: nn.Module,
    target_patterns: Iterable[str] = (r"(?:^|\.)qkv$", r"(?:^|\.)q_proj$", r"(?:^|\.)k_proj$",
                                      r"(?:^|\.)v_proj$", r"(?:^|\.)out_proj$", r"(?:^|\.)proj$",
                                      r"(?:^|\.)fc1$", r"(?:^|\.)fc2$",
                                      r"(?:^|\.)lin1$", r"(?:^|\.)lin2$"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    patterns = [re.compile(p) for p in target_patterns]
    wrapped = 0
    for name, child in list(module.named_modules()):
        for cname, c in list(child.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(c, nn.Linear) and any(p.search(full) for p in patterns):
                setattr(child, cname, LoRALinear(c, rank=rank, alpha=alpha, dropout=dropout))
                wrapped += 1
    return wrapped

def mark_only_lora_trainable(module: nn.Module, also_train: Iterable[str] = ()) -> None:
    also = tuple(also_train)
    for n, p in module.named_parameters():
        train = ("lora_A" in n) or ("lora_B" in n) or any(a in n for a in also)
        p.requires_grad_(train)

def lora_state_dict(module: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in module.named_parameters()
            if ("lora_A" in n) or ("lora_B" in n)}

def count_trainable(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
