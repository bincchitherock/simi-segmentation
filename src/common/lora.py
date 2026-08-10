"""
LoRA adapters, plus a save format that records how they were built.

Fine-tuning every weight of a big model is slow and needs a lot of memory. Instead
I froze the original weights and bolted a small trainable detour onto some of the
linear layers. Only the detour is trained and saved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

ADAPTER_FORMAT = "psai-lora"
ADAPTER_VERSION = 1

DEFAULT_TARGET_PATTERNS: tuple[str, ...] = (
    r"(?:^|\.)qkv$", r"(?:^|\.)q_proj$", r"(?:^|\.)k_proj$", r"(?:^|\.)v_proj$",
    r"(?:^|\.)out_proj$", r"(?:^|\.)proj$", r"(?:^|\.)fc1$", r"(?:^|\.)fc2$",
    r"(?:^|\.)lin1$", r"(?:^|\.)lin2$",
)


class LoRAError(RuntimeError):
    """Raised when adapters cannot be built, saved, or loaded."""


class LoRALinear(nn.Module):
    """A frozen Linear layer with a small trainable detour added to its output."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise LoRAError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if rank <= 0:
            raise LoRAError(f"rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


@dataclass(frozen=True)
class LoRASpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    targets: tuple[str, ...] = ("",)
    target_patterns: tuple[str, ...] = DEFAULT_TARGET_PATTERNS

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
                "targets": list(self.targets), "target_patterns": list(self.target_patterns)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoRASpec":
        unknown = set(data) - {"rank", "alpha", "dropout", "targets", "target_patterns"}
        if unknown:
            raise LoRAError(f"unknown LoRASpec fields: {sorted(unknown)}")
        spec = cls(rank=int(data["rank"]), alpha=int(data["alpha"]),
                   dropout=float(data.get("dropout", 0.0)))
        return replace(spec,
                       targets=tuple(data.get("targets", spec.targets)),
                       target_patterns=tuple(data.get("target_patterns",
                                                      spec.target_patterns)))


@dataclass(frozen=True)
class LoRALoadReport:
    loaded: int
    missing: tuple[str, ...]
    not_in_payload: tuple[str, ...]

    def describe(self) -> str:
        return (f"loaded {self.loaded} adapter tensors "
                f"({len(self.missing)} unmatched in file, "
                f"{len(self.not_in_payload)} trainable tensors left at init)")


def apply_lora(root: nn.Module, spec: LoRASpec) -> int:
    patterns = [re.compile(p) for p in spec.target_patterns]
    total = 0
    for target in spec.targets:
        try:
            sub = root.get_submodule(target)
        except AttributeError as exc:
            raise LoRAError(f"LoRA target {target!r} is not a submodule of "
                            f"{type(root).__name__}") from exc
        if any(isinstance(m, LoRALinear) for m in sub.modules()):
            raise LoRAError(f"LoRA target {target!r} already contains adapters; "
                            "apply_lora must not be called twice on the same module")

        wrapped = 0
        # I listed the children first. Replacing one mid-loop would revisit the new wrapper.
        for parent_name, parent in list(sub.named_modules()):
            for child_name, child in list(parent.named_children()):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                if isinstance(child, nn.Linear) and any(p.search(full) for p in patterns):
                    setattr(parent, child_name,
                            LoRALinear(child, rank=spec.rank, alpha=spec.alpha,
                                       dropout=spec.dropout))
                    wrapped += 1
        if wrapped == 0:
            raise LoRAError(
                f"LoRA target {target!r} matched 0 nn.Linear layers with patterns "
                f"{list(spec.target_patterns)}; nothing would train")
        total += wrapped
    return total


def lora_param_names(root: nn.Module) -> set[str]:
    names: set[str] = set()
    for mod_name, mod in root.named_modules():
        if isinstance(mod, LoRALinear):
            for p_name, _ in mod.named_parameters():
                if p_name.startswith(("lora_A.", "lora_B.")):
                    names.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


def mark_only_lora_trainable(root: nn.Module, also_train: Sequence[str] = ()) -> int:
    lora = lora_param_names(root)
    if not lora:
        raise LoRAError(f"{type(root).__name__} has no LoRA adapters; "
                        "call apply_lora() before mark_only_lora_trainable()")

    names = [n for n, _ in root.named_parameters()]
    extra = {prefix: {n for n in names if n == prefix or n.startswith(prefix + ".")}
             for prefix in also_train}
    dead = sorted(p for p, matches in extra.items() if not matches)
    if dead:
        raise LoRAError(f"also_train paths matched no parameters: {dead}. "
                        "They are dotted module paths relative to the module passed in.")

    trainable = lora.union(*extra.values()) if extra else lora
    for name, param in root.named_parameters():
        param.requires_grad_(name in trainable)
    return len(trainable)


def count_trainable(module: nn.Module) -> tuple[int, int]:
    """Count how many parameter values are trainable, and how many there are in total."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def adapter_state_dict(root: nn.Module, spec: LoRASpec,
                       root_name: str = "") -> dict[str, Any]:
    """Save every trainable parameter, named relative to `root`."""
    tensors = {n: p.detach().cpu().clone()
               for n, p in root.named_parameters() if p.requires_grad}
    if not tensors:
        raise LoRAError(f"{type(root).__name__} has no trainable parameters to save")
    return {"format": ADAPTER_FORMAT, "version": ADAPTER_VERSION,
            "root": root_name or type(root).__name__,
            "spec": spec.to_dict(), "tensors": tensors}


def load_adapter_state_dict(root: nn.Module, payload: Mapping[str, Any],
                            strict: bool = True) -> LoRALoadReport:
    """Copy a saved payload back into `root`."""
    if payload.get("format") != ADAPTER_FORMAT:
        raise LoRAError(f"not a {ADAPTER_FORMAT} adapter payload "
                        f"(format={payload.get('format')!r})")
    if payload.get("version") != ADAPTER_VERSION:
        raise LoRAError(f"adapter format version {payload.get('version')!r} "
                        f"!= supported {ADAPTER_VERSION}")
    tensors: Mapping[str, torch.Tensor] = payload["tensors"]

    dest = dict(root.named_parameters())
    matched = [k for k in tensors if k in dest]
    if not matched:
        raise LoRAError(
            f"none of the {len(tensors)} saved tensors matched {type(root).__name__} "
            f"(payload root={payload.get('root')!r}). "
            f"saved e.g. {sorted(tensors)[:2]}; module expects e.g. {sorted(dest)[:2]}. "
            "Load into the same module the checkpoint was saved from, and inject "
            "adapters with the payload's spec first.")

    with torch.no_grad():
        for key in matched:
            src, dst = tensors[key], dest[key]
            if src.shape != dst.shape:
                raise LoRAError(f"shape mismatch for {key}: checkpoint "
                                f"{tuple(src.shape)} vs module {tuple(dst.shape)}. "
                                "The rank you added probably differs from the saved one.")
            dst.copy_(src.to(dst.device, dst.dtype))

    missing = tuple(sorted(k for k in tensors if k not in dest))
    trainable = {n for n, p in root.named_parameters() if p.requires_grad}
    not_in_payload = tuple(sorted(trainable - set(tensors)))
    if strict and missing:
        raise LoRAError(f"{len(missing)} saved tensors have no destination in "
                        f"{type(root).__name__}, e.g. {list(missing[:3])}")
    return LoRALoadReport(len(matched), missing, not_in_payload)


def load_adapter_file(root: nn.Module, path: str, strict: bool = True) -> LoRALoadReport:
    """Read an adapter file from disk and apply it."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return load_adapter_state_dict(root, payload, strict=strict)


def read_adapter_spec(path: str) -> LoRASpec:
    """Read only the settings, so the adapters can be added before their weights load."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != ADAPTER_FORMAT:
        raise LoRAError(f"{path}: not a {ADAPTER_FORMAT} adapter payload")
    return LoRASpec.from_dict(payload["spec"])


__all__ = ["ADAPTER_FORMAT", "ADAPTER_VERSION", "DEFAULT_TARGET_PATTERNS", "LoRAError",
           "LoRALinear", "LoRASpec", "LoRALoadReport", "apply_lora", "lora_param_names",
           "mark_only_lora_trainable", "count_trainable", "adapter_state_dict",
           "load_adapter_state_dict", "load_adapter_file", "read_adapter_spec"]
