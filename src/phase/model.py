"""
The phase network: what step is happening, and which instruments are in view.

It works in three parts. An image encoder turns each frame into a vector on its own.
A temporal model then looks along the clip, so a frame is read in the light of what
came before and after it. Two small heads read off the answers, one picking a single
step and one flagging any number of instruments at once.

The middle part comes in two flavours, set by `temporal` in the config. `tcn` slides
filters along time and is the default. `transformer` lets every frame look at every
other frame directly, which is more flexible and needs more data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.backbone import FrameEncoder


def _dilations(n_layers: int, clip_len: int) -> list[int]:
    """
    How far apart each layer spaces the frames it looks at.

    The gaps double layer by layer, 1, 2, 4, 8, so a stack of a few layers can see
    across a long clip without needing a filter that wide. I capped the doubling at the
    clip length, then started over from 1, since a gap wider than the clip sees nothing.
    """
    max_exp = max(1, int(math.log2(max(2, clip_len // 2))) + 1)
    return [2 ** (i % max_exp) for i in range(n_layers)]


class DilatedResidualLayer(nn.Module):
    """One layer. It adds to what came in rather than replacing it, which trains better."""

    def __init__(self, dilation: int, ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.pointwise = nn.Conv1d(ch, ch, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.pointwise(self.act(self.conv(x)))
        return (x + self.drop(out)) * mask


class SingleStageTCN(nn.Module):
    """A stack of those layers, each looking further along the clip than the last."""

    def __init__(self, in_ch: int, ch: int, n_layers: int = 8, dropout: float = 0.1,
                 clip_len: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, ch, kernel_size=1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(d, ch, dropout) for d in _dilations(n_layers, clip_len)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x) * mask
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerTemporal(nn.Module):
    """The other option: every frame can look at every other frame in the clip."""

    def __init__(self, in_ch: int, ch: int, n_layers: int = 4, n_heads: int = 4,
                 dropout: float = 0.1, clip_len: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_ch, ch)
        self.pos = nn.Parameter(torch.zeros(1, clip_len, ch))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(ch, n_heads, ch * 4, dropout,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        if t > self.pos.shape[1]:
            raise ValueError(f"this clip is {t} frames long but the model was built for "
                             f"{self.pos.shape[1]}. Rebuild it with clip_len={t}")
        valid = mask.squeeze(1) > 0
        h = self.in_proj(x.transpose(1, 2)) + self.pos[:, :t]
        h = self.encoder(h, src_key_padding_mask=~valid)
        return h.transpose(1, 2) * mask


@dataclass(frozen=True)
class PhaseModelConfig:
    """
    Only the settings the network itself needs.

    You almost certainly want `PhaseConfig` in `src/common/config.py` instead. That is
    the one the YAML files fill in and the one to edit. `from_config` below copies the
    shared fields across from it when the model gets built.
    """

    backbone: str = "timm:convnext_tiny"
    pretrained: bool = True
    freeze_backbone: bool = False
    temporal: str = "tcn"
    temporal_ch: int = 256
    temporal_layers: int = 8
    n_heads: int = 4
    dropout: float = 0.1
    num_steps: int = 14
    num_instruments: int = 18
    instrument_loss_weight: float = 1.0
    clip_len: int = 64
    step_class_weight: tuple[float, ...] = ()
    instrument_pos_weight: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.temporal not in ("tcn", "transformer"):
            raise ValueError(f"temporal must be 'tcn' or 'transformer', got {self.temporal!r}")
        if self.temporal == "transformer" and self.temporal_ch % self.n_heads:
            raise ValueError(f"temporal_ch ({self.temporal_ch}) must be divisible by "
                             f"n_heads ({self.n_heads})")
        if self.step_class_weight and len(self.step_class_weight) != self.num_steps:
            raise ValueError(f"step_class_weight has {len(self.step_class_weight)} entries, "
                             f"expected num_steps={self.num_steps}")
        if self.instrument_pos_weight and len(self.instrument_pos_weight) != self.num_instruments:
            raise ValueError(f"instrument_pos_weight has {len(self.instrument_pos_weight)} "
                             f"entries, expected num_instruments={self.num_instruments}")

    @classmethod
    def from_config(cls, cfg: Any, **extra: Any) -> "PhaseModelConfig":
        """Copy across every field this shares with `cfg`. `extra` wins over both."""
        shared = {f.name: getattr(cfg, f.name) for f in fields(cls) if hasattr(cfg, f.name)}
        return cls(**{**shared, **extra})


def _weight_tensor(values: tuple[float, ...]) -> Optional[torch.Tensor]:
    return torch.tensor(values, dtype=torch.float32) if values else None


class SpatioTemporalMultiTask(nn.Module):
    """
    The whole thing: encode each frame, look along time, then read off both answers.

    Two heads share one body. The step head picks exactly one of the steps, since only
    one can be happening. The instrument head judges each instrument separately, since
    several can be in view at the same time.
    """

    def __init__(self, cfg: PhaseModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg.backbone, cfg.pretrained, cfg.freeze_backbone)
        d = self.encoder.feat_dim
        weights = "pretrained" if cfg.pretrained else "scratch"
        self.backend = f"{cfg.backbone}({weights})+{cfg.temporal}"

        if cfg.temporal == "tcn":
            self.temporal: nn.Module = SingleStageTCN(d, cfg.temporal_ch, cfg.temporal_layers,
                                                      cfg.dropout, cfg.clip_len)
        else:
            self.temporal = TransformerTemporal(d, cfg.temporal_ch, cfg.temporal_layers,
                                                cfg.n_heads, cfg.dropout, cfg.clip_len)

        self.step_head = nn.Conv1d(cfg.temporal_ch, cfg.num_steps, 1)
        self.instr_head = nn.Conv1d(cfg.temporal_ch, cfg.num_instruments, 1)

        # Registered rather than stored plainly, so moving the model to the GPU moves
        # these along with it.
        self.register_buffer("step_weight", _weight_tensor(cfg.step_class_weight))
        self.register_buffer("instr_pos_weight", _weight_tensor(cfg.instrument_pos_weight))

    def _encode(self, clip: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Encode the real frames only. Short clips are padded, and padding is skipped."""
        if valid.all():
            return self.encoder(clip)
        flat = self.encoder(clip[valid])  # (n_valid, D)
        out = flat.new_zeros(clip.shape[0], clip.shape[1], flat.shape[-1])
        out[valid] = flat
        return out

    def forward(self, clip: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if valid is None:
            valid = torch.ones(clip.shape[:2], dtype=torch.bool, device=clip.device)
        if not valid.any():
            raise ValueError("clip batch has no valid frames")
        feats = self._encode(clip, valid).transpose(1, 2)
        mask = valid.unsqueeze(1).to(feats.dtype)
        h = self.temporal(feats, mask)
        return {"step": self.step_head(h).transpose(1, 2),
                "instrument": self.instr_head(h).transpose(1, 2)}

    def compute_loss(self, out: dict[str, torch.Tensor], step_target: torch.Tensor,
                     instr_target: torch.Tensor,
                     valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Add up both heads' losses. Padded frames are left out of both."""
        if not valid.any():
            raise ValueError("batch contains no valid frames")
        step_logits = out["step"][valid]
        instr_logits = out["instrument"][valid]
        step_loss = F.cross_entropy(step_logits, step_target[valid].long(),
                                    weight=self.step_weight)
        instr_loss = F.binary_cross_entropy_with_logits(
            instr_logits, instr_target[valid].float(), pos_weight=self.instr_pos_weight)
        total = step_loss + self.cfg.instrument_loss_weight * instr_loss
        return total, {"step": step_loss.item(), "instrument": instr_loss.item()}


__all__ = ["PhaseModelConfig", "SpatioTemporalMultiTask", "SingleStageTCN",
           "TransformerTemporal", "DilatedResidualLayer"]
