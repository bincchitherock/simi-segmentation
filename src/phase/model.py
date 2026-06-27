from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.backbone import FrameEncoder

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, ch: int, dropout: float = 0.1):
        super().__init__()

        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.pointwise = nn.Conv1d(ch, ch, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.pointwise(self.act(self.conv(x)))
        return x + self.drop(out)

class SingleStageTCN(nn.Module):
    def __init__(self, in_ch: int, ch: int, n_layers: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, ch, kernel_size=1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2 ** i, ch, dropout) for i in range(n_layers)]
        )

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerTemporal(nn.Module):
    def __init__(self, in_ch: int, ch: int, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, ch)
        self.pos = nn.Parameter(torch.zeros(1, max_len, ch))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(ch, n_heads, ch * 4, dropout,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.in_proj(x) + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return x.transpose(1, 2)

@dataclass
class PhaseModelConfig:
    backbone: str = "timm:convnext_tiny"
    pretrained: bool = True
    freeze_backbone: bool = False
    temporal: str = "tcn"
    temporal_ch: int = 256
    temporal_layers: int = 8
    num_steps: int = 14
    num_instruments: int = 18
    instrument_loss_weight: float = 1.0
    dropout: float = 0.1

class SpatioTemporalMultiTask(nn.Module):
    def __init__(self, cfg: PhaseModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg.backbone, cfg.pretrained, cfg.freeze_backbone)

        d = self.encoder.feat_dim or cfg.temporal_ch

        if cfg.temporal == "tcn":
            self.temporal = SingleStageTCN(d, cfg.temporal_ch, cfg.temporal_layers, cfg.dropout)
        elif cfg.temporal == "transformer":
            self.temporal = TransformerTemporal(d, cfg.temporal_ch, cfg.temporal_layers,
                                                dropout=cfg.dropout)
        else:
            raise ValueError(cfg.temporal)

        self.step_head = nn.Conv1d(cfg.temporal_ch, cfg.num_steps, 1)
        self.instr_head = nn.Conv1d(cfg.temporal_ch, cfg.num_instruments, 1)

    def forward(self, clip: torch.Tensor):
        if clip.ndim == 5:
            feats = self.encoder(clip)
        else:
            feats = clip
        feats = feats.transpose(1, 2)
        h = self.temporal(feats)
        step_logits = self.step_head(h).transpose(1, 2)
        instr_logits = self.instr_head(h).transpose(1, 2)
        return {"step": step_logits, "instrument": instr_logits}

    def compute_loss(self, out, step_target, instr_target, ignore_index: int = -100):

        B, T, S = out["step"].shape
        step_loss = F.cross_entropy(
            out["step"].reshape(B * T, S),
            step_target.reshape(B * T).long(),
            ignore_index=ignore_index,
        )
        instr_loss = F.binary_cross_entropy_with_logits(
            out["instrument"], instr_target.float()
        )
        total = step_loss + self.cfg.instrument_loss_weight * instr_loss
        return total, {"step": step_loss.item(), "instrument": instr_loss.item()}
