"""
Per-frame image encoder for the phase model. `timm:<name>` backbones only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """
    Wraps a timm classifier as a pooled feature extractor.
    - Accepts (B, C, H, W) or (B, T, C, H, W); the latter returns (B, T, feat_dim).
    - `feat_dim` is a positive int known at construction.
    """

    def __init__(self, backbone: str = "timm:convnext_tiny", pretrained: bool = True,
                 freeze: bool = False) -> None:
        super().__init__()
        if not backbone.startswith("timm:"):
            raise ValueError(
                f"unknown backbone spec {backbone!r}; expected 'timm:<model_name>'. "
                "The 'sam2' branch was removed (dead code, and SAM2's image encoder "
                "returns a dict of feature maps, not a pooled vector).")

        import timm

        self.backbone_name = backbone
        self.net = timm.create_model(backbone.split(":", 1)[1], pretrained=pretrained,
                                     num_classes=0, global_pool="avg")
        feat_dim = int(getattr(self.net, "num_features", 0))
        if feat_dim <= 0:
            raise RuntimeError(f"{backbone!r} did not report a usable num_features "
                               f"({feat_dim!r}); cannot size the temporal stack")
        self.feat_dim: int = feat_dim

        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()

    def train(self, mode: bool = True) -> "FrameEncoder":
        """A frozen backbone stays in eval mode even when the parent calls train()."""
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                feats = self.net(x)
        else:
            feats = self.net(x)
        if feats.ndim != 2 or feats.shape[1] != self.feat_dim:
            raise RuntimeError(f"{self.backbone_name} returned {tuple(feats.shape)}, "
                               f"expected (N, {self.feat_dim})")
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t = x.shape[:2]
            return self._encode(x.flatten(0, 1)).view(b, t, self.feat_dim)
        if x.ndim == 4:
            return self._encode(x)
        raise ValueError(f"expected (B,C,H,W) or (B,T,C,H,W), got {tuple(x.shape)}")


__all__ = ["FrameEncoder"]
