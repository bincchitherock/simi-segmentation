"""
Turns one video frame into a list of numbers the phase model can work with.

The phase model never looks at raw pixels. I wrapped an off-the-shelf image network
from the `timm` library so that it hands back one vector per frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """
    One vector per frame, taken from a `timm` image network with its classifier removed.

    Give it a batch of images shaped (batch, colour, height, width) and you get back
    (batch, feat_dim). Give it a batch of clips shaped (batch, time, colour, height,
    width) and you get back (batch, time, feat_dim). `feat_dim` is fixed once the
    encoder is built, so the temporal model knows how wide to be.
    """

    def __init__(self, backbone: str = "timm:convnext_tiny", pretrained: bool = True,
                 freeze: bool = False) -> None:
        super().__init__()
        if not backbone.startswith("timm:"):
            raise ValueError(
                f"unknown backbone {backbone!r}. Expected 'timm:<model_name>'. "
                "I removed the 'sam2' option, because SAM 2 hands back a stack of "
                "feature maps rather than the single vector this needs.")

        import timm

        self.backbone_name = backbone
        self.net = timm.create_model(backbone.split(":", 1)[1], pretrained=pretrained,
                                     num_classes=0, global_pool="avg")
        feat_dim = int(getattr(self.net, "num_features", 0))
        if feat_dim <= 0:
            raise RuntimeError(f"{backbone!r} did not report how wide its output is "
                               f"(got {feat_dim!r}), so I could not size the model after it")
        self.feat_dim: int = feat_dim

        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()

    def train(self, mode: bool = True) -> "FrameEncoder":
        """A frozen encoder stays in eval mode even when the model around it trains."""
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
