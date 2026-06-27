from __future__ import annotations

import torch
import torch.nn as nn

class FrameEncoder(nn.Module):

    def __init__(self, backbone: str = "timm:convnext_tiny", pretrained: bool = True,
                 freeze: bool = False):
        super().__init__()
        self.backbone_name = backbone
        if backbone.startswith("timm:"):
            import timm
            name = backbone.split(":", 1)[1]
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                         global_pool="avg")
            self.feat_dim = self.net.num_features
            self._mode = "timm"
        elif backbone == "sam2":

            self.net = None
            self.feat_dim = None
            self._mode = "sam2"
        else:
            raise ValueError(f"Unknown backbone spec: {backbone!r}")

        if freeze and self.net is not None:
            for p in self.net.parameters():
                p.requires_grad_(False)

    def set_sam2_encoder(self, encoder: nn.Module, feat_dim: int, pool: str = "avg"):
        self.net = encoder
        self.feat_dim = feat_dim
        self._sam2_pool = pool
        self._mode = "sam2"

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "timm":
            return self.net(x)
        if self._mode == "sam2":
            if self.net is None:
                raise RuntimeError("sam2 backbone selected but encoder not injected. "
                                   "Call set_sam2_encoder() from the segmentation builder.")
            feats = self.net(x)
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            return feats
        raise RuntimeError("encoder not configured")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t = x.shape[:2]
            feats = self._encode_frames(x.flatten(0, 1))
            return feats.view(b, t, -1)
        return self._encode_frames(x)
