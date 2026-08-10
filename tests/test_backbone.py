from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.common.backbone import FrameEncoder


def test_feat_dim_is_always_a_positive_int():
    enc = FrameEncoder("timm:convnext_atto", pretrained=False)
    assert isinstance(enc.feat_dim, int) and enc.feat_dim > 0


def test_forward_shapes():
    enc = FrameEncoder("timm:convnext_atto", pretrained=False).eval()
    assert enc(torch.randn(2, 3, 32, 32)).shape == (2, enc.feat_dim)
    assert enc(torch.randn(2, 4, 3, 32, 32)).shape == (2, 4, enc.feat_dim)


def test_non_timm_backbone_spec_raises():
    with pytest.raises(ValueError, match="timm:"):
        FrameEncoder("sam2", pretrained=False)


def test_frozen_backbone_keeps_batchnorm_statistics_fixed():
    parent = nn.Sequential(FrameEncoder("timm:resnet18", pretrained=False, freeze=True))
    enc = parent[0]
    bn = next(m for m in enc.net.modules() if isinstance(m, nn.BatchNorm2d))
    before = bn.running_mean.clone()

    parent.train()
    enc(torch.randn(2, 3, 64, 64))

    assert torch.equal(bn.running_mean, before), "a frozen backbone drifted"
    assert bn.num_batches_tracked.item() == 0
    assert not any(p.requires_grad for p in enc.net.parameters())
