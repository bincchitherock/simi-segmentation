from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.lora import inject_lora, mark_only_lora_trainable, count_trainable

class SegmenterModule(nn.Module):
    feat_dim: int = 0

    def forward(self, images, point_coords=None, point_labels=None, boxes=None):
        raise NotImplementedError

    def image_encoder_module(self) -> nn.Module:
        raise NotImplementedError

class SAM2Segmenter(SegmenterModule):

    def __init__(self, checkpoint: str, model_cfg: str, lora_rank: int = 8,
                 lora_alpha: int = 16, train_decoder: bool = True, device="cpu"):
        super().__init__()
        from sam2.build_sam import build_sam2

        self.model = build_sam2(model_cfg, checkpoint, device=device)

        n_enc = inject_lora(self.model.image_encoder, rank=lora_rank, alpha=lora_alpha)
        n_dec = inject_lora(self.model.sam_mask_decoder, rank=lora_rank, alpha=lora_alpha)

        mark_only_lora_trainable(
            self.model,
            also_train=("sam_mask_decoder.output_hypernetworks",
                        "sam_mask_decoder.iou_prediction_head") if train_decoder else (),
        )
        tr, tot = count_trainable(self.model)
        print(f"[SAM2Segmenter] LoRA wrapped enc={n_enc} dec={n_dec} layers | "
              f"trainable {tr/1e6:.2f}M / {tot/1e6:.1f}M params "
              f"({100*tr/tot:.2f}%)")

        self.feat_dim = getattr(self.model.image_encoder, "neck_channels", 256)

    def image_encoder_module(self) -> nn.Module:
        return self.model.image_encoder

    def forward(self, images, point_coords=None, point_labels=None, boxes=None):

        backbone_out = self.model.forward_image(images)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        image_embed = vision_feats[-1].permute(1, 2, 0).view(
            images.shape[0], -1, *backbone_out["vision_features"].shape[-2:])
        sparse, dense = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None else None,
            boxes=boxes, masks=None,
        )
        low_res_masks, iou_pred, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
        )
        return F.interpolate(low_res_masks, size=images.shape[-2:],
                             mode="bilinear", align_corners=False)

class _DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())

    def forward(self, x):
        return self.net(x)

class DummySegmenter(SegmenterModule):

    def __init__(self, base=16, lora_rank=4):
        super().__init__()
        self.enc1 = _DoubleConv(3, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(base * 2, base * 2)
        self.proj = nn.Linear(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = _DoubleConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.feat_dim = base * 2
        n = inject_lora(self, rank=lora_rank, alpha=lora_rank * 2)
        mark_only_lora_trainable(self, also_train=("head", "dec"))
        tr, tot = count_trainable(self)
        print(f"[DummySegmenter] LoRA wrapped {n} linears | trainable "
              f"{tr/1e3:.1f}K / {tot/1e3:.1f}K params")

    def image_encoder_module(self):
        return nn.Sequential(self.enc1, self.pool, self.enc2)

    def forward(self, images, point_coords=None, point_labels=None, boxes=None):
        e1 = self.enc1(images)
        e2 = self.enc2(self.pool(e1))
        b, c, h, w = e2.shape
        z = e2.flatten(2).transpose(1, 2)
        z = self.proj(F.gelu(self.fc1(z)))
        e2 = z.transpose(1, 2).view(b, c, h, w)
        d = self.dec(torch.cat([self.up(e2), e1], 1))
        return self.head(d)

def dice_bce_loss(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    prob = logits.sigmoid()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    dice = 1 - (2 * inter + eps) / (prob.sum(dims) + target.sum(dims) + eps)
    return bce + dice.mean()

def build_segmenter(args, device) -> SegmenterModule:
    if getattr(args, "synthetic", False) or not getattr(args, "checkpoint", None):
        return DummySegmenter(lora_rank=getattr(args, "lora_rank", 4)).to(device)
    return SAM2Segmenter(checkpoint=args.checkpoint, model_cfg=args.model_cfg,
                         lora_rank=args.lora_rank, device=str(device)).to(device)
