"""
The two segmenters, and how they are wrapped so training treats them alike.

`TinyUNetSegmenter` is small, starts from nothing, and is what actually runs here.
Every number in the README came from it.

`SAM2Segmenter` wraps Meta's SAM 2. It is not installed and has never been run in
this repo, so treat the code as a sketch, not as something known to work. It says so
loudly at runtime. If you want to bring it up, install the package, fetch a
checkpoint, and expect to fix things.

Both are used the same way. Give them an image and a prompt, get back a score per
pixel saying how likely that pixel is part of the instrument.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.config import SegConfig
from src.common.lora import (LoRASpec, apply_lora, count_trainable,
                             mark_only_lora_trainable)
from src.segmentation.dataset import PROMPT_KEYS

SAM2_INSTALL_HINT = (
    "SAM2 is not installed. Install it with\n"
    "    pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
    "and fetch a checkpoint, e.g.\n"
    "    curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt\n"
    "then set `checkpoint: <path>.pt` and `model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml`\n"
    "(model_cfg is a path *inside the sam2 package*, resolved by hydra).\n"
    "For a run that needs no checkpoint, use `model: tinyunet`."
)

SAM2_DECODER_ALSO_TRAIN = ("sam_mask_decoder.output_hypernetworks_mlps",
                           "sam_mask_decoder.iou_prediction_head")


class SegmenterModule(nn.Module):
    """
    What both segmenters agree to provide, so the training loop can ignore the difference.

    `backend` is a short name for logs and figures. `adapter_root` is the part that
    gets saved and reloaded. `forward` takes an image and a prompt and returns a score
    per pixel.
    """

    backend: str
    lora_spec: LoRASpec

    @property
    def adapter_root(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, images: torch.Tensor, point_coords: torch.Tensor | None = None,
                point_labels: torch.Tensor | None = None,
                boxes: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class SAM2Segmenter(SegmenterModule):
    """
    Meta's SAM 2 with small trainable adapters added. Never run here.

    The original weights stay frozen and only the adapters and the output layers
    train, which is what makes fine-tuning a model this size practical at all.
    """

    def __init__(self, checkpoint: str, model_cfg: str, spec: LoRASpec,
                 train_decoder: bool = True, device: str = "cpu") -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise ImportError(SAM2_INSTALL_HINT) from exc

        print("[SAM2Segmenter] UNVERIFIED PATH. I wrote this against facebookresearch/sam2 "
              "@ main and never ran it, because sam2 is not installed here.")

        self.model = build_sam2(model_cfg, checkpoint, device=device,
                                apply_postprocessing=False)
        self.model.sam_mask_decoder.dynamic_multimask_via_stability = False

        self.lora_spec = spec
        n_wrapped = apply_lora(self.model, spec)
        also_train = SAM2_DECODER_ALSO_TRAIN if train_decoder else ()
        n_tensors = mark_only_lora_trainable(self.model, also_train=also_train)

        self.backend = f"sam2({model_cfg})"
        self.input_size = self.model.image_size

        tr, tot = count_trainable(self.model)
        print(f"[SAM2Segmenter] LoRA on {n_wrapped} linears, {n_tensors} trainable "
              f"tensors | {tr/1e6:.2f}M / {tot/1e6:.1f}M params ({100*tr/tot:.2f}%) | "
              f"input {self.input_size}px, {self.model.hidden_dim}-d features")

    @property
    def adapter_root(self) -> nn.Module:
        return self.model

    def _image_features(self, images: torch.Tensor
                        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        backbone_out = self.model.forward_image(images)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        b = images.shape[0]
        feats = [f.permute(1, 2, 0).view(b, -1, *size)
                 for f, size in zip(vision_feats, feat_sizes)]
        return feats[-1], feats[:-1]

    def _sparse_points(self, point_coords: torch.Tensor | None,
                       point_labels: torch.Tensor | None,
                       boxes: torch.Tensor | None
                       ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if boxes is None:
            return point_coords, point_labels
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int,
                                  device=boxes.device).repeat(boxes.shape[0], 1)
        if point_coords is None:
            return box_coords, box_labels
        return (torch.cat([box_coords, point_coords], dim=1),
                torch.cat([box_labels, point_labels.int()], dim=1))

    def forward(self, images: torch.Tensor, point_coords: torch.Tensor | None = None,
                point_labels: torch.Tensor | None = None,
                boxes: torch.Tensor | None = None) -> torch.Tensor:
        h, w = images.shape[-2:]
        if (h, w) != (self.input_size, self.input_size):
            raise ValueError(
                f"SAM2 expects {self.input_size}x{self.input_size} input (and prompt "
                f"coordinates in that space), got {h}x{w}; set img_size: "
                f"{self.input_size} in the config")
        if point_coords is None and boxes is None:
            raise ValueError("SAM2Segmenter needs exactly one prompt for every image. "
                             "It has no way to guess where to look on its own")

        image_embed, high_res_features = self._image_features(images)
        coords, labels = self._sparse_points(point_coords, point_labels, boxes)
        sparse, dense = self.model.sam_prompt_encoder(points=(coords, labels),
                                                      boxes=None, masks=None)
        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,  # one prompt per image, so nothing needs duplicating
            high_res_features=high_res_features,
        )
        return F.interpolate(low_res_masks, size=(h, w), mode="bilinear",
                             align_corners=False)


def prompt_channel(shape: torch.Size, point_coords: torch.Tensor | None,
                   point_labels: torch.Tensor | None,
                   boxes: torch.Tensor | None) -> torch.Tensor:
    """
    Paint the prompt into an extra image channel, so TinyUNet can see it.

    SAM 2 has a dedicated path for prompts. TinyUNet does not, so I drew the prompt as
    a picture instead: a soft bright spot where you clicked, or a filled rectangle for
    a box, stacked onto the image as a fourth channel.
    """
    b, _, h, w = shape
    if point_coords is not None and point_labels is None:
        raise ValueError("point_coords needs point_labels alongside it. Without them "
                         "I could not tell a click on the instrument from one off it")
    if point_coords is None and boxes is None:
        raise ValueError("no prompt given")
    device = point_coords.device if point_coords is not None else boxes.device
    out = torch.zeros(b, 1, h, w, device=device)

    if boxes is not None:
        for i, (x0, y0, x1, y1) in enumerate(boxes.round().long().tolist()):
            out[i, 0, max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1.0

    if point_coords is not None:
        sigma = 0.05 * max(h, w)
        ys = torch.arange(h, device=device).view(1, 1, h, 1)
        xs = torch.arange(w, device=device).view(1, 1, 1, w)
        px = point_coords[..., 0].view(b, -1, 1, 1)
        py = point_coords[..., 1].view(b, -1, 1, 1)
        bump = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        sign = point_labels.float().view(b, -1, 1, 1)
        sign = torch.where(sign == 1, 1.0, torch.where(sign == 0, -1.0, 0.0))
        out = out + (bump * sign).sum(dim=1, keepdim=True)

    return out.clamp(-1.0, 1.0)


class _DoubleConv(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        groups = math.gcd(8, co)
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(groups, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(groups, co), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNetSegmenter(SegmenterModule):
    """
    A small segmenter, about 30 thousand values, trained from scratch. This is the one
    that runs here.

    It narrows the image down, thinks, then widens it back out to full size, carrying
    the early detail across so edges stay sharp. Every score in the README came from
    this, so read them as TinyUNet numbers and not as SAM 2 numbers.
    """

    def __init__(self, spec: LoRASpec, base: int = 16, init_seed: int = 0,
                 train_decoder: bool = True) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.enc1 = _DoubleConv(4, base)  # 3 colour channels, plus the prompt as a 4th
            self.enc2 = _DoubleConv(base, base * 2)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(base * 2, base * 2)
            self.proj = nn.Linear(base * 2, base * 2)
            self.dec = _DoubleConv(base * 3, base)
            self.head = nn.Conv2d(base, 1, 1)
            n_wrapped = apply_lora(self, spec)

        self.backend = "tinyunet"
        self.lora_spec = spec
        # `train_decoder` means the same thing on both segmenters: train the output
        # layers as well as the adapters.
        n_tensors = mark_only_lora_trainable(
            self, also_train=("head", "dec") if train_decoder else ())
        tr, tot = count_trainable(self)
        print(f"[TinyUNetSegmenter] LoRA on {n_wrapped} linears, {n_tensors} trainable "
              f"tensors | {tr/1e3:.1f}K / {tot/1e3:.1f}K params "
              f"| decoder {'trained' if train_decoder else 'frozen'}")

    @property
    def adapter_root(self) -> nn.Module:
        return self

    def forward(self, images: torch.Tensor, point_coords: torch.Tensor | None = None,
                point_labels: torch.Tensor | None = None,
                boxes: torch.Tensor | None = None) -> torch.Tensor:
        if point_coords is None and boxes is None:
            raise ValueError("TinyUNetSegmenter needs a point or box prompt")
        x = torch.cat([images, prompt_channel(images.shape, point_coords,
                                              point_labels, boxes)], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b, c, h, w = e2.shape
        z = e2.flatten(2).transpose(1, 2)
        z = self.proj(F.gelu(self.fc1(z)))
        e2 = z.transpose(1, 2).view(b, c, h, w)
        # Resize to match e1 exactly rather than simply doubling, so odd-sized images
        # still line up.
        up = F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec(torch.cat([up, e1], dim=1)))


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                  smooth: float = 1.0) -> torch.Tensor:
    """
    The training loss, two parts added together.

    One part judges each pixel on its own. The other judges the shape as a whole. The
    first alone does badly when the instrument is small, because calling everything
    background is then nearly right. The second keeps that honest.
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and target "
                         f"{tuple(target.shape)} must have the same shape")
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    dice = 1 - (2 * inter + smooth) / (prob.sum(dims) + target.sum(dims) + smooth)
    return bce + dice.mean()


def forward_batch(model: SegmenterModule, batch: dict, device: torch.device) -> torch.Tensor:
    """Move one batch onto the device and run it, whichever prompt kind it carries."""
    prompts = {k: batch[k].to(device) if batch[k] is not None else None
               for k in PROMPT_KEYS}
    return model(batch["image"].to(device), point_coords=prompts["point_coords"],
                 point_labels=prompts["point_labels"], boxes=prompts["box"])


def build_segmenter(cfg: SegConfig, device: torch.device) -> SegmenterModule:
    """Build whichever segmenter the config asks for."""
    spec_kwargs = dict(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    if cfg.model == "tinyunet":
        return TinyUNetSegmenter(LoRASpec(targets=("",), **spec_kwargs),
                                 init_seed=cfg.seed,
                                 train_decoder=cfg.train_decoder).to(device)
    if cfg.model == "sam2":
        spec = LoRASpec(targets=("image_encoder", "sam_mask_decoder"), **spec_kwargs)
        return SAM2Segmenter(checkpoint=cfg.checkpoint, model_cfg=cfg.model_cfg,
                             spec=spec, train_decoder=cfg.train_decoder,
                             device=str(device)).to(device)
    raise ValueError(f"unknown model {cfg.model!r}")


__all__ = ["SAM2_DECODER_ALSO_TRAIN", "SAM2_INSTALL_HINT",
           "SegmenterModule", "SAM2Segmenter", "TinyUNetSegmenter",
           "prompt_channel", "dice_bce_loss", "forward_batch", "build_segmenter"]
