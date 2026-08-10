from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.config import SegConfig
from src.common.seed import derive_seed, make_generator
from src.data.splits import SplitError, resolve_splits

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PROMPT_KEYS = ("point_coords", "point_labels", "box")


def prompt_from_mask(mask: torch.Tensor, kind: str, generator: torch.Generator
                     ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor],
                                Optional[torch.Tensor]]:
    if mask.ndim != 2:
        raise ValueError(f"expected an (H, W) mask, got {tuple(mask.shape)}")
    if kind not in ("point", "box"):
        raise ValueError(f"prompt_kind must be 'point' or 'box', got {kind!r}")

    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        raise ValueError("cannot prompt an empty mask: a prompted segmenter needs "
                         "something to point at (MaskDataset drops these pairs)")

    if kind == "box":
        return None, None, torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).float()
    j = int(torch.randint(0, len(xs), (1,), generator=generator).item())
    return torch.stack([xs[j], ys[j]]).float().unsqueeze(0), torch.tensor([1]), None


def to_display(image: torch.Tensor) -> np.ndarray:
    """Invert `MaskDataset`'s normalisation: (3, H, W) tensor -> (H, W, 3) in [0, 1]."""
    return (image * IMAGENET_STD + IMAGENET_MEAN).permute(1, 2, 0).clamp(0, 1).numpy()


def read_pairs(data_root: str, pairs: str = "pairs.json") -> list[dict[str, Any]]:
    """`pairs.json` as a list of `{image, mask, video_id, ...}` records."""
    with open(Path(data_root) / pairs) as f:
        records = json.load(f)
    missing = [r for r in records if "video_id" not in r]
    if missing:
        raise SplitError(f"{Path(data_root) / pairs}: {len(missing)} entries have no "
                         "'video_id'; splits must be video-level, not per-frame")
    return records


def resolve_seg_splits(cfg: SegConfig) -> dict[str, list[str]]:
    """Train/val (and test, when declared) video ids"""
    available = sorted({r["video_id"] for r in read_pairs(cfg.data_root)})
    return resolve_splits(available=available, splits_file=cfg.splits,
                          data_root=cfg.data_root,
                          explicit={"train": cfg.train_split, "val": cfg.val_split,
                                    "test": cfg.test_split})


class MaskDataset(Dataset):
    """`pairs.json` entries `{image, mask, video_id}` for `split`"""
    def __init__(self, data_root: str, split: Sequence[str], *, pairs: str = "pairs.json",
                 img_size: int = 1024, prompt_kind: str = "point", seed: int = 0,
                 name: str = "data") -> None:
        self.root = Path(data_root)
        self.img_size = img_size
        self.prompt_kind = prompt_kind
        self.seed = seed
        self.name = name

        all_pairs = read_pairs(data_root, pairs)
        in_split = [p for p in all_pairs if p["video_id"] in set(split)]
        if not in_split:
            raise SplitError(f"{self.root / pairs}: split {list(split)} matched none of "
                             f"{sorted({p['video_id'] for p in all_pairs})}")
        self.pairs = [p for p in in_split if self._mask_has_foreground(p["mask"])]
        if not self.pairs:
            raise SplitError(f"{self.root / pairs}: every mask in split {list(split)} is "
                             "empty, so there is nothing to prompt")

        n_dropped = len(in_split) - len(self.pairs)
        print(f"[MaskDataset:{name}] {len(self.pairs)} pairs from "
              f"{len(set(split))} videos at {img_size}px, {prompt_kind} prompts "
              f"({n_dropped} of {len(in_split)} dropped: no instrument in view, so no "
              "prompt exists)")

    def _mask_has_foreground(self, rel: str) -> bool:
        with Image.open(self.root / rel) as img:
            return bool(np.any(np.asarray(img.convert("L")) > 127))

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, rel: str, is_mask: bool) -> torch.Tensor:
        with Image.open(self.root / rel) as raw:
            img = raw.convert("L" if is_mask else "RGB").resize(
                (self.img_size, self.img_size),
                resample=Image.NEAREST if is_mask else Image.BILINEAR)
            arr = torch.from_numpy(np.asarray(img).copy()).float()
        if is_mask:
            return (arr > 127).float().unsqueeze(0)
        return (arr.permute(2, 0, 1) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, i: int) -> dict[str, Any]:
        p = self.pairs[i]
        image = self._load(p["image"], is_mask=False)
        mask = self._load(p["mask"], is_mask=True)
        gen = make_generator(derive_seed(self.seed, self.name, i))
        pc, pl, box = prompt_from_mask(mask[0], self.prompt_kind, gen)
        item = {"image": image, "mask": mask, "pair_id": p["image"]}
        if box is None:
            item["point_coords"], item["point_labels"] = pc, pl
        else:
            item["box"] = box
        return item


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"image": torch.stack([b["image"] for b in batch]),
                 "mask": torch.stack([b["mask"] for b in batch]),
                 "pair_id": [b["pair_id"] for b in batch]}
    for key in PROMPT_KEYS:
        present = [b[key] for b in batch if key in b]
        if present and len(present) != len(batch):
            raise ValueError(f"batch mixes prompt kinds: {key} is set for "
                             f"{len(present)}/{len(batch)} samples")
        out[key] = torch.stack(present) if present else None
    return out


__all__ = ["PROMPT_KEYS", "MaskDataset", "collate", "prompt_from_mask", "read_pairs",
           "resolve_seg_splits", "to_display"]
