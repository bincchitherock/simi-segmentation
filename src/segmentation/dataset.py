from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset

def prompt_from_mask(mask: torch.Tensor, kind: str = "point"):
    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return torch.tensor([[[w / 2, h / 2]]]), torch.tensor([[0]]), None
    if kind == "box":
        box = torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]]).float()
        return None, None, box
    j = torch.randint(0, len(xs), (1,)).item()
    pt = torch.tensor([[[xs[j].float(), ys[j].float()]]])
    return pt, torch.tensor([[1]]), None

class MaskDataset(Dataset):
    def __init__(self, data_root: str, pairs: str = "pairs.json", img_size: int = 1024,
                 prompt_kind: str = "point"):
        self.root = data_root
        self.img_size = img_size
        self.prompt_kind = prompt_kind
        with open(os.path.join(data_root, pairs)) as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def _load(self, rel, is_mask):
        from PIL import Image
        import numpy as np
        mode = "L" if is_mask else "RGB"
        img = Image.open(os.path.join(self.root, rel)).convert(mode).resize(
            (self.img_size, self.img_size), resample=(Image.NEAREST if is_mask else Image.BILINEAR))
        arr = torch.from_numpy(np.array(img)).float()
        if is_mask:
            return (arr > 127).float().unsqueeze(0)
        arr = arr.permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (arr - mean) / std

    def __getitem__(self, i):
        p = self.pairs[i]
        image = self._load(p["image"], is_mask=False)
        mask = self._load(p["mask"], is_mask=True)
        pc, pl, box = prompt_from_mask(mask[0], self.prompt_kind)
        return {"image": image, "mask": mask, "point_coords": pc,
                "point_labels": pl, "box": box}

class SyntheticMaskDataset(Dataset):
    def __init__(self, n: int = 8, img_size: int = 128, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, img_size, img_size, generator=g) * 0.3

        self.masks = torch.zeros(n, 1, img_size, img_size)
        for k in range(n):
            x0 = torch.randint(0, img_size // 2, (1,)).item()
            y0 = torch.randint(0, img_size // 2, (1,)).item()
            self.masks[k, 0, y0:y0 + img_size // 3, x0:x0 + img_size // 3] = 1.0

        self.images += self.masks * 2.0

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        pc, pl, box = prompt_from_mask(self.masks[i, 0], "point")
        return {"image": self.images[i], "mask": self.masks[i],
                "point_coords": pc, "point_labels": pl, "box": box}
