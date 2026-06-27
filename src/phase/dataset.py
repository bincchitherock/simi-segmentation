from __future__ import annotations

import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    def __init__(self, data_root: str, manifest: str = "videos.json",
                 clip_len: int = 64, stride: int = 32, img_size: int = 224,
                 num_instruments: int = 18, transform=None, split: Optional[list] = None):
        self.root = data_root
        self.clip_len = clip_len
        self.img_size = img_size
        self.num_instruments = num_instruments
        self.transform = transform
        with open(os.path.join(data_root, manifest)) as f:
            videos = json.load(f)
        if split is not None:
            videos = [v for v in videos if v["video_id"] in split]

        self.videos = videos
        self.index = []
        for vi, v in enumerate(videos):
            n = len(v["steps"])
            for start in range(0, max(1, n - clip_len + 1), stride):
                self.index.append((vi, start))

    def __len__(self):
        return len(self.index)

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (arr - mean) / std

    def __getitem__(self, i):
        vi, start = self.index[i]
        v = self.videos[vi]
        frames_dir = os.path.join(self.root, v["frames_dir"])
        files = sorted(os.listdir(frames_dir))
        end = start + self.clip_len
        clip_files = files[start:end]
        imgs = torch.stack([self._load_image(os.path.join(frames_dir, f)) for f in clip_files])
        steps = torch.tensor(v["steps"][start:end], dtype=torch.long)
        instr = torch.tensor(v["instruments"][start:end], dtype=torch.float32)

        if imgs.shape[0] < self.clip_len:
            pad = self.clip_len - imgs.shape[0]
            imgs = torch.cat([imgs, imgs[-1:].repeat(pad, 1, 1, 1)], 0)
            steps = torch.cat([steps, torch.full((pad,), -100)], 0)
            instr = torch.cat([instr, torch.zeros(pad, self.num_instruments)], 0)
        if self.transform:
            imgs = self.transform(imgs)
        return {"clip": imgs, "step": steps, "instrument": instr, "video_id": v["video_id"]}

class SyntheticClipDataset(Dataset):
    def __init__(self, n: int = 8, clip_len: int = 32, img_size: int = 64,
                 num_steps: int = 14, num_instruments: int = 18, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.clips = torch.randn(n, clip_len, 3, img_size, img_size, generator=g)
        self.steps = torch.randint(0, num_steps, (n, clip_len), generator=g)
        self.instr = (torch.rand(n, clip_len, num_instruments, generator=g) > 0.7).float()

    def __len__(self):
        return self.clips.shape[0]

    def __getitem__(self, i):
        return {"clip": self.clips[i], "step": self.steps[i],
                "instrument": self.instr[i], "video_id": f"synthetic_{i}"}
