"""
Loads video frames and their labels, and hands them to the phase model in clips.

A whole video did not fit in memory at once, so I broke it into overlapping clips of
`clip_len` frames, taking a new one every `stride` frames. The overlap means most
frames get predicted more than once, and `LogitAggregator` in `metrics.py` averages
those repeats back together at scoring time.

Clips near the end of a video can come up short. Those get padded, and a `valid` flag
marks which frames are real so the padding never reaches the loss or the score.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.config import PhaseConfig
from src.data.splits import resolve_splits
from src.phase.metrics import IGNORE_INDEX

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# Standard per-colour averages, used to centre the pixel values before the encoder
# sees them. Shaped as columns so they line up against an image without extra work.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def list_frames(frames_dir: str) -> list[str]:
    """Image files in a folder, ordered by the number in each filename."""
    names = [f for f in os.listdir(frames_dir)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    digits = [re.findall(r"\d+", n) for n in names]
    if names and not all(digits):
        unnumbered = sorted(n for n, d in zip(names, digits) if not d)
        raise ValueError(f"{frames_dir}: {unnumbered[:3]} have no number in the filename "
                         "while the other frames do, so I could not work out what order "
                         "they go in. Remove them, or list the frames in videos.json")
    if not names:
        return []
    return [n for _, n in sorted(zip((int(d[-1]) for d in digits), names))]


def manifest_video_ids(data_root: str, manifest: str = "videos.json") -> list[str]:
    """Every video id in `videos.json`, in the order the file lists them."""
    with open(os.path.join(data_root, manifest)) as f:
        return [v["video_id"] for v in json.load(f)]


def resolve_phase_splits(cfg: PhaseConfig) -> dict[str, list[str]]:
    """Which videos are in train, val and test, for a phase config."""
    return resolve_splits(available=manifest_video_ids(cfg.data_root),
                          splits_file=cfg.splits, data_root=cfg.data_root,
                          explicit={"train": cfg.train_split, "val": cfg.val_split,
                                    "test": cfg.test_split},
                          required=("train", "val"))


def window_starts(n_frames: int, clip_len: int, stride: int) -> list[int]:
    """
    Where each clip starts, chosen so that every frame lands in at least one clip.

    Stepping by `stride` can leave a few frames stranded at the end, so I added one last
    clip pinned to the final frame. It overlaps the one before it, which is fine.
    """
    starts = list(range(0, max(1, n_frames - clip_len + 1), stride))
    if n_frames > clip_len and starts[-1] + clip_len < n_frames:
        starts.append(n_frames - clip_len)
    return starts


class ClipDataset(Dataset):
    """
    Serves one clip at a time, as images plus labels.

    Each item is a dict holding `clip` (the images), `step` and `instrument` (the
    labels), `valid` (which frames are real rather than padding), plus the video id
    and frame numbers so predictions can be put back in place afterwards.

    Labels are checked against the frames up front, so a video with the wrong number
    of labels fails here rather than training quietly on nonsense.
    """

    def __init__(self, data_root: str, manifest: str = "videos.json", clip_len: int = 64,
                 stride: int = 32, img_size: int = 224, num_steps: int = 14,
                 num_instruments: int = 18, split: Optional[Sequence[str]] = None,
                 split_name: str = "") -> None:
        self.root = data_root
        self.clip_len = clip_len
        self.img_size = img_size
        self.num_steps = num_steps
        self.num_instruments = num_instruments
        self.split_name = split_name

        with open(os.path.join(data_root, manifest)) as f:
            by_id = {v["video_id"]: v for v in json.load(f)}
        wanted = list(by_id) if split is None else list(split)
        missing = [vid for vid in wanted if vid not in by_id]
        if missing:
            raise ValueError(f"{manifest} has no video(s) {missing}; it contains "
                             f"{sorted(by_id)}")
        if not wanted:
            raise ValueError(f"ClipDataset({split_name or data_root!r}) selected no "
                             "videos. A split has to name at least one")

        self.video_ids = wanted
        self.files: list[list[str]] = []
        self.frame_dirs: list[str] = []
        self.steps: list[torch.Tensor] = []
        self.instruments: list[torch.Tensor] = []
        self.index: list[tuple[int, int]] = []

        for vi, vid in enumerate(self.video_ids):
            v = by_id[vid]
            frames_dir = os.path.join(data_root, v["frames_dir"])
            files = self._frame_files(vid, frames_dir, v.get("frames"))
            steps = np.asarray(v["steps"], dtype=np.int64)
            instr = np.asarray(v["instruments"], dtype=np.float32)
            self._validate(vid, files, steps, instr)

            self.frame_dirs.append(frames_dir)
            self.files.append(files)
            self.steps.append(torch.from_numpy(steps))
            self.instruments.append(torch.from_numpy(instr))
            self.index += [(vi, s) for s in window_starts(len(files), clip_len, stride)]

    @staticmethod
    def _frame_files(vid: str, frames_dir: str, declared: Optional[Sequence[str]]) -> list[str]:
        if declared is None:
            return list_frames(frames_dir)
        missing = [f for f in declared if not os.path.isfile(os.path.join(frames_dir, f))]
        if missing:
            raise ValueError(f"{vid}: videos.json declares {len(missing)} frame(s) that are "
                             f"not in {frames_dir}, e.g. {missing[:3]}")
        return list(declared)

    def _validate(self, vid: str, files: list[str], steps: np.ndarray,
                  instr: np.ndarray) -> None:
        if not files:
            raise ValueError(f"{vid}: no image files in the frames directory")
        if not (len(files) == steps.shape[0] == instr.shape[0]):
            raise ValueError(f"{vid}: {len(files)} frame images, but {steps.shape[0]} step "
                             f"labels and {instr.shape[0]} instrument labels. The frames "
                             "and the labels do not line up")
        if instr.ndim != 2 or instr.shape[1] != self.num_instruments:
            raise ValueError(f"{vid}: instrument labels are {instr.shape}, expected "
                             f"(n_frames, {self.num_instruments})")
        bad = steps[(steps < 0) | (steps >= self.num_steps)]
        if bad.size:
            raise ValueError(f"{vid}: step label {int(bad[0])} outside "
                             f"[0, {self.num_steps})")

    def __len__(self) -> int:
        return len(self.index)

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """How often each step and instrument appears. Training uses this to weight rare ones."""
        step_counts = torch.zeros(self.num_steps)
        instr_pos = torch.zeros(self.num_instruments)
        total = 0
        for steps, instr in zip(self.steps, self.instruments):
            step_counts += torch.bincount(steps, minlength=self.num_steps).float()
            instr_pos += (instr > 0.5).float().sum(0)
            total += steps.shape[0]
        return step_counts, instr_pos, total

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as raw:
            img = raw.convert("RGB").resize((self.img_size, self.img_size))
            arr = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 255.0
        return (arr - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, i: int) -> dict[str, Any]:
        vi, start = self.index[i]
        end = min(start + self.clip_len, len(self.files[vi]))
        frames_dir = self.frame_dirs[vi]

        imgs = torch.stack([self._load_image(os.path.join(frames_dir, f))
                            for f in self.files[vi][start:end]])
        steps = self.steps[vi][start:end]
        instr = self.instruments[vi][start:end]
        frame_index = torch.arange(start, end)
        n = end - start
        valid = torch.ones(n, dtype=torch.bool)

        pad = self.clip_len - n
        if pad > 0:
            imgs = torch.cat([imgs, imgs[-1:].expand(pad, -1, -1, -1)], 0)
            steps = torch.cat([steps, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
            instr = torch.cat([instr, torch.zeros(pad, self.num_instruments)])
            frame_index = torch.cat([frame_index, torch.full((pad,), -1)])
            valid = torch.cat([valid, torch.zeros(pad, dtype=torch.bool)])

        return {"clip": imgs, "step": steps, "instrument": instr, "valid": valid,
                "frame_index": frame_index, "video_id": self.video_ids[vi]}


__all__ = ["ClipDataset", "IMAGE_EXTS", "list_frames", "manifest_video_ids",
           "resolve_phase_splits", "window_starts"]
