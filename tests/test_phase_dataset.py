"""ClipDataset: frame/label alignment, window coverage, and padding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import write_phase_videos
from src.phase.dataset import ClipDataset

NUM_INSTRUMENTS = 3


def write_gradient_video(root: Path, names: list[str], *, vid: str = "video_00",
                         size: int = 8, n_labels: int | None = None,
                         declare_frames: list[str] | None = None) -> Path:
    """One flat-colour frame per name, frame k uniformly k*20.

    A clip's per-frame means rise monotonically exactly when the frames were read in
    the order of `names`.
    """
    root.mkdir(parents=True, exist_ok=True)
    vdir = root / vid
    vdir.mkdir(exist_ok=True)
    for k, name in enumerate(names):
        arr = np.full((size, size, 3), k * 20, dtype=np.uint8)
        Image.fromarray(arr).save(vdir / name)
    n = len(names) if n_labels is None else n_labels
    record = {"video_id": vid, "frames_dir": vid, "fps": 1,
              "steps": list(range(n)),
              "instruments": [[0] * NUM_INSTRUMENTS] * n}
    if declare_frames is not None:
        record["frames"] = list(declare_frames)
    (root / "videos.json").write_text(json.dumps([record]))
    return root


def load_clip(root: Path, clip_len: int) -> dict:
    ds = ClipDataset(str(root), clip_len=clip_len, stride=clip_len, img_size=8,
                     num_instruments=NUM_INSTRUMENTS)
    assert len(ds) == 1
    return ds[0]


def frame_means(clip: torch.Tensor, upto: int) -> list[float]:
    return [float(clip[i].mean()) for i in range(upto)]


# Ordering and alignment

def test_stray_non_image_file_does_not_break_alignment(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(12)]
    root = write_gradient_video(tmp_path / "phase", names)
    (root / "video_00" / ".DS_Store").write_bytes(b"\x00\x01junk")

    item = load_clip(root, clip_len=12)
    assert item["step"].tolist() == list(range(12))
    means = frame_means(item["clip"], 12)
    assert means == sorted(means)


def test_frames_are_sorted_numerically_not_lexicographically(tmp_path):
    names = [f"frame_{i}.jpg" for i in range(12)]
    root = write_gradient_video(tmp_path / "phase", names)

    means = frame_means(load_clip(root, clip_len=12)["clip"], 12)
    assert means == sorted(means)


def test_the_declared_frames_list_is_what_is_read(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(12)]
    root = write_gradient_video(tmp_path / "phase", names,
                                declare_frames=list(reversed(names)))

    # brightness rises with the filename number, so the reversed declaration
    # must give a decreasing sequence
    means = frame_means(load_clip(root, clip_len=12)["clip"], 12)
    assert means == sorted(means, reverse=True), "the declared order was ignored"


def test_a_declared_frame_that_does_not_exist_raises_with_the_video_id(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(12)]
    root = write_gradient_video(tmp_path / "phase", names, declare_frames=names)
    manifest = json.loads((root / "videos.json").read_text())
    manifest[0]["frames"][3] = "typo.jpg"
    (root / "videos.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="video_00"):
        load_clip(root, clip_len=12)


def test_mixing_numbered_and_unnumbered_frame_names_raises(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(11)] + ["intro.jpg"]
    root = write_gradient_video(tmp_path / "phase", names)

    with pytest.raises(ValueError, match="what order"):
        load_clip(root, clip_len=12)


def test_frame_count_mismatching_the_labels_raises_with_the_video_id(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(12)]
    root = write_gradient_video(tmp_path / "phase", names, n_labels=12)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "video_00" / "thumb.jpg")

    with pytest.raises(ValueError, match="video_00"):
        ClipDataset(str(root), clip_len=12, stride=12, img_size=8,
                    num_instruments=NUM_INSTRUMENTS)


def test_fewer_frames_than_labels_raises(tmp_path):
    names = [f"frame_{i:06d}.jpg" for i in range(10)]
    root = write_gradient_video(tmp_path / "phase", names, n_labels=12)
    with pytest.raises(ValueError, match="video_00"):
        ClipDataset(str(root), clip_len=12, stride=12, img_size=8,
                    num_instruments=NUM_INSTRUMENTS)

# Windowing and padding
def test_every_frame_of_a_long_video_is_covered_by_some_window(tmp_path):
    """The window index covers the tail of a video that is not a multiple of the stride."""
    root = write_phase_videos(tmp_path / "phase", {"video_00": 100},
                              num_instruments=NUM_INSTRUMENTS)
    ds = ClipDataset(str(root), clip_len=64, stride=32, img_size=8,
                     num_instruments=NUM_INSTRUMENTS)

    covered: set[int] = set()
    for _, start in ds.index:
        covered.update(range(start, min(start + 64, 100)))
    assert covered == set(range(100))


def test_short_video_is_padded_and_the_padding_is_marked_invalid(tmp_path):
    root = write_phase_videos(tmp_path / "phase", {"video_00": 40},
                              num_instruments=NUM_INSTRUMENTS)
    ds = ClipDataset(str(root), clip_len=64, stride=64, img_size=8,
                     num_instruments=NUM_INSTRUMENTS)
    item = ds[0]

    assert item["clip"].shape[0] == 64
    assert item["step"].shape[0] == 64
    assert item["instrument"].shape[:1] == (64,)

    valid = item["valid"].bool()
    assert valid.sum().item() == 40
    assert valid[:40].all() and not valid[40:].any()
    assert (item["step"][40:] == -100).all()


def test_lengths_agree_across_every_returned_tensor(tmp_path):
    root = write_phase_videos(tmp_path / "phase", {"video_00": 37},
                              num_instruments=NUM_INSTRUMENTS)
    ds = ClipDataset(str(root), clip_len=16, stride=8, img_size=8,
                     num_instruments=NUM_INSTRUMENTS)
    for i in range(len(ds)):
        item = ds[i]
        n = item["clip"].shape[0]
        assert n == 16
        assert item["step"].shape[0] == n
        assert item["instrument"].shape[0] == n
        assert item["valid"].shape[0] == n

# Splits and determinism
def test_split_selects_only_the_named_videos(tmp_path):
    root = write_phase_videos(tmp_path / "phase", {"video_00": 16, "video_01": 16},
                              num_instruments=NUM_INSTRUMENTS)
    ds = ClipDataset(str(root), clip_len=16, stride=16, img_size=8,
                     num_instruments=NUM_INSTRUMENTS, split=["video_01"])
    assert {ds[i]["video_id"] for i in range(len(ds))} == {"video_01"}


def test_two_constructions_yield_identical_tensors(tmp_path):
    root = write_phase_videos(tmp_path / "phase", {"video_00": 20},
                              num_instruments=NUM_INSTRUMENTS)
    kwargs = dict(clip_len=16, stride=8, img_size=8, num_instruments=NUM_INSTRUMENTS)
    a, b = ClipDataset(str(root), **kwargs), ClipDataset(str(root), **kwargs)
    assert len(a) == len(b) and len(a) > 0
    for i in range(len(a)):
        assert torch.equal(a[i]["clip"], b[i]["clip"])
        assert torch.equal(a[i]["step"], b[i]["step"])
