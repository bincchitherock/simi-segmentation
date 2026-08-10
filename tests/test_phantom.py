"""
The endoscopic phantom: determinism, a task no threshold solves, and the on-disk data.
The renderer and `make_dummy_data` share one scene definition, so both are checked here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.data.phantom import (N_INSTRUMENT_KINDS, N_STEPS, best_threshold_dice,
                              render_labelled_frame, render_video)

SIZE = 96
# Measured: the phantom's best single-threshold Dice is 0.09-0.60 over 10 frames;
# adding the mask into the image takes it to 0.71-0.99. 0.75 sits between the two.
LEAK_DICE = 0.75


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def frames(seed: int, n: int = 6) -> list:
    gen = rng(seed)
    return [render_labelled_frame(gen, SIZE) for _ in range(n)]


# Determinisim

def test_same_seed_renders_byte_identical_frames():
    for a, b in zip(frames(0), frames(0)):
        assert np.array_equal(a.image, b.image)
        assert np.array_equal(a.mask, b.mask)
        assert (a.step, a.instruments) == (b.step, b.instruments)


def test_the_renderer_does_not_touch_the_global_numpy_rng():
    np.random.seed(1)
    first = render_labelled_frame(rng(0), SIZE)
    np.random.seed(999)
    assert np.array_equal(render_labelled_frame(rng(0), SIZE).image, first.image)


def test_different_seeds_render_different_scenes():
    a, b = frames(0, 1)[0], frames(1, 1)[0]
    assert not np.array_equal(a.image, b.image)


def test_render_video_is_deterministic():
    a = render_video(rng(3), n_frames=12, size=SIZE)
    b = render_video(rng(3), n_frames=12, size=SIZE)
    assert [f.step for f in a] == [f.step for f in b]
    assert all(np.array_equal(x.mask, y.mask) for x, y in zip(a, b))


# Task must be non trivial
def test_the_mask_is_not_painted_into_the_image():
    for i, f in enumerate(frames(0, 8)):
        if not f.mask.any():
            continue
        gray = np.asarray(Image.fromarray(f.image).convert("L"), dtype=np.float32)
        d = best_threshold_dice(gray, f.mask > 127)
        assert d < LEAK_DICE, f"frame {i}: a global threshold reaches Dice {d:.3f}"


def test_instrument_masks_are_present_and_small():
    rendered = frames(0, 12)
    with_instruments = [f for f in rendered if f.instruments]
    assert with_instruments, "no frame in 12 showed an instrument"
    for f in with_instruments:
        frac = (f.mask > 127).mean()
        assert 0.0 < frac < 0.5, f"instrument covers {frac:.1%} of the frame"


def test_mask_emptiness_agrees_with_the_declared_instruments():
    for f in frames(0, 12):
        assert bool(f.mask.any()) == bool(f.instruments)


def test_labels_stay_inside_their_declared_ranges():
    for f in frames(0, 8):
        assert 0 <= f.step < N_STEPS
        assert all(0 <= k < N_INSTRUMENT_KINDS for k in f.instruments)
        assert f.presence().count(1) == len(set(f.instruments))


def test_video_steps_are_piecewise_constant_and_monotonic():
    steps = [f.step for f in render_video(rng(0), n_frames=30, size=SIZE)]
    assert len(steps) == 30
    assert steps == sorted(steps), "steps must not go backwards"
    assert len(set(steps)) > 1, "a single step over 30 frames is not a workflow"

def test_on_disk_seg_masks_are_not_readable_by_thresholding(generated_data):
    masks = sorted((generated_data / "seg" / "masks").glob("*.png"))
    assert masks, f"no masks under {generated_data / 'seg'}"
    checked = 0
    for mask_path in masks[:8]:
        image_path = generated_data / "seg" / "images" / mask_path.name
        assert image_path.exists(), image_path
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if not mask.any():
            continue
        img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        d = best_threshold_dice(img, mask)
        assert d < LEAK_DICE, f"{mask_path.name}: a global threshold reaches Dice {d:.3f}"
        checked += 1
    assert checked, "every sampled mask was empty"


def test_on_disk_phase_manifest_is_self_consistent(generated_data):
    videos = json.loads((generated_data / "phase" / "videos.json").read_text())
    assert videos
    for v in videos:
        n_files = len(list((generated_data / "phase" / v["frames_dir"]).glob("*.jpg")))
        assert n_files == len(v["steps"]) == len(v["instruments"]), v["video_id"]
        assert all(0 <= s < N_STEPS for s in v["steps"])


def test_generated_data_is_reproducible(tmp_path, generated_data):
    """Same seed, same bytes on disk."""
    from conftest import run_module

    proc = run_module("src.data.make_dummy_data", "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    sample = sorted((generated_data / "seg" / "images").glob("*.png"))[:3]
    assert sample
    for path in sample:
        again = tmp_path / "seg" / "images" / path.name
        assert again.read_bytes() == path.read_bytes(), path.name


@pytest.mark.parametrize("n_frames", [1, 4])
def test_render_video_rejects_clips_too_short_to_hold_a_segment(n_frames):
    with pytest.raises(ValueError):
        render_video(rng(0), n_frames=n_frames, size=SIZE)
