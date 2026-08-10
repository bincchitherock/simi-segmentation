"""
LogitAggregator: overlapping windows merge into one contiguous sequence per video.
A frame covered by several windows holds the mean of their logits.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.phase.metrics import LogitAggregator

NUM_STEPS, NUM_INSTR = 3, 2


def add_window(agg: LogitAggregator, video_id: str, frames: list[int],
               step_fill: float, target: list[int]) -> None:
    n = len(frames)
    agg.add(video_id,
            torch.tensor(frames),
            torch.full((n, NUM_STEPS), step_fill),
            torch.tensor(target),
            torch.full((n, NUM_INSTR), step_fill),
            torch.zeros(n, NUM_INSTR))


def test_overlapping_windows_are_averaged_so_every_frame_counts_once():
    agg = LogitAggregator()
    add_window(agg, "video_00", [0, 1, 2, 3], 1.0, [0, 0, 1, 1])
    add_window(agg, "video_00", [2, 3, 4, 5], 3.0, [1, 1, 2, 2])

    (video,) = agg.videos()
    assert video.step_logits.shape == (6, NUM_STEPS)
    assert video.step_target.tolist() == [0, 0, 1, 1, 2, 2]
    # frames 2 and 3 fall in both windows, so they hold the mean of the two fills
    assert video.step_logits[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert video.instr_logits[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_videos_are_kept_apart_and_returned_in_id_order():
    agg = LogitAggregator()
    add_window(agg, "video_01", [0, 1], 5.0, [2, 2])
    add_window(agg, "video_00", [0, 1], 1.0, [0, 0])

    ids = [v.video_id for v in agg.videos()]
    assert ids == ["video_00", "video_01"]
    assert np.allclose(agg.videos()[1].step_logits, 5.0)


def test_a_gap_in_the_frame_index_raises():
    agg = LogitAggregator()
    add_window(agg, "video_00", [0, 1], 1.0, [0, 0])
    add_window(agg, "video_00", [5, 6], 1.0, [1, 1])

    with pytest.raises(ValueError, match="gaps"):
        agg.videos()


@pytest.mark.parametrize("what", ["step", "instrument"])
def test_windows_disagreeing_about_a_frame_label_raises(what):
    agg = LogitAggregator()
    add_window(agg, "video_00", [0, 1, 2], 1.0, [0, 0, 1])
    if what == "step":
        add_window(agg, "video_00", [2, 3], 1.0, [2, 2])  # frame 2 was labelled 1
    else:
        agg.add("video_00", torch.tensor([2, 3]), torch.zeros(2, NUM_STEPS),
                torch.tensor([1, 1]), torch.zeros(2, NUM_INSTR),
                torch.ones(2, NUM_INSTR))  # frame 2 had all-zero instruments

    with pytest.raises(ValueError, match=f"{what} label"):
        agg.videos()
