"""Phase metrics, against hand-computed fixtures.

Segmental values follow the MS-TCN reference convention (0-100 scale).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from src.phase.metrics import (macro_f1_multiclass, macro_f1_multilabel,
                               segmental_edit_score, segmental_f1_at_k)


def test_phase_eval_module_is_gone():
    """Phase metrics live only in src.phase.metrics."""
    assert importlib.util.find_spec("src.phase.eval") is None

# Frame-wise
def test_absent_classes_count_as_zero_in_the_macro_average():
    """Perfect on the 2 classes present, 4 classes declared -> 2/4, not 1.0."""
    target = torch.tensor([0] * 10 + [1] * 10)
    assert macro_f1_multiclass(target, target, 4, from_logits=False) == pytest.approx(0.5)
    assert macro_f1_multiclass(target, target, 2, from_logits=False) == pytest.approx(1.0)


def test_macro_f1_multiclass_matches_a_hand_computed_value():
    # class 0: tp 8, fp 0, fn 2  -> P 1.000, R 0.800, F1 0.8888...
    # class 1: tp 10, fp 2, fn 0 -> P 0.833, R 1.000, F1 0.9090...
    pred = torch.tensor([0] * 8 + [1] * 12)
    target = torch.tensor([0] * 10 + [1] * 10)
    expected = (2 * 1.0 * 0.8 / 1.8 + 2 * (10 / 12) * 1.0 / (10 / 12 + 1)) / 2
    assert macro_f1_multiclass(pred, target, 2, from_logits=False) == pytest.approx(expected)


def test_ignore_index_frames_are_dropped():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 0, -100, -100])
    assert macro_f1_multiclass(pred, target, 1, from_logits=False) == pytest.approx(1.0)


def test_scoring_padded_frames_penalises_a_perfect_model():
    """macro_f1_multilabel has no ignore mask, so the caller must drop padded frames first."""
    logits = torch.tensor([[5.0], [5.0], [5.0], [5.0]])   # always "present"
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])   # last two are padding
    valid = torch.tensor([True, True, False, False])
    assert macro_f1_multilabel(logits, target) < 1.0
    assert macro_f1_multilabel(logits[valid], target[valid]) == pytest.approx(1.0)


# Segmental
def test_edit_score_matches_hand_computed_values():
    gt = torch.tensor([0] * 10 + [1] * 10)                      # segments [0, 1]
    assert segmental_edit_score(gt, gt) == pytest.approx(100.0)

    flicker = torch.tensor(([0] * 5 + [1] * 5) * 2)             # segments [0,1,0,1]
    # Levenshtein([0,1,0,1], [0,1]) = 2 insertions; norm by max(4,2) -> (1-2/4)*100
    assert segmental_edit_score(flicker, gt) == pytest.approx(50.0)

    gt3 = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)             # segments [0,1,2]
    pred3 = torch.tensor([0] * 6 + [2] * 6)                     # segments [0,2]
    # one deletion; norm by max(2,3) -> (1-1/3)*100
    assert segmental_edit_score(pred3, gt3) == pytest.approx(100 * 2 / 3)


def test_segmental_f1_at_k_matches_hand_computed_values():
    gt = torch.tensor([0] * 10 + [1] * 10)     # segments 0:[0,10) 1:[10,20)

    # pred 0:[0,4)  IoU vs gt-0 = 4/10 = 0.40
    # pred 1:[4,20) IoU vs gt-1 = 10/16 = 0.625
    pred_a = torch.tensor([0] * 4 + [1] * 16)
    assert segmental_f1_at_k(pred_a, gt, 0.10) == pytest.approx(100.0)
    assert segmental_f1_at_k(pred_a, gt, 0.25) == pytest.approx(100.0)
    # at 0.50 the first segment misses: tp 1, fp 1, fn 1 -> P=R=0.5 -> F1 0.5
    assert segmental_f1_at_k(pred_a, gt, 0.50) == pytest.approx(50.0)

    # pred 0:[0,2)  IoU = 2/10 = 0.20 ; pred 1:[2,20) IoU = 10/18 = 0.556
    pred_b = torch.tensor([0] * 2 + [1] * 18)
    assert segmental_f1_at_k(pred_b, gt, 0.10) == pytest.approx(100.0)
    assert segmental_f1_at_k(pred_b, gt, 0.25) == pytest.approx(50.0)
    assert segmental_f1_at_k(pred_b, gt, 0.50) == pytest.approx(50.0)


def test_segmental_metrics_separate_over_segmentation_from_a_clean_prediction():
    """Equal frame-wise F1, but a flickering prediction scores far lower segmentally."""
    gt = torch.tensor([0] * 10 + [1] * 10)
    clean = torch.tensor([0] * 8 + [1] * 12)
    flicker = gt.clone()
    flicker[3], flicker[15] = 1, 0

    frame_clean = macro_f1_multiclass(clean, gt, 2, from_logits=False)
    frame_flicker = macro_f1_multiclass(flicker, gt, 2, from_logits=False)
    assert abs(frame_clean - frame_flicker) < 0.01

    assert segmental_edit_score(clean, gt) == pytest.approx(100.0)
    assert segmental_edit_score(flicker, gt) < 40.0
