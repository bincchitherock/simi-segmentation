from __future__ import annotations

import math

import pytest
import torch

from src.common.metrics import dice_iou, merge_scores

PRED = torch.tensor([[1., 1., 0., 0.],
                     [1., 1., 1., 1.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.]])
TARGET = torch.tensor([[1., 0., 0., 0.],
                       [1., 1., 1., 1.],
                       [0., 0., 0., 0.],
                       [1., 1., 0., 0.]])
EXPECTED_DICE = (2 / 3 + 1.0 + 0.0) / 3
EXPECTED_IOU = (0.5 + 1.0 + 0.0) / 3


def test_dice_iou_matches_hand_computed_values():
    score = dice_iou(PRED, TARGET, from_logits=False)
    assert score.dice == pytest.approx(EXPECTED_DICE)
    assert score.iou == pytest.approx(EXPECTED_IOU)


def test_both_empty_is_excluded_not_scored_one():
    score = dice_iou(PRED, TARGET, from_logits=False)
    assert (score.n_scored, score.n_empty_pairs, score.n_total) == (3, 1, 4)
    # below the mean the empty pair would produce if it were scored 1.0
    assert score.dice < (EXPECTED_DICE * 3 + 1.0) / 4


def test_empty_prediction_against_a_real_mask_scores_zero():
    score = dice_iou(PRED[3:], TARGET[3:], from_logits=False)
    assert score.dice == 0.0 and score.iou == 0.0 and score.n_scored == 1


def test_nothing_scorable_is_nan_and_says_so():
    score = dice_iou(torch.zeros(2, 4), torch.zeros(2, 4), from_logits=False)
    assert math.isnan(score.dice) and score.n_scored == 0
    assert "0 of 2" in score.describe()


def test_describe_names_the_population():
    text = dice_iou(PRED, TARGET, from_logits=False).describe("val")
    assert text.startswith("val ") and "3/4" in text and "1 skipped" in text


def test_merge_scores_is_sample_weighted():
    whole = dice_iou(PRED, TARGET, from_logits=False)
    parts = merge_scores([dice_iou(PRED[:2], TARGET[:2], from_logits=False),
                          dice_iou(PRED[2:], TARGET[2:], from_logits=False)])
    assert parts.dice == pytest.approx(whole.dice)
    assert parts.iou == pytest.approx(whole.iou)
    assert (parts.n_scored, parts.n_empty_pairs) == (whole.n_scored, whole.n_empty_pairs)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_iou(torch.zeros(2, 4), torch.zeros(2, 5), from_logits=False)


def test_logits_are_thresholded_at_zero_by_default():
    logits = torch.tensor([[2.0, -2.0, -2.0, -2.0]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert dice_iou(logits, target).dice == pytest.approx(1.0)
