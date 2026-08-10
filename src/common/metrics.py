"""
Scoring a predicted mask against the true one. Both models use this.

Dice and IoU both measure how much two masks overlap, on a scale of 0 to 1. Dice is
the more forgiving of the two. I skipped any pair where both masks were empty, because
there is nothing there to agree or disagree about, and kept a count of what went, so a
score can never quietly average over a different set of frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class SegScore:
    """The average scores, and how many masks went into them."""

    dice: float
    iou: float
    n_scored: int
    n_empty_pairs: int

    @property
    def n_total(self) -> int:
        return self.n_scored + self.n_empty_pairs

    def describe(self, label: str = "") -> str:
        """One line with the scores and how many masks they cover."""
        prefix = f"{label} " if label else ""
        if self.n_scored == 0:
            return (f"{prefix}Dice n/a | IoU n/a, 0 of {self.n_total} masks scored "
                    f"({self.n_empty_pairs} had nothing in either mask)")
        return (f"{prefix}Dice {self.dice:.3f} | IoU {self.iou:.3f} "
                f"over {self.n_scored}/{self.n_total} masks "
                f"({self.n_empty_pairs} skipped, nothing in either mask)")


def dice_iou(pred: torch.Tensor, target: torch.Tensor, *, from_logits: bool = True,
             threshold: float = 0.5) -> SegScore:
    """Score a batch of predicted masks against the true ones."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} "
                         "must have the same shape")
    if pred.ndim < 2:
        raise ValueError("expected a batched tensor of shape (B, ...)")

    prob = pred.sigmoid() if from_logits else pred.float()
    p = (prob > threshold).float()
    t = (target > 0.5).float()

    dims = tuple(range(1, p.ndim))
    inter = (p * t).sum(dims)
    psum, tsum = p.sum(dims), t.sum(dims)
    denom = psum + tsum

    scorable = denom > 0
    n_scored = int(scorable.sum().item())
    n_empty = int(p.shape[0] - n_scored)
    if n_scored == 0:
        return SegScore(float("nan"), float("nan"), 0, n_empty)

    inter, denom = inter[scorable], denom[scorable]
    dice = (2 * inter / denom).mean().item()
    iou = (inter / (denom - inter)).mean().item()
    return SegScore(dice, iou, n_scored, n_empty)


def merge_scores(scores: Iterable[SegScore]) -> SegScore:
    """Combine scores from several batches, weighted by how many masks each covers."""
    scores = list(scores)
    n_scored = sum(s.n_scored for s in scores)
    n_empty = sum(s.n_empty_pairs for s in scores)
    if n_scored == 0:
        return SegScore(float("nan"), float("nan"), 0, n_empty)
    dice = sum(s.dice * s.n_scored for s in scores if s.n_scored) / n_scored
    iou = sum(s.iou * s.n_scored for s in scores if s.n_scored) / n_scored
    if not math.isfinite(dice):
        raise ValueError("a SegScore with n_scored > 0 carried a non-finite dice")
    return SegScore(dice, iou, n_scored, n_empty)


__all__ = ["SegScore", "dice_iou", "merge_scores"]
