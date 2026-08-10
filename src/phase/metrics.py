"""
Frame-wise and segmental metrics for surgical workflow recognition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

IGNORE_INDEX = -100
SEG_OVERLAPS: tuple[float, ...] = (0.10, 0.25, 0.50)


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _flat_multiclass(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                     ignore_index: int, from_logits: bool) -> tuple[np.ndarray, np.ndarray]:
    p = _as_numpy(pred)
    t = _as_numpy(target).reshape(-1)
    p = p.reshape(-1, p.shape[-1]).argmax(-1) if from_logits else p.reshape(-1)
    keep = t != ignore_index
    return p[keep], t[keep]


def frame_accuracy(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                   ignore_index: int = IGNORE_INDEX, from_logits: bool = True) -> float:
    """Fraction of scored frames whose predicted step equals the labelled step."""
    p, t = _flat_multiclass(pred, target, ignore_index, from_logits)
    return float((p == t).mean()) if p.size else float("nan")


def macro_f1_multiclass(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                        num_classes: int, ignore_index: int = IGNORE_INDEX,
                        from_logits: bool = True) -> float:
    p, t = _flat_multiclass(pred, target, ignore_index, from_logits)
    f1s = [_f1(*_counts(p == c, t == c)) for c in range(num_classes)]
    return float(np.mean(f1s)) if f1s else float("nan")


def macro_f1_multilabel(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                        threshold: float = 0.5, from_logits: bool = True) -> float:
    """Macro-F1 over the columns of an (..., C) multi-label prediction."""
    p = _as_numpy(pred).reshape(-1, _as_numpy(pred).shape[-1])
    t = _as_numpy(target).reshape(-1, _as_numpy(target).shape[-1])
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    hit = (_sigmoid(p) > threshold) if from_logits else (p > threshold)
    positive = t > 0.5
    f1s = [_f1(*_counts(hit[:, c], positive[:, c])) for c in range(hit.shape[1])]
    return float(np.mean(f1s)) if f1s else float("nan")


def _counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred, target = pred.astype(bool), target.astype(bool)
    return (int((pred & target).sum()), int((pred & ~target).sum()),
            int((~pred & target).sum()))


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom else 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# Segmental
def rle(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if labels.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, empty
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [labels.size]))
    return labels[starts], starts, ends


def segmental_edit_score(pred: torch.Tensor | np.ndarray,
                         target: torch.Tensor | np.ndarray) -> float:
    """(1 - normalised Levenshtein distance) x 100 over the two segment sequences."""
    p, _, _ = rle(_as_numpy(pred).reshape(-1))
    y, _, _ = rle(_as_numpy(target).reshape(-1))
    longest = max(len(p), len(y))
    if longest == 0:
        return float("nan")
    return 100.0 * (1.0 - _levenshtein(p, y) / longest)


def _levenshtein(p: np.ndarray, y: np.ndarray) -> int:
    prev = np.arange(len(y) + 1)
    for i in range(1, len(p) + 1):
        cur = np.empty(len(y) + 1, dtype=int)
        cur[0] = i
        for j in range(1, len(y) + 1):
            cost = 0 if p[i - 1] == y[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return int(prev[len(y)])


def segment_counts(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                   overlap: float) -> tuple[int, int, int]:
    p_lab, p_start, p_end = rle(_as_numpy(pred).reshape(-1))
    y_lab, y_start, y_end = rle(_as_numpy(target).reshape(-1))
    hits = np.zeros(len(y_lab), dtype=bool)
    tp = fp = 0
    for lab, s, e in zip(p_lab, p_start, p_end):
        if y_lab.size == 0:
            fp += 1
            continue
        inter = np.minimum(e, y_end) - np.maximum(s, y_start)
        union = np.maximum(e, y_end) - np.minimum(s, y_start)
        iou = np.where(y_lab == lab, inter / union, 0.0)
        best = int(iou.argmax())
        if iou[best] >= overlap and not hits[best]:
            tp += 1
            hits[best] = True
        else:
            fp += 1
    return tp, fp, int((~hits).sum())


def segmental_f1_at_k(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                      overlap: float) -> float:
    """Segment-level F1 at IoU threshold `overlap`, x 100, for one sequence.

    Multi-video scoring pools tp/fp/fn first (`score_videos`), which is not the
    mean of per-video F1s.
    """
    return 100.0 * _f1(*segment_counts(pred, target, overlap))

@dataclass(frozen=True)
class VideoPrediction:
    """One video's per-frame predictions, each frame present exactly once."""

    video_id: str
    step_logits: np.ndarray   # (N, num_steps)
    step_target: np.ndarray   # (N,)
    instr_logits: np.ndarray  # (N, num_instruments)
    instr_target: np.ndarray  # (N, num_instruments)


class LogitAggregator:

    def __init__(self) -> None:
        self._parts: dict[str, list[tuple[np.ndarray, ...]]] = defaultdict(list)

    def add(self, video_id: str, frame_index: torch.Tensor, step_logits: torch.Tensor,
            step_target: torch.Tensor, instr_logits: torch.Tensor,
            instr_target: torch.Tensor) -> None:
        """Add the valid frames of one clip; every tensor is (T, ...) for that clip."""
        self._parts[video_id].append((
            _typed(frame_index, np.int64), _typed(step_logits, np.float32),
            _typed(step_target, np.int64), _typed(instr_logits, np.float32),
            _typed(instr_target, np.float32)))

    def videos(self) -> list[VideoPrediction]:
        """One VideoPrediction per video, frames in order, duplicates averaged."""
        return [self._merge(vid, parts) for vid, parts in sorted(self._parts.items())]

    @staticmethod
    def _merge(video_id: str, parts: list[tuple[np.ndarray, ...]]) -> VideoPrediction:
        frames = np.concatenate([p[0] for p in parts])
        uniq, inv = np.unique(frames, return_inverse=True)
        if not np.array_equal(uniq, np.arange(uniq[0], uniq[0] + uniq.size)):
            raise ValueError(f"{video_id}: evaluated frames are not contiguous "
                             f"({uniq.size} frames spanning {uniq[0]}..{uniq[-1]}); "
                             "the clip index does not cover the whole video")
        counts = np.bincount(inv, minlength=uniq.size).astype(np.float32)
        return VideoPrediction(
            video_id=video_id,
            step_logits=_mean_by_group(np.concatenate([p[1] for p in parts]), inv, counts),
            step_target=_one_per_group(np.concatenate([p[2] for p in parts]), inv,
                                       uniq.size, video_id, "step"),
            instr_logits=_mean_by_group(np.concatenate([p[3] for p in parts]), inv, counts),
            instr_target=_one_per_group(np.concatenate([p[4] for p in parts]), inv,
                                        uniq.size, video_id, "instrument"))


def _typed(t: torch.Tensor, dtype: type) -> np.ndarray:
    return _as_numpy(t).astype(dtype, copy=False)


def _mean_by_group(values: np.ndarray, inv: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.zeros((counts.size, values.shape[1]), dtype=np.float32)
    np.add.at(out, inv, values)
    return out / counts[:, None]


def _one_per_group(values: np.ndarray, inv: np.ndarray, n_groups: int,
                   video_id: str, what: str) -> np.ndarray:
    out = np.empty((n_groups,) + values.shape[1:], dtype=values.dtype)
    out[inv] = values
    if not np.array_equal(out[inv], values):
        raise ValueError(f"{video_id}: overlapping windows disagree about the "
                         f"{what} label of a frame -- frames and labels are misaligned")
    return out


# Reported score
@dataclass(frozen=True)
class PhaseScore:
    split: str
    n_videos: int
    n_frames: int
    step_acc: float
    step_macro_f1: float
    step_edit: float
    step_f1: tuple[float, ...]
    steps_present: int
    num_steps: int
    instr_macro_f1: float
    instr_present: int
    num_instruments: int

    def describe(self) -> str:
        at = " ".join(f"F1@{int(k * 100)} {v:.1f}"
                      for k, v in zip(SEG_OVERLAPS, self.step_f1))
        return (f"[{self.split}] {self.n_frames} frames / {self.n_videos} videos | "
                f"step acc {self.step_acc:.3f} macro-F1 {self.step_macro_f1:.3f} "
                f"({self.steps_present}/{self.num_steps} steps present) | "
                f"edit {self.step_edit:.1f} {at} | "
                f"instr macro-F1 {self.instr_macro_f1:.3f} "
                f"({self.instr_present}/{self.num_instruments} instruments present)")


def score_videos(videos: Sequence[VideoPrediction], *, num_steps: int,
                 num_instruments: int, instrument_threshold: float,
                 split: str) -> PhaseScore:
    if not videos:
        raise ValueError(f"{split}: nothing to score")

    step_pred = [v.step_logits.argmax(-1) for v in videos]
    step_true = [v.step_target for v in videos]
    flat_pred = np.concatenate(step_pred)
    flat_true = np.concatenate(step_true)

    seg_f1 = []
    for k in SEG_OVERLAPS:
        tp, fp, fn = np.sum([segment_counts(p, t, k)
                             for p, t in zip(step_pred, step_true)], axis=0)
        seg_f1.append(100.0 * _f1(int(tp), int(fp), int(fn)))

    instr_logits = np.concatenate([v.instr_logits for v in videos])
    instr_true = np.concatenate([v.instr_target for v in videos])

    return PhaseScore(
        split=split,
        n_videos=len(videos),
        n_frames=int(flat_true.size),
        step_acc=frame_accuracy(flat_pred, flat_true, from_logits=False),
        step_macro_f1=macro_f1_multiclass(flat_pred, flat_true, num_steps, from_logits=False),
        step_edit=float(np.mean([segmental_edit_score(p, t)
                                 for p, t in zip(step_pred, step_true)])),
        step_f1=tuple(seg_f1),
        steps_present=int(np.unique(flat_true).size),
        num_steps=num_steps,
        instr_macro_f1=macro_f1_multilabel(instr_logits, instr_true,
                                           threshold=instrument_threshold),
        instr_present=int((instr_true > 0.5).any(0).sum()),
        num_instruments=num_instruments,
    )


__all__ = ["IGNORE_INDEX", "SEG_OVERLAPS", "VideoPrediction", "PhaseScore",
           "LogitAggregator", "frame_accuracy", "macro_f1_multiclass",
           "macro_f1_multilabel", "rle", "segmental_edit_score", "segment_counts",
           "segmental_f1_at_k", "score_videos"]
