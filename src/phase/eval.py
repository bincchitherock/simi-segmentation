from __future__ import annotations

import torch

def macro_f1_multiclass(logits_or_preds: torch.Tensor, target: torch.Tensor,
                        num_classes: int, ignore_index: int = -100,
                        from_logits: bool = True) -> float:
    preds = logits_or_preds.argmax(-1) if from_logits else logits_or_preds
    preds = preds.reshape(-1)
    target = target.reshape(-1)
    mask = target != ignore_index
    preds, target = preds[mask], target[mask]
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (target == c)).sum().item()
        fp = ((preds == c) & (target != c)).sum().item()
        fn = ((preds != c) & (target == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0

def macro_f1_multilabel(logits: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> float:
    preds = (logits.sigmoid() > threshold).float().reshape(-1, logits.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    f1s = []
    for c in range(preds.shape[1]):
        p, t = preds[:, c], target[:, c]
        tp = ((p == 1) & (t == 1)).sum().item()
        fp = ((p == 1) & (t == 0)).sum().item()
        fn = ((p == 0) & (t == 1)).sum().item()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0

def dice_iou(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = (pred_logits.sigmoid() > 0.5).float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    psum, tsum = pred.sum(dims), target.sum(dims)
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice.mean().item(), iou.mean().item()
