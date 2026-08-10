from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.common.config import SegConfig, config_to_dict, load_config
from src.common.device import device_report, get_device
from src.common.lora import adapter_state_dict
from src.common.metrics import SegScore, dice_iou, merge_scores
from src.common.seed import make_generator, seed_everything, seed_worker
from src.segmentation.dataset import MaskDataset, collate, resolve_seg_splits
from src.segmentation.sam2_lora import (SegmenterModule, build_segmenter, dice_bce_loss,
                                        forward_batch)


@torch.no_grad()
def evaluate(model: SegmenterModule, loader: DataLoader, device: torch.device) -> SegScore:
    """Mean Dice/IoU over `loader`, with the scored population carried along."""
    model.eval()
    return merge_scores(dice_iou(forward_batch(model, b, device).cpu(), b["mask"])
                        for b in loader)


def _loader(ds: MaskDataset, cfg: SegConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.workers, collate_fn=collate,
                      generator=make_generator(cfg.seed), worker_init_fn=seed_worker)


def train_epoch(model: SegmenterModule, loader: DataLoader,
                opt: torch.optim.Optimizer, cfg: SegConfig, device: torch.device,
                step: int, max_steps: int) -> tuple[int, list[dict[str, float]]]:
    """One pass over `loader`, stopping at `max_steps`. Returns (step, loss history)."""
    model.train()
    history: list[dict[str, float]] = []
    for batch in loader:
        loss = dice_bce_loss(forward_batch(model, batch, device),
                             batch["mask"].to(device))
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"step {step}: loss is {loss.item()} -- aborting rather "
                               "than silently training on NaN")
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:  # a max_norm of 0.0 would zero every gradient
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           cfg.grad_clip)
        opt.step()
        step += 1
        history.append({"step": step, "loss": loss.item()})
        if step % max(1, max_steps // 10) == 0 or step == 1:
            print(f"step {step:04d}/{max_steps} | loss {loss.item():.4f}")
        if step >= max_steps:
            break
    return step, history


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/seg")
    ap.add_argument("--steps", type=int, default=None,
                    help="cap total optimizer steps (smoke tests); epochs still bound the run")
    ap.add_argument("--model", default=None, choices=["sam2", "tinyunet"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-cfg", dest="model_cfg", default=None)
    ap.add_argument("--img-size", dest="img_size", type=int, default=None)
    ap.add_argument("--prompt-kind", dest="prompt_kind", default=None,
                    choices=["point", "box"])
    ap.add_argument("--lora-rank", dest="lora_rank", type=int, default=None)
    ap.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "out", "steps")}
    cfg = load_config(args.config, SegConfig, overrides=overrides)

    seed_everything(cfg.seed)
    device = get_device()
    print(device_report(device))

    splits = resolve_seg_splits(cfg)  # raises unless train/val exist and are disjoint
    train_ids, val_ids = splits["train"], splits["val"]

    common = dict(data_root=cfg.data_root, img_size=cfg.img_size,
                  prompt_kind=cfg.prompt_kind, seed=cfg.seed)
    train_ds = MaskDataset(split=train_ids, name="train", **common)
    val_ds = MaskDataset(split=val_ids, name="val", **common)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)

    model = build_segmenter(cfg, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "adapters.pt")
    history_path = os.path.join(args.out, "history.json")
    history: dict[str, Any] = {"backend": model.backend, "val_split": list(val_ids),
                     "loss": [], "eval": []}

    max_steps = args.steps if args.steps is not None else cfg.epochs * len(train_loader)
    step = 0
    best = -1.0
    best_epoch = -1

    def save(score: SegScore, epoch: int) -> None:
        torch.save({**adapter_state_dict(model.adapter_root, model.lora_spec,
                                         root_name=model.backend),
                    "cfg": config_to_dict(cfg), "backend": model.backend,
                    "val_dice": score.dice, "val_split": list(val_ids),
                    "epoch": epoch, "train_steps": step}, out_path)

    for epoch in range(cfg.epochs):
        step, losses = train_epoch(model, train_loader, opt, cfg, device, step, max_steps)
        history["loss"] += losses

        score = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | {model.backend} | {score.describe('val')}")
        # the scored population travels with the score
        history["eval"].append({"epoch": epoch, "step": step, "dice": score.dice,
                                "iou": score.iou, "n_scored": score.n_scored,
                                "n_total": score.n_total})
        if score.n_scored != score.n_total:
            raise RuntimeError(
                f"epoch {epoch}: {score.n_empty_pairs} val masks were excluded as "
                "empty-vs-empty, so this epoch's Dice is an average over a different "
                "population than the others' and selecting on it would be meaningless")
        if score.dice > best:
            best, best_epoch = score.dice, epoch
            save(score, epoch)
        if step >= max_steps:
            break

    history["selected_epoch"] = best_epoch
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"done. best val Dice {best:.3f} at epoch {best_epoch} on videos "
          f"{list(val_ids)} using {model.backend} — this is a model-SELECTION score, "
          f"not a held-out test score. adapters -> {out_path}, curve -> {history_path}\n"
          f"for the held-out number: python -m src.segmentation.predict "
          f"--checkpoint {out_path} --split test")


if __name__ == "__main__":
    main()
