"""
Trains the phase model.

    python -m src.phase.train --config configs/phase_phantom.yaml

Each epoch runs through the training videos, then scores the val videos. I saved the
model whenever that val score beat the best so far, so what ends up on disk is the best
epoch rather than the last one.

Val is used to choose, so its score is flattering and is not a fair estimate. If a test
split exists, the saved model is scored on it once at the very end, and that is the
number worth quoting.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.common.config import (PhaseConfig, config_from_mapping, config_to_dict,
                               load_config)
from src.common.device import device_report, get_device
from src.common.seed import make_generator, seed_everything, seed_worker
from src.phase.dataset import ClipDataset, resolve_phase_splits
from src.phase.metrics import (LogitAggregator, PhaseScore, VideoPrediction,
                               score_videos)
from src.phase.model import PhaseModelConfig, SpatioTemporalMultiTask

CHECKPOINT_FORMAT = "psai-phase"
CHECKPOINT_VERSION = 1
SELECTION_METRIC = "val step macro-F1"
# The transformer needs its learning rate eased in over the first few percent of
# training, or the loss spikes early on. The tcn does not, so it skips this.
WARMUP_FRACTION = 0.05


@dataclass(frozen=True)
class PhaseTrainConfig(PhaseConfig):
    """Everything in `PhaseConfig`, plus the one setting only training cares about."""

    class_weighting: bool = True


# Data
def build_dataset(cfg: PhaseTrainConfig, video_ids: Sequence[str], split: str) -> ClipDataset:
    return ClipDataset(cfg.data_root, clip_len=cfg.clip_len, stride=cfg.stride,
                       img_size=cfg.img_size, num_steps=cfg.num_steps,
                       num_instruments=cfg.num_instruments, split=video_ids,
                       split_name=split)


def build_loader(ds: ClipDataset, cfg: PhaseTrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.workers, generator=make_generator(cfg.seed),
                      worker_init_fn=seed_worker)


def class_weights(step_counts: torch.Tensor, instr_pos: torch.Tensor,
                  n_frames: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Give rare steps and rare instruments more say in the loss.

    Some steps fill most of a video and others appear briefly. Left alone, the model
    does well by ignoring the rare ones. Weighting each by how uncommon it is stops
    that. Anything absent from the training videos gets a weight of 1 and no special
    treatment, since there is nothing to learn from.
    """
    num_steps = step_counts.numel()
    step_w = torch.where(step_counts > 0,
                         n_frames / (num_steps * step_counts.clamp(min=1.0)),
                         torch.ones_like(step_counts))
    pos_w = torch.where(instr_pos > 0,
                        (n_frames - instr_pos) / instr_pos.clamp(min=1.0),
                        torch.ones_like(instr_pos))
    return tuple(step_w.tolist()), tuple(pos_w.tolist())


# Evaluation
@torch.no_grad()
def collect_videos(model: SpatioTemporalMultiTask, loader: DataLoader,
                   device: torch.device) -> list[VideoPrediction]:
    """Run the model over every clip and stitch the results back into whole videos."""
    model.eval()
    agg = LogitAggregator()
    for batch in loader:
        valid = batch["valid"]
        out = model(batch["clip"].to(device), valid.to(device))
        step = out["step"].cpu()
        instr = out["instrument"].cpu()
        for b, video_id in enumerate(batch["video_id"]):
            m = valid[b]
            agg.add(video_id, batch["frame_index"][b][m], step[b][m], batch["step"][b][m],
                    instr[b][m], batch["instrument"][b][m])
    return agg.videos()


def evaluate(model: SpatioTemporalMultiTask, loader: DataLoader, device: torch.device,
             cfg: PhaseTrainConfig, split: str) -> PhaseScore:
    """Score a split, counting every real frame exactly once."""
    return score_videos(collect_videos(model, loader, device), num_steps=cfg.num_steps,
                        num_instruments=cfg.num_instruments,
                        instrument_threshold=cfg.instrument_threshold, split=split)


# Saving and reloading a trained model.
def save_checkpoint(path: str, model: SpatioTemporalMultiTask, model_cfg: PhaseModelConfig,
                    cfg: PhaseTrainConfig, splits: Mapping[str, Sequence[str]],
                    epoch: int, score: PhaseScore) -> None:
    """Save the weights along with the settings and splits that produced them."""
    torch.save({"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION,
                "model": model.state_dict(),
                "model_cfg": config_to_dict(model_cfg),
                "cfg": config_to_dict(cfg),
                "splits": {k: list(v) for k, v in splits.items()},
                "epoch": epoch,
                "selected_by": {"metric": SELECTION_METRIC, "value": score.step_macro_f1},
                "score": dataclasses.asdict(score)}, path)


def load_checkpoint(path: str, map_location: str = "cpu"
                    ) -> tuple[SpatioTemporalMultiTask, PhaseTrainConfig, dict]:
    """Rebuild a saved model. Returns the model, its settings, and the rest of the file."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} v{CHECKPOINT_VERSION} checkpoint "
                         f"(got {payload.get('format')!r} v{payload.get('version')!r})")
    model_cfg = config_from_mapping(payload["model_cfg"], PhaseModelConfig, source=path)
    cfg = config_from_mapping(payload["cfg"], PhaseTrainConfig, source=path)
    # The saved weights replace everything anyway, so there is no point downloading
    # the pretrained ones first.
    model = SpatioTemporalMultiTask(dataclasses.replace(model_cfg, pretrained=False))
    model.load_state_dict(payload["model"])
    return model, cfg, payload


# The training loop itself.
def build_scheduler(opt: torch.optim.Optimizer, cfg: PhaseTrainConfig,
                    steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Ease the learning rate down to zero over the run, along a smooth curve."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = int(WARMUP_FRACTION * total) if cfg.temporal == "transformer" else 0

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def train_one_epoch(model: SpatioTemporalMultiTask, loader: DataLoader,
                    opt: torch.optim.Optimizer,
                    sched: torch.optim.lr_scheduler.LambdaLR,
                    device: torch.device, cfg: PhaseTrainConfig, epoch: int) -> float:
    """One pass over the training clips. Returns the average loss."""
    model.train()
    running = 0.0
    for i, batch in enumerate(loader):
        valid = batch["valid"].to(device)
        out = model(batch["clip"].to(device), valid)
        loss, _ = model.compute_loss(out, batch["step"].to(device),
                                     batch["instrument"].to(device), valid)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"epoch {epoch} batch {i}: the loss came out as "
                               f"{loss.item()}. Stopping here, because carrying on would "
                               "quietly wreck the weights")
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        running += loss.item()
    return running / max(1, len(loader))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="YAML matching PhaseTrainConfig")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    ap.add_argument("--out", default="runs/phase", help="checkpoint directory")
    args = ap.parse_args()

    cfg = load_config(args.config, PhaseTrainConfig,
                      overrides={"epochs": args.epochs, "seed": args.seed})
    seed_everything(cfg.seed)
    device = get_device()
    print(device_report(device))

    splits = resolve_phase_splits(cfg)
    test_ids = splits.get("test", [])
    train_ds = build_dataset(cfg, splits["train"], "train")
    val_ds = build_dataset(cfg, splits["val"], "val")
    print(f"seed {cfg.seed} | video-level split, disjoint | "
          f"train {len(splits['train'])} videos / {len(train_ds)} clips | "
          f"val {len(splits['val'])} videos / {len(val_ds)} clips {list(splits['val'])} | "
          f"test {len(test_ids)} videos {list(test_ids)}")

    if cfg.class_weighting:
        step_counts, instr_pos, n_frames = train_ds.label_counts()
        step_w, pos_w = class_weights(step_counts, instr_pos, n_frames)
        present = [w for w, c in zip(step_w, step_counts.tolist()) if c > 0]
        print(f"class weighting ON (from {n_frames} train frames): step weight "
              f"{min(present):.2f}..{max(present):.2f} over the {len(present)} steps "
              f"present, 1.0 for the absent ones | instrument pos_weight "
              f"{min(pos_w):.2f}..{max(pos_w):.2f}")
    else:
        step_w, pos_w = (), ()
        print("class weighting OFF: both heads use unweighted losses")

    model_cfg = PhaseModelConfig.from_config(cfg, step_class_weight=step_w,
                                             instrument_pos_weight=pos_w)
    model = SpatioTemporalMultiTask(model_cfg).to(device)

    train_loader = build_loader(train_ds, cfg, shuffle=True)
    val_loader = build_loader(val_ds, cfg, shuffle=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = build_scheduler(opt, cfg, len(train_loader))

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    best = -1.0
    best_epoch = -1
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, opt, sched, device, cfg, epoch)
        score = evaluate(model, val_loader, device, cfg, "val")
        print(f"epoch {epoch:02d} | {model.backend} | train loss {loss:.4f} | "
              f"{score.describe()}")
        if score.step_macro_f1 > best:
            best, best_epoch = score.step_macro_f1, epoch
            save_checkpoint(best_path, model, model_cfg, cfg, splits, epoch, score)

    print(f"kept epoch {best_epoch} of {model.backend}, chosen by {SELECTION_METRIC} = "
          f"{best:.3f}. That was the score I picked on, so it flatters the model. "
          f"Saved to {best_path}")

    if test_ids:
        best_model, _, _ = load_checkpoint(best_path)
        test_loader = build_loader(build_dataset(cfg, test_ids, "test"), cfg, shuffle=False)
        score = evaluate(best_model.to(device), test_loader, device, cfg, "test")
        print(f"{best_model.backend} | {score.describe()}. These videos were kept out of "
              "both the training and the choice of epoch")
    else:
        print("no test split is set up, so nothing here is a held-out score")


if __name__ == "__main__":
    main()
