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
WARMUP_FRACTION = 0.05  # transformer only; post-norm encoders spike without it


@dataclass(frozen=True)
class PhaseTrainConfig(PhaseConfig):
    """PhaseConfig plus the one knob the training loop owns."""

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
    Inverse-frequency step weights and per-instrument pos_weight.
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
    """Score every real frame of the split exactly once."""
    return score_videos(collect_videos(model, loader, device), num_steps=cfg.num_steps,
                        num_instruments=cfg.num_instruments,
                        instrument_threshold=cfg.instrument_threshold, split=split)


# Checkpoints
def save_checkpoint(path: str, model: SpatioTemporalMultiTask, model_cfg: PhaseModelConfig,
                    cfg: PhaseTrainConfig, splits: Mapping[str, Sequence[str]],
                    epoch: int, score: PhaseScore) -> None:
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
    """Rebuild the model a checkpoint holds; returns (model, cfg, payload)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} v{CHECKPOINT_VERSION} checkpoint "
                         f"(got {payload.get('format')!r} v{payload.get('version')!r})")
    model_cfg = config_from_mapping(payload["model_cfg"], PhaseModelConfig, source=path)
    cfg = config_from_mapping(payload["cfg"], PhaseTrainConfig, source=path)
    # the state dict overwrites any pretrained weights, so skip downloading them
    model = SpatioTemporalMultiTask(dataclasses.replace(model_cfg, pretrained=False))
    model.load_state_dict(payload["model"])
    return model, cfg, payload


# Training
def build_scheduler(opt: torch.optim.Optimizer, cfg: PhaseTrainConfig,
                    steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay, with linear warmup for the transformer variant."""
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
    model.train()
    running = 0.0
    for i, batch in enumerate(loader):
        valid = batch["valid"].to(device)
        out = model(batch["clip"].to(device), valid)
        loss, _ = model.compute_loss(out, batch["step"].to(device),
                                     batch["instrument"].to(device), valid)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"epoch {epoch} batch {i}: loss is {loss.item()} -- "
                               "aborting rather than silently training on NaN")
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

    print(f"selected epoch {best_epoch} of {model.backend} by {SELECTION_METRIC} = "
          f"{best:.3f} (this is a selection score on val, not a held-out estimate) "
          f"-> {best_path}")

    if test_ids:
        best_model, _, _ = load_checkpoint(best_path)
        test_loader = build_loader(build_dataset(cfg, test_ids, "test"), cfg, shuffle=False)
        score = evaluate(best_model.to(device), test_loader, device, cfg, "test")
        print(f"{best_model.backend} | {score.describe()} — held out from both training "
              "and checkpoint selection")
    else:
        print("no test split configured: no held-out estimate was computed")


if __name__ == "__main__":
    main()
