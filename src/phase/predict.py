"""
Score a trained phase model on one split and draw its step timeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")  # no display in CI or over ssh

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from src.common.device import device_report, get_device  # noqa: E402
from src.common.seed import seed_everything  # noqa: E402
from src.data.splits import SPLIT_PROVENANCE  # noqa: E402
from src.phase.dataset import resolve_phase_splits  # noqa: E402
from src.phase.metrics import PhaseScore, VideoPrediction, score_videos  # noqa: E402
from src.phase.model import SpatioTemporalMultiTask  # noqa: E402
from src.phase.train import (build_dataset, build_loader, collect_videos,  # noqa: E402
                             load_checkpoint)


def render_timeline(videos: list[VideoPrediction], num_steps: int, out_path: str,
                    title: str) -> None:
    fig, axes = plt.subplots(len(videos), 1, squeeze=False, layout="constrained",
                             figsize=(11.5, 2.1 * len(videos) + 1.6))
    cmap = plt.get_cmap("tab20", num_steps)
    norm = BoundaryNorm(np.arange(num_steps + 1) - 0.5, num_steps)

    for i, (ax, video) in enumerate(zip(axes[:, 0], videos)):
        pred = video.step_logits.argmax(-1)
        true = video.step_target
        ax.imshow(np.stack([true, pred]), aspect="auto", interpolation="nearest",
                  cmap=cmap, norm=norm, extent=(0, len(true), 2, 0))
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["ground truth", "prediction"], fontsize=10)
        if i == len(videos) - 1:  # one shared x axis label; per-panel labels collide
            ax.set_xlabel("frame index", fontsize=11)
        acc = float((pred == true).mean())
        ax.set_title(f"{video.video_id} — {len(true)} frames, frame accuracy {acc:.3f}, "
                     f"{int((np.diff(pred) != 0).sum())} predicted step changes "
                     f"({int((np.diff(true) != 0).sum())} in the ground truth)",
                     fontsize=10.5, loc="left")
        ax.axhline(1.0, color="white", lw=2)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, 0].tolist(),
                       ticks=range(num_steps), fraction=0.03, pad=0.01)
    bar.set_label(f"surgical step id (0-{num_steps - 1})", fontsize=10)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12.5)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def random_reference(videos: list[VideoPrediction], seed: int, **score_kwargs) -> PhaseScore:
    """Score uniform-random logits on the same frames as a measured chance floor."""
    rng = np.random.default_rng(seed)
    fake = [VideoPrediction(
        video_id=v.video_id,
        step_logits=rng.standard_normal(v.step_logits.shape).astype(np.float32),
        step_target=v.step_target,
        instr_logits=rng.standard_normal(v.instr_logits.shape).astype(np.float32),
        instr_target=v.instr_target) for v in videos]
    return score_videos(fake, **score_kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="best.pt from src.phase.train")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="test is the held-out split; val is what selected the checkpoint")
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg, payload = load_checkpoint(args.checkpoint)
    seed_everything(cfg.seed)
    device = get_device()
    print(device_report(device))

    splits = resolve_phase_splits(cfg)
    if args.split not in splits:
        raise SystemExit(f"{cfg.data_root}: no {args.split!r} split; "
                         f"available: {sorted(splits)}")
    ids = splits[args.split]
    dataset = build_dataset(cfg, ids, args.split)
    loader = build_loader(dataset, cfg, shuffle=False)

    scoring = dict(num_steps=cfg.num_steps, num_instruments=cfg.num_instruments,
                   instrument_threshold=cfg.instrument_threshold)
    videos = collect_videos(model.to(device), loader, device)
    score = score_videos(videos, split=args.split, **scoring)

    untrained = SpatioTemporalMultiTask(
        dataclasses.replace(model.cfg, pretrained=False)).to(device)
    base = score_videos(collect_videos(untrained, loader, device),
                        split=f"{args.split} (untrained)", **scoring)
    chance = random_reference(videos, cfg.seed, split=f"{args.split} (uniform random)",
                              **scoring)

    os.makedirs(args.out, exist_ok=True)
    fig_path = os.path.join(args.out, "phase_timeline.png")
    render_timeline(videos, cfg.num_steps, fig_path,
                    f"{model.backend} — [{args.split}] {list(ids)}, "
                    f"{SPLIT_PROVENANCE[args.split]}")

    summary = {
        "backend": model.backend,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_provenance": SPLIT_PROVENANCE[args.split],
        "split_video_ids": {name: list(v) for name, v in splits.items()},
        "split_sizes": {name: len(v) for name, v in splits.items()},
        "n_clips_scored": len(dataset),
        "seed": cfg.seed,
        "selected_epoch": payload.get("epoch"),
        "selected_by": payload.get("selected_by"),
        "config": payload["cfg"],
        "score": dataclasses.asdict(score),
        "untrained_baseline": dataclasses.asdict(base),
        "uniform_random_baseline": dataclasses.asdict(chance),
        "per_video": [{"video_id": v.video_id, "n_frames": int(v.step_target.size),
                       "accuracy": float((v.step_logits.argmax(-1) == v.step_target).mean())}
                      for v in videos],
    }
    metrics_path = os.path.join(args.out, "phase_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{model.backend} | {score.describe()} — {SPLIT_PROVENANCE[args.split]}")
    print(f"same architecture untrained on the same frames | {base.describe()}")
    print(f"uniform-random logits on the same frames | {chance.describe()}")
    print(f"wrote {metrics_path}, {fig_path}")


if __name__ == "__main__":
    main()
