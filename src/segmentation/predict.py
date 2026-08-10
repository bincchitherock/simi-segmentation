from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.common.config import (SegConfig, config_from_mapping, config_to_dict,  # noqa: E402
                               load_config)
from src.common.device import device_report, get_device  # noqa: E402
from src.common.lora import load_adapter_state_dict  # noqa: E402
from src.common.metrics import dice_iou  # noqa: E402
from src.common.seed import seed_everything, seed_worker  # noqa: E402
from src.data.splits import SPLIT_PROVENANCE  # noqa: E402
from src.segmentation.dataset import (MaskDataset, collate, resolve_seg_splits,  # noqa: E402
                                      to_display)
from src.segmentation.sam2_lora import (SegmenterModule, build_segmenter,  # noqa: E402
                                        forward_batch)

GT_COLOUR = "#00d5ff"
PRED_COLOUR = "#7cff3c"
FIGURE_DPI = 84

MODEL_FIELDS = ("model", "seed", "lora_rank", "lora_alpha", "lora_dropout",
                "train_decoder", "img_size", "checkpoint", "model_cfg")


def config_hash(cfg: SegConfig) -> str:
    """Stable digest of the config, so a metrics file can be tied to its run."""
    blob = json.dumps(config_to_dict(cfg), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def load_run(checkpoint: str, config: str | None) -> tuple[SegConfig, dict[str, Any]]:

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "cfg" not in payload:
        raise SystemExit(f"{checkpoint}: no 'cfg' — not written by src.segmentation.train_sam2")
    trained = config_from_mapping(payload["cfg"], SegConfig, source=f"{checkpoint}:cfg")
    if not config:
        return trained, payload

    cfg = load_config(config, SegConfig)
    differ = [f"{k}: {getattr(trained, k)!r} in the checkpoint, {getattr(cfg, k)!r} in "
              f"{config}" for k in MODEL_FIELDS if getattr(trained, k) != getattr(cfg, k)]
    if differ:
        raise SystemExit(f"{config} would build a different model than {checkpoint} was "
                         "trained on, and the adapter file holds only the trainable "
                         "tensors:\n  " + "\n  ".join(differ))
    return cfg, payload


def score_split(model: SegmenterModule, loader: DataLoader,
                device: torch.device) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device).cpu()
            for i, pair_id in enumerate(batch["pair_id"]):
                s = dice_iou(logits[i:i + 1], batch["mask"][i:i + 1])
                rows.append({"pair_id": pair_id, "dice": s.dice, "iou": s.iou,
                             "scored": s.n_scored == 1,
                             "pred": (logits[i, 0] > 0).numpy()})
    return rows


def _stats(vals: list[float]) -> dict[str, Any]:
    return {"mean": statistics.fmean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
            "n": len(vals)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:

    scored = [r for r in rows if r["scored"]]
    out: dict[str, Any] = {"n_samples": len(rows), "n_scored": len(scored),
                 "n_excluded_empty_pairs": len(rows) - len(scored)}
    for key in ("dice", "iou"):
        out[key] = _stats([r[key] for r in scored])
    return out


def rank_by_dice(rows: list[dict[str, Any]], n: int) -> list[int]:
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["dice"], i))
    picks = np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)
    return [order[p] for p in dict.fromkeys(picks.tolist())]


def render_grid(dataset: MaskDataset, rows: list[dict[str, Any]], idxs: list[int],
                out_path: str, title: str) -> None:
    n = len(idxs)
    height = 2.9 * n + 1.4
    fig, axes = plt.subplots(n, 4, figsize=(11.5, height), squeeze=False)
    titles = ("input (what the model sees)", "ground truth", "prediction",
              "overlay: GT vs prediction")

    for r, idx in enumerate(idxs):
        sample = dataset[idx]
        img = to_display(sample["image"])
        gt = sample["mask"][0].numpy()
        row = rows[idx]
        pred = row["pred"]

        for c, panel in enumerate((img, gt, pred, img)):
            ax = axes[r][c]
            if panel.ndim == 3:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11, pad=8)
        # ground truth thicker and underneath, so a matching prediction leaves the cyan visible
        for mask, colour, lw in ((gt, GT_COLOUR, 2.6), (pred.astype(float), PRED_COLOUR, 1.2)):
            if mask.any():
                axes[r][3].contour(mask, levels=[0.5], colors=colour, linewidths=lw)

        axes[r][0].set_ylabel(f"{os.path.basename(row['pair_id'])}\n"
                              f"Dice {row['dice']:.3f}\nIoU {row['iou']:.3f}", fontsize=9)

    handles = [plt.Line2D([], [], color=GT_COLOUR, lw=2, label="ground truth"),
               plt.Line2D([], [], color=PRED_COLOUR, lw=2, label="prediction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0.35 / height, 1, 1 - 0.75 / height))
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def render_curve(history: dict[str, Any], out_path: str, title: str) -> None:
    steps = [p["step"] for p in history["loss"]]
    losses = [p["loss"] for p in history["loss"]]
    evals = history["eval"]

    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(steps, losses, lw=0.8, color="#8892b0", label="per-step loss")
    window = max(1, len(losses) // 40)
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    top.plot(steps[window - 1:], smooth, lw=2.0, color="#e05252",
             label=f"moving average ({window} steps)")
    top.set_ylabel("Dice + BCE loss (train)", fontsize=11)
    top.set_title("training loss on the train split", fontsize=11.5, loc="left")
    top.legend(fontsize=10, frameon=False)
    top.grid(alpha=0.25)

    scored = {e["n_scored"] for e in evals}
    bot.plot([e["step"] for e in evals], [e["dice"] for e in evals], lw=1.6,
             color="#2f7fd1", zorder=1)
    bot.scatter([e["step"] for e in evals], [e["dice"] for e in evals], s=26,
                color="#2f7fd1", zorder=2,
                label=f"mean over the same {min(scored)} masks every epoch"
                      if len(scored) == 1 else "mean over a varying population")
    selected = next((e for e in evals if e["epoch"] == history["selected_epoch"]), None)
    if selected:
        bot.axvline(selected["step"], color="#c44", ls="--", lw=1.2,
                    label=f"selected epoch {selected['epoch']} (Dice {selected['dice']:.3f})")
    bot.set_xlabel("optimizer step", fontsize=11)
    bot.set_ylabel("Dice (val)", fontsize=11)
    bot.set_title(f"validation Dice on held-out clips {history['val_split']}",
                  fontsize=11.5, loc="left")
    bot.legend(fontsize=9.5, frameon=False, loc="lower right")
    bot.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="adapters.pt from train_sam2.py")
    ap.add_argument("--config", default=None,
                    help="override the config stored in the checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="test is the held-out split; val is what selected the checkpoint")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="", help="prepended to every written filename")
    ap.add_argument("--max-figures", dest="max_figures", type=int, default=6)
    ap.add_argument("--baseline", action="store_true",
                    help="also score and render the same model before training")
    ap.add_argument("--history", default=None,
                    help="history.json from train_sam2.py; renders training_curve.png")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg, payload = load_run(args.checkpoint, args.config)
    seed_everything(cfg.seed)
    device = get_device()
    print(device_report(device))

    splits = resolve_seg_splits(cfg)
    if args.split not in splits:
        raise SystemExit(f"{cfg.data_root}: no {args.split!r} split is declared; "
                         f"available: {sorted(splits)}")
    ids = splits[args.split]
    provenance = SPLIT_PROVENANCE[args.split]
    dataset = MaskDataset(data_root=cfg.data_root, split=ids, img_size=cfg.img_size,
                          prompt_kind=cfg.prompt_kind, seed=cfg.seed, name=args.split)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, collate_fn=collate,
                        worker_init_fn=seed_worker)

    print("[predict] building the model the adapters go into")
    model = build_segmenter(cfg, device)
    base_model = None
    if args.baseline:
        print("[predict] building a second, never-adapted copy for the untrained baseline")
        base_model = build_segmenter(cfg, device)
    os.makedirs(args.out, exist_ok=True)

    def path(name: str) -> str:
        return os.path.join(args.out, args.prefix + name)

    baseline: dict[str, Any] | None = None
    if base_model is not None:
        base_rows = score_split(base_model, loader, device)
        baseline = summarise(base_rows)

    print(f"[predict] {load_adapter_state_dict(model.adapter_root, payload).describe()}")
    rows = score_split(model, loader, device)
    summary = summarise(rows)
    shown = rank_by_dice(rows, args.max_figures)

    steps = payload.get("train_steps")
    trained_title = (f"{model.backend} — [{args.split}] split {list(ids)}, "
                     f"{len(rows)} masks, {steps} training steps, "
                     f"{cfg.prompt_kind} prompt from the ground-truth mask\n"
                     f"{len(shown)} of {len(rows)} masks, evenly spaced by Dice rank "
                     f"(worst at the top) — {provenance}")
    render_grid(dataset, rows, shown, path("qualitative.png"), trained_title)

    written = [path("qualitative.png"), path("metrics.json")]
    if baseline is not None:
        floor = ("adapters at init, i.e. zero-shot SAM 2" if cfg.model == "sam2"
                 else "random weights, adapters at init")
        render_grid(dataset, base_rows, shown, path("untrained_baseline.png"),
                    f"{model.backend} BEFORE training ({floor}) "
                    f"— same {len(shown)} [{args.split}] masks as the trained figure\n"
                    f"mean Dice {baseline['dice']['mean']:.3f} over "
                    f"{baseline['n_scored']}/{baseline['n_samples']} masks")
        written.append(path("untrained_baseline.png"))
    if args.history:
        with open(args.history) as f:
            history = json.load(f)
        render_curve(history, path("training_curve.png"),
                     f"{model.backend} training run, seed {cfg.seed}")
        written.append(path("training_curve.png"))

    summary.update({
        "backend": model.backend,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_video_ids": list(ids),
        "split_provenance": provenance,
        "selection_split_video_ids": list(payload.get("val_split", ())),
        "seed": cfg.seed,
        "n_training_steps": steps,
        "selected_epoch": payload.get("epoch"),
        "config_hash": config_hash(cfg),
        "config": config_to_dict(cfg),
        "prompt_kind": cfg.prompt_kind,
        "prompt_source": "sampled from the ground-truth mask (oracle prompt)",
        "figure_pair_ids": [rows[i]["pair_id"] for i in shown],
        "untrained_baseline": baseline,
        "samples": [{k: v for k, v in r.items() if k != "pred"} for r in rows],
    })
    with open(path("metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    line = (f"{model.backend} | [{args.split}] {list(ids)} | "
            f"Dice {summary['dice']['mean']:.3f} +/- {summary['dice']['std']:.3f} | "
            f"IoU {summary['iou']['mean']:.3f} +/- {summary['iou']['std']:.3f} "
            f"over {summary['n_scored']}/{summary['n_samples']} masks, each prompted with "
            f"a {cfg.prompt_kind} taken from its ground-truth mask — {provenance}")
    if baseline is not None:
        line += (f"\nsame model BEFORE training: Dice {baseline['dice']['mean']:.3f} "
                 f"| IoU {baseline['iou']['mean']:.3f} over "
                 f"{baseline['n_scored']}/{baseline['n_samples']} masks")
    print(line)
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
