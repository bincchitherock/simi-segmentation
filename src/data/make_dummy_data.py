from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.phantom import N_INSTRUMENT_KINDS, N_STEPS, Frame, render_video
from src.data.splits import assign_splits, write_splits


def _split_sizes(n: int, val_frac: float = 0.2) -> dict[str, int]:
    held = max(1, round(val_frac * n))
    if n - 2 * held < 1:
        raise ValueError(f"{n} videos cannot be split into train/val/test at {val_frac}")
    return {"train": n - 2 * held, "val": held, "test": held}


def _rule(sizes: dict[str, int], why: str) -> str:
    order = ", then ".join(f"{n} to {name}" for name, n in sizes.items())
    return f"ids sorted ascending, {order}. {why}"


def make_phase(root: Path, *, n_videos: int, n_frames: int, size: int, seed: int) -> None:
    """Render `n_videos` coherent clips with piecewise-constant step labels."""
    root.mkdir(parents=True, exist_ok=True)
    records, all_frames = [], []
    for v in range(n_videos):
        vid = f"video_{v:02d}"
        vdir = root / vid
        if vdir.exists():
            shutil.rmtree(vdir)  # a shorter run must not leave frames behind
        vdir.mkdir(parents=True)

        frames = render_video(np.random.default_rng([seed, 1, v]), n_frames, size)
        names = []
        for i, f in enumerate(frames):
            name = f"frame_{i:06d}.jpg"
            Image.fromarray(f.image).save(vdir / name, quality=92)
            names.append(name)
        all_frames.extend(frames)
        records.append({
            "video_id": vid,
            "frames_dir": vid,
            "fps": 1,
            "frames": names,
            "steps": [f.step for f in frames],
            "instruments": [f.presence(N_INSTRUMENT_KINDS) for f in frames],
        })

    (root / "videos.json").write_text(json.dumps(records, indent=1) + "\n")
    sizes = _split_sizes(n_videos, val_frac=0.25)
    write_splits(root / "splits.json",
                 assign_splits([r["video_id"] for r in records], sizes), unit="video_id",
                 rule=_rule(sizes, "Contiguous and deterministic, so it can be checked by hand. "
                                   "Split by video: overlapping clips share frames."))
    _report_phase(root, records, all_frames)


def make_seg(root: Path, *, n_clips: int, n_frames: int, size: int, seed: int) -> None:
    """Render `n_clips` short clips as image/mask pairs"""
    for sub in ("images", "masks"):
        if (root / sub).exists():
            shutil.rmtree(root / sub)
        (root / sub).mkdir(parents=True)

    pairs, all_frames = [], []
    for c in range(n_clips):
        cid = f"clip_{c:02d}"
        frames = render_video(np.random.default_rng([seed, 2, c]), n_frames, size)
        for i, f in enumerate(frames):
            name = f"{cid}_{i:06d}.png"
            Image.fromarray(f.image).save(root / "images" / name)
            Image.fromarray(f.mask).save(root / "masks" / name)
            pairs.append({"image": f"images/{name}", "mask": f"masks/{name}",
                          "video_id": cid, "frame_index": i})
        all_frames.extend(frames)

    (root / "pairs.json").write_text(json.dumps(pairs, indent=1) + "\n")
    sizes = _split_sizes(n_clips, val_frac=0.2)
    write_splits(root / "splits.json",
                 assign_splits(sorted({p["video_id"] for p in pairs}), sizes), unit="video_id",
                 rule=_rule(sizes, "Split by clip, not by frame: frames inside a clip are "
                                   "near-duplicates and a per-frame split would leak. val "
                                   "selects the checkpoint, test is scored once."))
    _report_seg(root, pairs, all_frames)


def _report_phase(root: Path, records: list[dict[str, Any]], frames: list[Frame]) -> None:
    steps = np.array([f.step for f in frames])
    instr = np.array([f.presence(N_INSTRUMENT_KINDS) for f in frames])
    print(f"phase -> {root}")
    print(f"  {len(records)} videos, {len(frames)} frames total")
    print(f"  steps: {len(np.unique(steps))}/{N_STEPS} distinct, "
          f"most frequent covers {np.bincount(steps).max() / len(steps):.0%} of frames")
    print(f"  instruments: {instr.sum(1).mean():.2f} present per frame on average, "
          f"{int((instr.sum(1) == 0).sum())} frames with none")


def _report_seg(root: Path, pairs: list[dict[str, Any]], frames: list[Frame]) -> None:
    area = np.array([(f.mask > 0).mean() for f in frames])
    print(f"seg -> {root}")
    print(f"  {len(pairs)} image/mask pairs from {len({p['video_id'] for p in pairs})} clips")
    print(f"  instrument covers {area.mean():.1%} of the frame on average "
          f"(max {area.max():.1%}); {int((area == 0).sum())} empty masks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data", help="root directory to write into")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase-videos", type=int, default=12)
    ap.add_argument("--phase-frames", type=int, default=48)
    ap.add_argument("--phase-size", type=int, default=96)
    ap.add_argument("--seg-clips", type=int, default=10)
    ap.add_argument("--seg-frames", type=int, default=12)
    ap.add_argument("--seg-size", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    make_phase(out / "phase", n_videos=args.phase_videos, n_frames=args.phase_frames,
               size=args.phase_size, seed=args.seed)
    make_seg(out / "seg", n_clips=args.seg_clips, n_frames=args.seg_frames,
             size=args.seg_size, seed=args.seed)
    print("\nThis is a procedural phantom, not surgery. It encodes the step in the mucosa "
          "tint and the instrument pair, so a model can learn it; any metric measured on it "
          "describes the pipeline, not clinical performance.")


if __name__ == "__main__":
    main()
