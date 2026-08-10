"""
Labels a whole clip from one click, using SAM 2's video tracking. Never run here.

Point at the instrument once on the first frame, and SAM 2 follows it through the
rest of the clip, writing a mask for every frame. The idea is to cut down the hand
labelling: correct what comes out rather than draw each mask from nothing.

This needs the SAM 2 package, which is not installed, so none of this has ever run.
It says so when you start it. Treat what it writes as a rough draft to review, never
as ground truth to train on directly.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from src.common.device import device_report, get_device
from src.common.lora import (apply_lora, load_adapter_file, mark_only_lora_trainable,
                             read_adapter_spec)
from src.segmentation.sam2_lora import SAM2_DECODER_ALSO_TRAIN, SAM2_INSTALL_HINT

SAM2_JPEG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def sam2_frame_names(frames_dir: str) -> list[str]:
    """Frame files in order. SAM 2 insists the names be plain numbers, so I added a check."""
    names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] in SAM2_JPEG_EXTS]
    if not names:
        raise SystemExit(f"no JPEG frames in {frames_dir}")
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError as exc:
        raise SystemExit(
            f"{frames_dir}: SAM 2 needs frames named as plain numbers, like 0.jpg or "
            f"00001.jpg, because it puts them in order by reading the name as a number. "
            f"I found {sorted(names)[:3]}. Rename or link them, for example with "
            f"ffmpeg -i v.mp4 -q:v 2 -start_number 0 out/'%05d.jpg'.") from exc
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", required=True,
                    help="folder of JPEGs named as numbers, like 0.jpg or 00001.jpg")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-cfg", dest="model_cfg", required=True,
                    help="path inside the sam2 package, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--lora", default=None, help="adapter file written by train_sam2.py")
    ap.add_argument("--seed-frame", type=int, default=0)
    ap.add_argument("--seed-point", type=float, nargs=2, default=None,
                    help="x y of a click on the instrument, on the starting frame")
    ap.add_argument("--seed-box", type=float, nargs=4, default=None,
                    help="x0 y0 x1 y1 box around the instrument, on the starting frame")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--out", default="autolabels")
    args = ap.parse_args()

    if (args.seed_point is None) == (args.seed_box is None):
        raise SystemExit("give exactly one of --seed-point x y or --seed-box x0 y0 x1 y1")

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise SystemExit(SAM2_INSTALL_HINT) from exc

    device = get_device()
    print(device_report(device))
    print("[autolabel] UNVERIFIED PATH. I wrote this against facebookresearch/sam2 @ main "
          "and never ran it, because sam2 is not installed here.")

    frame_names = sam2_frame_names(args.frames_dir)
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint,
                                           device=str(device),
                                           apply_postprocessing=False)
    predictor.sam_mask_decoder.dynamic_multimask_via_stability = False

    if args.lora:
        spec = read_adapter_spec(args.lora)
        apply_lora(predictor, spec)
        mark_only_lora_trainable(predictor, also_train=SAM2_DECODER_ALSO_TRAIN)
        print(f"[autolabel] {load_adapter_file(predictor, args.lora).describe()}")

    state = predictor.init_state(video_path=args.frames_dir)

    if args.seed_point is not None:
        predictor.add_new_points_or_box(
            state, frame_idx=args.seed_frame, obj_id=args.obj_id,
            points=np.array([args.seed_point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32))
    else:
        predictor.add_new_points_or_box(
            state, frame_idx=args.seed_frame, obj_id=args.obj_id,
            box=np.array(args.seed_box, dtype=np.float32))

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for row, obj_id in enumerate(obj_ids):
            mask = (mask_logits[row, 0] > 0).cpu().numpy().astype(np.uint8) * 255
            name = f"mask_obj{int(obj_id)}_{frame_idx:06d}.png"
            Image.fromarray(mask).save(os.path.join(args.out, name))
            manifest.append({"frame": frame_names[frame_idx], "mask": name,
                             "obj_id": int(obj_id), "source": "sam2_autolabel"})

    with open(os.path.join(args.out, "encord_import.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} masks and encord_import.json into {args.out}/. "
          "Check and fix them by hand before adding any of it to the training set.")


if __name__ == "__main__":
    main()
