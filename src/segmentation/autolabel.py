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
    names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] in SAM2_JPEG_EXTS]
    if not names:
        raise SystemExit(f"no JPEG frames in {frames_dir}")
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError as exc:
        raise SystemExit(
            f"{frames_dir}: SAM2 requires frames named '<frame_index>.jpg' "
            f"(0.jpg, 00001.jpg, ...) because it sorts by int(stem); found "
            f"{sorted(names)[:3]}. Rename or symlink them, e.g. "
            f"ffmpeg -i v.mp4 -q:v 2 -start_number 0 out/'%05d.jpg'.") from exc
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", required=True,
                    help="dir of JPEGs named <frame_index>.jpg (0.jpg, 00001.jpg ...)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-cfg", dest="model_cfg", required=True,
                    help="sam2-package-relative, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--lora", default=None, help="adapters written by train_sam2.py")
    ap.add_argument("--seed-frame", type=int, default=0)
    ap.add_argument("--seed-point", type=float, nargs=2, default=None,
                    help="x y of a foreground click on the seed frame")
    ap.add_argument("--seed-box", type=float, nargs=4, default=None,
                    help="x0 y0 x1 y1 box on the seed frame")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--out", default="autolabels")
    args = ap.parse_args()

    if (args.seed_point is None) == (args.seed_box is None):
        raise SystemExit("give exactly one of --seed-point x y / --seed-box x0 y0 x1 y1")

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise SystemExit(SAM2_INSTALL_HINT) from exc

    device = get_device()
    print(device_report(device))
    print("[autolabel] UNVERIFIED PATH: written against facebookresearch/sam2 @ main "
          "but never executed in this repo (sam2 not installed here).")

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
    print(f"wrote {len(manifest)} masks + encord_import.json -> {args.out}/  "
          f"(review/correct them, then add to the training set)")


if __name__ == "__main__":
    main()
