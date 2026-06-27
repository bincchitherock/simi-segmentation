from __future__ import annotations

import argparse
import json
import os

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="dir of frame_000001.jpg ...")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-cfg", required=True)
    ap.add_argument("--lora", default=None, help="optional LoRA adapters from train_sam2.py")
    ap.add_argument("--seed-frame", type=int, default=0)
    ap.add_argument("--seed-point", type=float, nargs=2, default=None,
                    help="x y of a foreground click on the seed frame")
    ap.add_argument("--seed-box", type=float, nargs=4, default=None,
                    help="x0 y0 x1 y1 box on the seed frame")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--out", default="autolabels")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    from src.common.device import get_device, device_report
    from src.common.lora import inject_lora

    device = get_device()
    print(device_report(device))

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=str(device))

    if args.lora:
        inject_lora(predictor.image_encoder)
        state = torch.load(args.lora, map_location="cpu")["lora"]
        missing, unexpected = predictor.load_state_dict(state, strict=False)
        print(f"loaded LoRA: {len(state)} tensors "
              f"({len(missing)} missing / {len(unexpected)} unexpected keys)")

    state = predictor.init_state(video_path=args.frames_dir)

    if args.seed_point is not None:
        points = np.array([args.seed_point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(state, frame_idx=args.seed_frame,
                                        obj_id=args.obj_id, points=points, labels=labels)
    elif args.seed_box is not None:
        predictor.add_new_points_or_box(state, frame_idx=args.seed_frame,
                                        obj_id=args.obj_id,
                                        box=np.array(args.seed_box, dtype=np.float32))
    else:
        raise SystemExit("Provide --seed-point x y or --seed-box x0 y0 x1 y1")

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    files = sorted(os.listdir(args.frames_dir))
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0).cpu().numpy().astype(np.uint8)[0] * 255
        name = f"mask_{frame_idx:06d}.png"
        Image.fromarray(mask).save(os.path.join(args.out, name))
        manifest.append({"frame": files[frame_idx] if frame_idx < len(files) else frame_idx,
                         "mask": name, "obj_id": args.obj_id, "source": "sam2_autolabel"})

    with open(os.path.join(args.out, "encord_import.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} masks + encord_import.json -> {args.out}/  "
          f"(review/correct in Encord, then add to the training set)")

if __name__ == "__main__":
    main()
