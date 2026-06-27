from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.device import get_device, device_report
from src.common.lora import lora_state_dict
from src.segmentation.sam2_lora import build_segmenter, dice_bce_loss
from src.segmentation.dataset import MaskDataset, SyntheticMaskDataset
from src.phase.eval import dice_iou

def collate(batch):

    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    return {"image": images, "mask": masks,
            "point_coords": [b["point_coords"] for b in batch],
            "point_labels": [b["point_labels"] for b in batch],
            "box": [b["box"] for b in batch]}

def forward_batch(model, batch, device):

    logits = []
    for i in range(batch["image"].shape[0]):
        img = batch["image"][i:i + 1].to(device)
        pc = batch["point_coords"][i]
        pl = batch["point_labels"][i]
        box = batch["box"][i]
        out = model(img,
                    point_coords=pc.to(device) if pc is not None else None,
                    point_labels=pl.to(device) if pl is not None else None,
                    boxes=box.to(device) if box is not None else None)
        logits.append(out)
    return torch.cat(logits, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-cfg", dest="model_cfg", default=None)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="runs/seg")
    args = ap.parse_args()

    cfg = {"lr": 5e-6, "weight_decay": 0.1, "epochs": 40, "batch_size": 2,
           "img_size": 1024, "prompt_kind": "point", "lora_rank": 8}
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    args.lora_rank = cfg.get("lora_rank", args.lora_rank)

    device = get_device()
    print(device_report(device))

    if args.synthetic:
        train = SyntheticMaskDataset(n=8, img_size=128, seed=0)
        val = SyntheticMaskDataset(n=4, img_size=128, seed=1)
        cfg["lr"] = 1e-3
    else:
        train = MaskDataset(cfg["data_root"], img_size=cfg["img_size"],
                            prompt_kind=cfg["prompt_kind"])
        val = train

    model = build_segmenter(args, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    bs = cfg["batch_size"]
    tl = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate)
    vl = DataLoader(val, batch_size=bs, shuffle=False, collate_fn=collate)

    os.makedirs(args.out, exist_ok=True)
    max_steps = args.steps if args.steps is not None else cfg["epochs"] * len(tl)
    step = 0
    model.train()
    while step < max_steps:
        for batch in tl:
            logits = forward_batch(model, batch, device)
            loss = dice_bce_loss(logits, batch["mask"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % max(1, max_steps // 10) == 0 or step == 1:
                print(f"step {step:04d}/{max_steps} | loss {loss.item():.4f}")
            if step >= max_steps:
                break

    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for batch in vl:
            logits = forward_batch(model, batch, device)
            d, j = dice_iou(logits.cpu(), batch["mask"])
            dices.append(d); ious.append(j)
    print(f"val Dice {sum(dices)/len(dices):.3f} | IoU {sum(ious)/len(ious):.3f}")

    torch.save({"lora": lora_state_dict(model), "cfg": cfg},
               os.path.join(args.out, "lora_adapters.pt"))
    print(f"saved adapters -> {args.out}/lora_adapters.pt")

if __name__ == "__main__":
    main()
