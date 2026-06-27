from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.device import get_device, device_report
from src.phase.model import SpatioTemporalMultiTask, PhaseModelConfig
from src.phase.dataset import ClipDataset, SyntheticClipDataset
from src.phase.eval import macro_f1_multiclass, macro_f1_multilabel

def build_loaders(args, cfg):
    if args.synthetic:
        ns = cfg["num_steps"]; ni = cfg["num_instruments"]
        train = SyntheticClipDataset(n=8, clip_len=16, img_size=64,
                                     num_steps=ns, num_instruments=ni, seed=0)
        val = SyntheticClipDataset(n=4, clip_len=16, img_size=64,
                                   num_steps=ns, num_instruments=ni, seed=1)
    else:
        common = dict(data_root=cfg["data_root"], clip_len=cfg["clip_len"],
                      stride=cfg["stride"], img_size=cfg["img_size"],
                      num_instruments=cfg["num_instruments"])
        train = ClipDataset(**common, split=cfg.get("train_split"))
        val = ClipDataset(**common, split=cfg.get("val_split"))
    bs = cfg.get("batch_size", 2)
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 0)),
            DataLoader(val, batch_size=bs, shuffle=False, num_workers=cfg.get("workers", 0)))

@torch.no_grad()
def evaluate(model, loader, device, num_steps):
    model.eval()
    step_logits_all, step_t_all, instr_logits_all, instr_t_all = [], [], [], []
    for batch in loader:
        out = model(batch["clip"].to(device))
        step_logits_all.append(out["step"].cpu()); step_t_all.append(batch["step"])
        instr_logits_all.append(out["instrument"].cpu()); instr_t_all.append(batch["instrument"])
    sf1 = macro_f1_multiclass(torch.cat(step_logits_all), torch.cat(step_t_all), num_steps)
    if1 = macro_f1_multilabel(torch.cat(instr_logits_all), torch.cat(instr_t_all))
    return sf1, if1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="runs/phase")
    args = ap.parse_args()

    cfg = {"backbone": "timm:convnext_tiny", "pretrained": True, "temporal": "tcn",
           "temporal_ch": 128, "temporal_layers": 6, "num_steps": 14,
           "num_instruments": 18, "batch_size": 2, "lr": 1e-4, "epochs": 20,
           "clip_len": 64, "stride": 32, "img_size": 224, "freeze_backbone": False}
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    if args.synthetic:
        cfg.update(backbone="timm:convnext_atto", pretrained=False, temporal_ch=64,
                   temporal_layers=3, freeze_backbone=False)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    device = get_device()
    print(device_report(device))

    model_cfg = PhaseModelConfig(
        backbone=cfg["backbone"], pretrained=cfg["pretrained"],
        freeze_backbone=cfg["freeze_backbone"], temporal=cfg["temporal"],
        temporal_ch=cfg["temporal_ch"], temporal_layers=cfg["temporal_layers"],
        num_steps=cfg["num_steps"], num_instruments=cfg["num_instruments"])
    model = SpatioTemporalMultiTask(model_cfg).to(device)

    train_loader, val_loader = build_loaders(args, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])

    os.makedirs(args.out, exist_ok=True)
    best = -1.0
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            out = model(batch["clip"].to(device))
            loss, parts = model.compute_loss(out, batch["step"].to(device),
                                             batch["instrument"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        sf1, if1 = evaluate(model, val_loader, device, cfg["num_steps"])
        print(f"epoch {epoch:02d} | loss {running/len(train_loader):.4f} "
              f"| step macro-F1 {sf1:.3f} | instr macro-F1 {if1:.3f}")
        if sf1 > best:
            best = sf1
            torch.save({"model": model.state_dict(), "cfg": cfg},
                       os.path.join(args.out, "best.pt"))
    print(f"done. best step macro-F1 = {best:.3f}. checkpoint -> {args.out}/best.pt")

if __name__ == "__main__":
    main()
