#!/usr/bin/env bash
set -euo pipefail

echo "== device check =="
python -c "from src.common.device import get_device, device_report; print(device_report(get_device()))"

echo
echo "== [1/3] on-disk dummy data (exercises real Dataset classes) =="
python -m src.data.make_dummy_data --out data

echo
echo "== [2/3] phase model: synthetic train + eval =="
python -m src.phase.train --synthetic --epochs 2

echo
echo "== [3/3] SAM2 LoRA loop: synthetic (DummySegmenter, no checkpoint) =="
python -m src.segmentation.train_sam2 --synthetic --steps 20

echo
echo "all green — pipelines wired correctly. Swap synthetic for real data + a CUDA GPU to train for real."
