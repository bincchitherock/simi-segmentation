set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
RUNS="${RUNS:-runs/smoke}"

echo " Device "
"$PY" -c "import src.common.device as d; print(d.device_report(d.get_device()))"

echo
echo "[1/4] render the endoscopic phantom to disk "
"$PY" -m src.data.make_dummy_data --out data

echo
echo "[2/4] phase: 2 epochs through ClipDataset (on-disk frames + videos.json)"
"$PY" -m src.phase.train --config configs/phase_phantom.yaml --epochs 2 --out "$RUNS/phase"

echo
echo "[3/4] segmentation: 20 steps through MaskDataset (on-disk pairs.json)"
"$PY" -m src.segmentation.train_sam2 --config configs/seg_phantom.yaml --steps 20 --out "$RUNS/seg"

echo
echo "[4/4] segmentation inference: reload the adapters, render figures"
"$PY" -m src.segmentation.predict --checkpoint "$RUNS/seg/adapters.pt" --split test --out "$RUNS/results"

cat <<'EOF'
EOF
