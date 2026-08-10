set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
SEED="${SEED:-0}"
OUT="${OUT:-results}"
RUNS="${RUNS:-runs}"


SEG_EPOCHS="${SEG_EPOCHS:-80}"

mkdir -p "$OUT"

echo "[1/5] data"
"$PY" -m src.data.make_dummy_data --out data --seed "$SEED" | tee "$OUT/data.log"

echo
echo "[2/5] phase recognition: train on the train split, select on val"
"$PY" -m src.phase.train --config configs/phase_phantom.yaml --seed "$SEED" \
      --out "$RUNS/phase" | tee "$OUT/phase_train.log"

echo
echo "[3/5] phase recognition: score the held out test videos, draw the timeline"
"$PY" -m src.phase.predict --checkpoint "$RUNS/phase/best.pt" --split test \
      --out "$OUT" | tee "$OUT/phase_predict.log"

echo
echo "[4/5] instrument segmentation (TinyUNet; SAM 2 needs the sam2 package)"
"$PY" -m src.segmentation.train_sam2 --config configs/seg_phantom.yaml --seed "$SEED" \
      --epochs "$SEG_EPOCHS" --out "$RUNS/seg" | tee "$OUT/seg_train.log"

echo
echo "[5/5] segmentation inference: val figures + baseline + curve, then test"
"$PY" -m src.segmentation.predict --checkpoint "$RUNS/seg/adapters.pt" --split val \
      --baseline --history "$RUNS/seg/history.json" --max-figures 8 \
      --prefix segmentation_ --out "$OUT" | tee "$OUT/seg_predict.log"
"$PY" -m src.segmentation.predict --checkpoint "$RUNS/seg/adapters.pt" --split test \
      --baseline --max-figures 8 --prefix segmentation_test_ --out "$OUT" \
      | tee "$OUT/seg_predict_test.log"

cat <<EOF
EOF
