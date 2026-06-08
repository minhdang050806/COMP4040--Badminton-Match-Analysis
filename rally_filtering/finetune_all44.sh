#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-protocol}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLES_PER_CLASS_PER_VIDEO="${SAMPLES_PER_CLASS_PER_VIDEO:-300}"
SEED="${SEED:-20260605}"
EARLY_STOP_EPOCHS="${EARLY_STOP_EPOCHS:-5}"

DATA_ROOT="${DATA_ROOT:-project/dataset/sa_cnn_data_all44}"

case "${MODE}" in
  protocol)
    TRAIN_SPLITS="train"
    VAL_SPLITS="val"
    RUN_ROOT="${RUN_ROOT:-project/outputs/sacnn_training_protocol}"
    WEIGHT_OUT="${WEIGHT_OUT:-project/weights/sacnn_shuttleset_finetuned_protocol.pt}"
    ;;
  final-all44)
    TRAIN_SPLITS="train,val,test"
    VAL_SPLITS="val"
    RUN_ROOT="${RUN_ROOT:-project/outputs/sacnn_training_all44_final}"
    WEIGHT_OUT="${WEIGHT_OUT:-project/weights/sacnn_shuttleset_finetuned_all44.pt}"
    ;;
  *)
    echo "Usage: $0 [protocol|final-all44]" >&2
    echo "  protocol    : train on videos outside val/test, validate on 35-39, hold out 40-44." >&2
    echo "  final-all44 : train on train,val,test folders for a deployment checkpoint." >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}" "$(dirname "${WEIGHT_OUT}")"

echo "[phase05] Building SA-CNN ImageFolder dataset from all selected ShuttleSet videos"
python rally_filtering/build_dataset.py \
  --video-ids 1-44 \
  --max-videos 0 \
  --val-ids 35-39 \
  --test-ids 40-44 \
  --samples-per-class-per-video "${SAMPLES_PER_CLASS_PER_VIDEO}" \
  --positive-pad-sec 2.0 \
  --negative-guard-sec 4.0 \
  --min-sample-gap-sec 1.0 \
  --decoder auto \
  --output-root "${DATA_ROOT}" \
  --overwrite \
  2>&1 | tee "${RUN_ROOT}/dataset_build.log"

python -c '
import csv
from pathlib import Path
root = Path("'"${DATA_ROOT}"'")
rows = list(csv.DictReader((root / "video_summary.csv").open()))
bad = [
    row for row in rows
    if row.get("status") != "ok"
    or int(row.get("positive_images") or 0) <= 0
    or int(row.get("negative_images") or 0) <= 0
]
if len(rows) != 44 or bad:
    print(f"[phase05] Refusing to train: expected 44 complete videos, found {len(rows)} rows and {len(bad)} incomplete rows.")
    for row in bad:
        print(row)
    raise SystemExit(1)
print("[phase05] Dataset completeness check passed for 44 videos.")
'

echo "[phase05] Fine-tuning SA-CNN mode=${MODE}"
python rally_filtering/train.py \
  --data-root "${DATA_ROOT}" \
  --train-splits "${TRAIN_SPLITS}" \
  --val-splits "${VAL_SPLITS}" \
  --output-root "${RUN_ROOT}" \
  --weight-out "${WEIGHT_OUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpu-id "${GPU_ID}" \
  --seed "${SEED}" \
  --early-stop-epochs "${EARLY_STOP_EPOCHS}" \
  2>&1 | tee "${RUN_ROOT}/training.log"

echo "[phase05] Finished"
echo "[phase05] Dataset: ${DATA_ROOT}"
echo "[phase05] Run outputs: ${RUN_ROOT}"
echo "[phase05] Weight: ${WEIGHT_OUT}"
