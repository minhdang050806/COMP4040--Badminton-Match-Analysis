# Phase 05: SA-CNN Fine-Tuning and Rally Filtering

## Objective

Implement Phase 05 as a clean, reproducible SA-CNN rally-filtering stage:

1. Build a ShuttleSet-derived ImageFolder dataset from all 44 raw videos.
2. Provide fine-tuning commands for a protocol checkpoint and optional final-all44 deployment checkpoint.
3. Run rally interval extraction only after a selected checkpoint exists.

Status: implementation ready; training and full rally inference not executed by request.

## Source Files and Directories Inspected

- `project/__guidance__/systems.md`
- `project/tools/phase05_build_sacnn_dataset.py`
- `project/tools/phase05_train_sacnn.py`
- `project/tools/phase05_finetune_all44.sh`
- `project/tools/phase05_rally_filtering.py`
- `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/sacnn.py`
- `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt`
- `project/outputs/tables/shuttleset_ground_truth_strokes.csv`
- `project/outputs/tables/phase02_rally_summary.csv`

## Old Result Cleanup

Deleted stale generated Phase 05 artifacts to avoid mixing old smoke/fine-tune outputs with new all-44 protocol outputs:

```text
project/outputs/rallies/
project/outputs/sacnn_training_all44/
project/outputs/sacnn_training_gpu1/
project/dataset/sa_cnn_data/
project/dataset/sa_cnn_data_all44/
project/weights/sacnn_shuttleset_finetuned_gpu1.pt
project/weights/sacnn_shuttleset_finetuned_protocol.pt
project/weights/sacnn_shuttleset_finetuned_all44.pt
```

Raw videos, ShuttleSet annotations, Phase 02 tables, and upstream `sacnn.pt` were not deleted.

## Inputs

Raw videos:

```text
project/dataset/ShuttleSet_raw_videos/*.mp4
```

Annotations used for dataset labels:

```text
project/dataset/ShuttleSet/set/match.csv
project/dataset/ShuttleSet/set/<match_name>/set*.csv
```

Validation tables used by rally filtering:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
project/outputs/tables/phase02_rally_summary.csv
```

Initial SA-CNN checkpoint:

```text
external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt
```

## Implementation

Dataset builder:

```text
project/tools/phase05_build_sacnn_dataset.py
```

Training script:

```text
project/tools/phase05_train_sacnn.py
```

All-44 protocol wrapper:

```text
project/tools/phase05_finetune_all44.sh
```

Rally filtering inference wrapper:

```text
project/tools/phase05_rally_filtering.py
```

Implementation details:

- `phase05_build_sacnn_dataset.py` builds an ImageFolder tree with `train/val/test` and `0/1` class folders.
- Positive samples come from annotated rally/hit intervals in `set*.csv`, padded by `--positive-pad-sec`.
- Negative samples come from complement intervals with `--negative-guard-sec` around positives.
- Dataset rebuilds now refuse a non-empty output root unless `--overwrite` is explicit.
- Video-level dataset status is `ok` only when both positive and negative images are produced.
- Dataset extraction supports `--decoder {auto,opencv,ffmpeg}`. The default `auto` uses `ffprobe` to route AV1 videos through system `ffmpeg` instead of OpenCV, avoiding hardware-accelerated AV1 decode warnings such as `Missing Sequence Header` and `Failed to get pixel format`.
- `phase05_train_sacnn.py` fine-tunes upstream `SACNN`, records split overlap, supports early stopping, and writes a training summary.
- `phase05_finetune_all44.sh` now writes protocol and final-all44 checkpoints to separate paths.
- `phase05_rally_filtering.py` now defaults to the protocol checkpoint path:

```text
project/weights/sacnn_shuttleset_finetuned_protocol.pt
```

The rally wrapper keeps the project-local `ShotAngleQueue` path and avoids upstream `VideoResolver`, because that upstream path pulls in blocked `RallyProcessor` dependencies outside Phase 05.

## Commands To Run

Compile and CLI checks already run:

```bash
python3 -m py_compile \
  project/tools/phase05_build_sacnn_dataset.py \
  project/tools/phase05_train_sacnn.py \
  project/tools/phase05_rally_filtering.py

python3 project/tools/phase05_build_sacnn_dataset.py --help
python3 project/tools/phase05_train_sacnn.py --help
python3 project/tools/phase05_rally_filtering.py --help
bash -n project/tools/phase05_finetune_all44.sh
```

Optional small dataset smoke build:

```bash
python3 project/tools/phase05_build_sacnn_dataset.py \
  --video-ids 1 \
  --max-videos 1 \
  --samples-per-class-per-video 8 \
  --decoder auto \
  --output-root project/dataset/sa_cnn_data_smoke \
  --overwrite
```

Optional small training smoke run after the smoke dataset exists:

```bash
python3 project/tools/phase05_train_sacnn.py \
  --data-root project/dataset/sa_cnn_data_smoke \
  --train-splits train \
  --val-splits train \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --output-root project/outputs/sacnn_training_smoke \
  --weight-out project/weights/sacnn_shuttleset_finetuned_smoke.pt
```

Protocol all-44 fine-tuning command:

```bash
GPU_ID=0 \
EPOCHS=20 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
SAMPLES_PER_CLASS_PER_VIDEO=300 \
SEED=20260605 \
EARLY_STOP_EPOCHS=5 \
bash project/tools/phase05_finetune_all44.sh protocol
```

Protocol outputs:

```text
project/dataset/sa_cnn_data_all44/
project/outputs/sacnn_training_protocol/
project/weights/sacnn_shuttleset_finetuned_protocol.pt
```

Run held-out rally filtering on videos `40-44` after protocol training:

```bash
python3 project/tools/phase05_rally_filtering.py \
  --video-id 40 \
  --video-id 41 \
  --video-id 42 \
  --video-id 43 \
  --video-id 44 \
  --max-videos 0 \
  --sacnn-path project/weights/sacnn_shuttleset_finetuned_protocol.pt \
  --training-summary project/outputs/sacnn_training_protocol/training_summary.json \
  --decoder auto \
  --sample-period-sec 1.0 \
  --batch-size 128 \
  --visual-samples 2
```

Optional final all-44 deployment checkpoint after the protocol report is complete:

```bash
GPU_ID=0 \
EPOCHS=20 \
BATCH_SIZE=64 \
NUM_WORKERS=4 \
SAMPLES_PER_CLASS_PER_VIDEO=300 \
SEED=20260605 \
EARLY_STOP_EPOCHS=5 \
bash project/tools/phase05_finetune_all44.sh final-all44
```

Final-all44 outputs:

```text
project/dataset/sa_cnn_data_all44/
project/outputs/sacnn_training_all44_final/
project/weights/sacnn_shuttleset_finetuned_all44.pt
```

The final-all44 command intentionally trains on `train,val,test`; do not use its validation metrics as an unbiased estimate.

## Expected Outputs

Dataset outputs:

```text
project/dataset/sa_cnn_data_all44/manifest.csv
project/dataset/sa_cnn_data_all44/video_summary.csv
project/dataset/sa_cnn_data_all44/summary.json
project/dataset/sa_cnn_data_all44/train/{0,1}/*.jpg
project/dataset/sa_cnn_data_all44/val/{0,1}/*.jpg
project/dataset/sa_cnn_data_all44/test/{0,1}/*.jpg
```

Training outputs:

```text
project/outputs/sacnn_training_protocol/best_sacnn.pt
project/outputs/sacnn_training_protocol/last_sacnn.pt
project/outputs/sacnn_training_protocol/training_summary.json
project/weights/sacnn_shuttleset_finetuned_protocol.pt
```

Rally filtering outputs:

```text
project/outputs/rallies/<video_id>/sampled_shot_angle_states.csv
project/outputs/rallies/<video_id>/rally_intervals.csv
project/outputs/rallies/phase05_video_validation.csv
project/outputs/rallies/visual_review_manifest.csv
project/outputs/rallies/clip_manifest.csv
project/outputs/rallies/phase05_rally_filtering_summary.json
```

Optional clips when `--export-clips` is set:

```text
project/outputs/rallies/<video_id>/clips/rally_<rally_id>.mp4
```

## Validation Performed

Validation completed without building the full dataset or running training:

- `python3 -m py_compile project/tools/phase05_build_sacnn_dataset.py project/tools/phase05_train_sacnn.py project/tools/phase05_rally_filtering.py`
- `python3 project/tools/phase05_build_sacnn_dataset.py --help`
- `python3 project/tools/phase05_train_sacnn.py --help`
- `python3 project/tools/phase05_rally_filtering.py --help`
- `bash -n project/tools/phase05_finetune_all44.sh`
- Verified `project/dataset/ShuttleSet_raw_videos/01 - Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals.mp4` reports codec `av1` and `--decoder auto` selects `ffmpeg`.

Confirmed stale generated Phase 05 artifacts were removed.

Validation still required after you run the protocol command:

- `project/dataset/sa_cnn_data_all44/video_summary.csv` has 44 rows.
- Every row has `status=ok`, positive images `>0`, and negative images `>0`.
- `project/outputs/sacnn_training_protocol/training_summary.json` records disjoint `train` and `val` splits.
- Held-out videos `40-44` are evaluated through `phase05_rally_filtering.py`.
- `phase05_rally_filtering_summary.json` reports annotation hit coverage, rally overlap rate, decoder counts, and human visual review status.

## Assumptions and Architectural Notes

- Label `1` means rally/shot-angle-positive frame; label `0` means sampled non-rally/background frame.
- Phase 05 is independent of Phase 04. It consumes raw videos and annotation CSVs, not BST predictions.
- The preprocessing contract is resize `(216, 384)`, center crop `(216, 216)`, `ToTensor()`, and ImageNet normalization.
- The default decoder is `auto`; AV1 videos should route through system `ffmpeg`.
- Human visual review remains separate from machine coverage metrics.

## Troubleshooting

If dataset building prints repeated OpenCV messages like:

```text
Missing Sequence Header
Your platform doesn't suppport hardware accelerated AV1 decoding
Failed to get pixel format
```

then the build is using the old OpenCV-only path or an explicit `--decoder opencv`. Stop that run, delete the partial output root or rerun with `--overwrite`, and use the current builder with:

```bash
python3 project/tools/phase05_build_sacnn_dataset.py \
  --video-ids 1-44 \
  --max-videos 0 \
  --decoder auto \
  --output-root project/dataset/sa_cnn_data_all44 \
  --overwrite
```

Use `--decoder ffmpeg` only if you want to force ffmpeg for every video. The `auto` path is preferred because it keeps OpenCV for normally decodable videos and uses ffmpeg for AV1.

## Blockers and Handoff

No code blocker remains for starting Phase 05 protocol fine-tuning.

Operational caveats:

- Full dataset construction decodes frames from all 44 videos and may take time.
- If OpenCV or ffmpeg fails on a video, do not train final checkpoints until the failed row is fixed or explicitly excluded with a documented reason.
- Do not hand Phase 05 to Phase 06 for full production processing until held-out coverage and visual review are acceptable.
