# Phase 04: BST Fine-Tuning and Stroke Classification

## Objective

Provide a project-owned Phase 04 path for BST fine-tuning on the Phase 03 ShuttleSet collated features, while preserving the existing inference wrapper for evaluation after a checkpoint has been selected.

Status: implementation ready; training not executed by request.

## Source Files and Directories Inspected

- `project/__guidance__/systems.md`
- `project/tools/phase03_collate_bst_features.py`
- `project/tools/phase04_run_bst_inference.py`
- `project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits/phase03_collation_summary.json`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/main_on_shuttleset/bst_main.py`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/model/bst.py`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/preparing_data/shuttleset_dataset.py`

## Old Result Cleanup

Deleted stale Phase 04 inference outputs to avoid confusing old pretrained-checkpoint results with future fine-tuned results:

```text
project/outputs/predictions/bst_test_predictions.csv
project/outputs/predictions/bst_test_logits.npy
project/outputs/predictions/bst_test_probabilities.npy
project/outputs/predictions/bst_test_confusion_matrix.csv
project/outputs/predictions/bst_test_metrics_by_class.csv
project/outputs/predictions/phase04_bst_test_summary.json
project/outputs/tables/shuttleset_strokes_with_bst_predictions.csv
```

Preserved Phase 03 collation outputs under:

```text
project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits/
```

Those arrays are inputs to Phase 04, not old Phase 04 results.

## Data Inputs

Primary collated feature root:

```text
project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits/
```

Required split arrays:

```text
<split>/JnB_bone.npy
<split>/pos.npy
<split>/shuttle.npy
<split>/videos_len.npy
<split>/labels.npy
```

Protocol split policy:

- Train: Phase 03 `train` split, 25,741 clips.
- Validation: Phase 03 `val` split, 4,241 clips.
- Test: Phase 03 `test` split, 3,499 clips.

Initial checkpoint:

```text
project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt
```

The training script starts from this checkpoint by default. Use `--from-scratch` only for an explicit ablation.

## Implementation

New training entry point:

```text
project/tools/phase04_train_bst.py
```

Existing inference/evaluation entry point:

```text
project/tools/phase04_run_bst_inference.py
```

The training script:

- Reuses the Phase 04 `BST_CG_AP` construction and local `positional_encodings` shim from the inference wrapper.
- Loads Phase 03 memory-mapped collated arrays.
- Keeps the required model contract: `JnB_bone`, `seq_len=100`, 25 merged classes.
- Loads the ShuttleSet checkpoint with strict key validation unless `--from-scratch` is set.
- Uses deterministic seeds, class-weighted cross entropy, label smoothing, AdamW, cosine LR scheduling, optional gradient clipping, and early stopping on validation macro-F1.
- Applies small random translation only to joint coordinates during training; bone vectors are not shifted.
- Writes best/last checkpoints, training history, validation metrics, confusion matrix, and JSON summary.

Output paths:

```text
project/outputs/bst_training_all44/training_history.csv
project/outputs/bst_training_all44/val_metrics_by_class.csv
project/outputs/bst_training_all44/val_confusion_matrix.csv
project/outputs/bst_training_all44/training_summary.json
project/outputs/bst_training_all44/best_bst.pt
project/outputs/bst_training_all44/last_bst.pt
project/weights/bst_shuttleset_finetuned_protocol.pt
```

## Commands To Run

Compile check already run:

```bash
python3 -m py_compile project/tools/phase04_train_bst.py project/tools/phase04_run_bst_inference.py
```

CLI check already run:

```bash
python3 project/tools/phase04_train_bst.py --help
```

Optional small smoke run for your machine:

```bash
python3 project/tools/phase04_train_bst.py \
  --device auto \
  --gpu-id 0 \
  --epochs 1 \
  --batch-size 16 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --output-root project/outputs/bst_training_smoke \
  --weight-out project/weights/bst_shuttleset_finetuned_smoke.pt
```

Protocol fine-tuning run:

```bash
python3 project/tools/phase04_train_bst.py \
  --device auto \
  --gpu-id 0 \
  --train-splits train \
  --val-splits val \
  --epochs 40 \
  --batch-size 128 \
  --num-workers 0 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --random-translation 0.02 \
  --early-stop-epochs 8 \
  --output-root project/outputs/bst_training_all44 \
  --weight-out project/weights/bst_shuttleset_finetuned_protocol.pt
```

After protocol training, evaluate on the untouched test split:

```bash
python3 project/tools/phase04_run_bst_inference.py \
  --checkpoint project/weights/bst_shuttleset_finetuned_protocol.pt \
  --split test \
  --batch-size 128 \
  --num-workers 0 \
  --num-threads 8 \
  --device auto
```

Optional final all-44 deployment checkpoint after the protocol report is complete:

```bash
python3 project/tools/phase04_train_bst.py \
  --device auto \
  --gpu-id 0 \
  --train-splits train,val,test \
  --val-splits val \
  --epochs 40 \
  --batch-size 128 \
  --num-workers 0 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05 \
  --random-translation 0.02 \
  --early-stop-epochs 8 \
  --output-root project/outputs/bst_training_all44_final \
  --weight-out project/weights/bst_shuttleset_finetuned_all44.pt
```

The final all-44 command intentionally overlaps train and validation data. Treat it as a deployment checkpoint, not an unbiased validation estimate.

## Validation Performed

Machine validation completed without starting training:

- `python3 -m py_compile project/tools/phase04_train_bst.py project/tools/phase04_run_bst_inference.py`
- `python3 project/tools/phase04_train_bst.py --help`
- Verified old Phase 04 prediction artifacts no longer exist under `project/outputs/predictions/` or `project/outputs/tables/`.

Training validation still to be performed by the user after running the protocol command:

- Check `training_summary.json` for split counts, seed, device, initialization checkpoint, best epoch, and best validation macro-F1.
- Check `training_history.csv` for stable loss and validation behavior.
- Check `val_metrics_by_class.csv` and `val_confusion_matrix.csv` for per-class failures.
- Run test inference with `project/tools/phase04_run_bst_inference.py --checkpoint project/weights/bst_shuttleset_finetuned_protocol.pt --split test`.
- Confirm the new `phase04_bst_test_summary.json` reports row count 3,499 and valid label range `[0, 24]`.

## Assumptions and Architectural Notes

- Phase 04 uses stored ShuttleSet feature tensors rather than rerunning pose, shuttle, or player-position extraction.
- Fine-tuning does not change the BST architecture. It only updates model weights for the existing `BST_CG_AP` model.
- The Phase 03 collation contract is fixed at `seq_len=100`; the training script rejects other sequence lengths for this phase.
- Random translation augmentation is applied only to the first 17 joint coordinates in `JnB_bone`; the 19 bone-vector channels are preserved.
- The active project default remains 2D pose features with merged 25-class stroke labels.

## Blockers and Handoff

No code blocker remains for starting Phase 04 BST protocol fine-tuning.

Operational caveats:

- The previous validated environment had `torch.cuda.is_available() == False`; check GPU availability before expecting fast training.
- If `--device auto` falls back to CPU, reduce batch size or run on a CUDA-ready environment.
- Do not run Phase 10 or tactical mining with Phase 04 predictions until the protocol checkpoint has been trained and evaluated on the held-out test split.
