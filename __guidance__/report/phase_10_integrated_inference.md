# Phase 10: BST-Compatible Tracked Integrated Inference

## Status

Implementation and full CPU input preparation are complete for:

```text
project/outputs/features_yolo26x_bst_tracked_phase02_41_44
```

The selected BST checkpoint and prepared arrays pass validation. BST inference
has not been run; use the documented command below.

Phase 10 does not train or fine-tune a model. It integrates the completed Phase
09 features with the selected Phase 04 BST checkpoint.

## Active Inputs

Tracked Phase 09 features:

```text
project/outputs/features_yolo26x_bst_tracked_phase02_41_44/
  phase09_feature_manifest.csv
  phase09_feature_validation.csv
  phase09_feature_summary.json
  <video_id>/<clip_id>_{joints,pos,shuttle}.npy
  <video_id>/<clip_id>_validity.npz
```

Ground-truth stroke metadata:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
```

Reference BST labels and split identities:

```text
project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits/
```

Selected checkpoint:

```text
project/weights/bst_shuttleset_finetuned_protocol.pt
```

This checkpoint predicts 25 side-aware merged classes:

```text
none + Top_<12 stroke types> + Bottom_<12 stroke types>
```

## Implementation

```text
project/tools/phase10_prepare_bst_inputs.py
project/tools/phase10_run_bst_inference.py
```

Phase 10 preparation now:

- consumes the Phase 09 `bst-compatible` event/window metadata;
- preserves `Top,Bottom` player ordering and exact event offsets;
- joins ground truth and reference labels strictly by exact `clip_id`;
- reads separate Top/Bottom pose and position validity rates;
- preserves the Phase 09 validity-sidecar path;
- repairs simultaneous same-frame annotation aliases caused by Phase 09's
  frame-keyed clip-ID lookup, while recording the shared feature source;
- computes quality groups from pose-valid masks instead of legacy tensor-zero
  heuristics;
- reuses the Phase 03 sequence-100 and `JnB_bone` construction.

Phase 10 inference now reports:

- full 25-class accuracy and macro metrics;
- top-3 accuracy;
- player-side accuracy;
- stroke-type accuracy ignoring side;
- per-class precision, recall, and F1;
- quality-stratified results, including Top-player pose-valid groups;
- per-video, reference-validation, and reference-test results.
- automatic primary accuracy and macro-F1 deltas against
  `baseline_old_phase09`.

## Prepared Bundle

Prepared output:

```text
project/outputs/integration/bst_tracked_phase09_41_44/inputs/
```

| Metric | Value |
|---|---:|
| Prepared rows | 3,192 |
| Exact labeled rows | 3,192 |
| Candidate-only rows | 0 |
| Reference validation rows, video 41 | 1,153 |
| Reference test rows, videos 42-44 | 2,039 |
| Clean primary test rows after alias exclusion | 2,027 |
| Prepared size | 186 MB |
| Window strategy | `bst-compatible` for all rows |
| Player order | `Top,Bottom` for all rows |
| Tensor shape validation | Passed |
| Finite-value validation | Passed |
| Checkpoint/model validation | Passed |

Strict source-contract checks:

| Check | Result |
|---|---:|
| Duplicate logical clip IDs after repair | 0 |
| Missing validation rows | 0 |
| Missing validity sidecars | 0 |
| Event frames outside windows | 0 |
| Manifest/validation frame-count mismatches | 0 |

Phase 09 contained six simultaneous-event groups where multiple ShuttleSet
annotations share the exact same frame. Its frame-keyed lookup collapsed these
into duplicate clip IDs and one shared saved feature sequence. Phase 10 repairs
the logical label rows before collation:

```text
simultaneous event groups repaired: 6
logical rows using shared feature aliases: 14
```

`feature_alias_source_clip_id` records these rows explicitly in Phase 10
metadata and predictions. Because their original distinct temporal windows were
overwritten, all 14 alias rows are excluded from primary and per-class accuracy
metrics and retained only in:

```text
simultaneous_event_alias_diagnostic
reference_test_all_including_aliases
reference_val_all_including_aliases
```

Prepared tensors:

```text
JnB_bone.npy   (3192, 100, 2, 36, 2) float32
pos.npy        (3192, 100, 2, 2) float32
shuttle.npy    (3192, 100, 2) float32
videos_len.npy (3192,) int64
labels.npy     (3192,) int64
```

## Phase 09 Handoff Quality

The corrected temporal construction closely matches reference BST clips:

| Metric | Generated | Reference |
|---|---:|---:|
| Mean clip length | 57.37 | 58.06 |
| Length MAE | 0.91 frames | - |
| Current-hit offset MAE | 0.42 frames | - |
| Player side consistency | 100% | 99.93% |
| Position outside `[-0.2, 1.2]` | 0 | 0 |

Phase 09 is structurally complete but still reports
`completed_needs_quality_review` because far-side Top-player detection remains
weak:

| Quality mean | Value |
|---|---:|
| Top pose valid rate | 54.09% |
| Bottom pose valid rate | 98.90% |
| Top position valid rate | 54.20% |
| Bottom position valid rate | 99.35% |
| Both players pose-missing rate | 0.88% |
| Both players position-missing rate | 0.38% |
| Pose tensor zero rate | 32.87% |
| Position tensor zero rate | 23.23% |

Quality groups:

| Group | Rows |
|---|---:|
| Pose dropout `<25%` | 1,687 |
| Pose dropout `25-50%` | 1,291 |
| Pose dropout `>=50%` | 214 |
| All pose missing | 0 |

These groups must be reported with the final classifier metrics.

## Evaluation Policy

All 3,192 clips have exact reference labels. The primary unbiased metric is:

```text
reference_test_primary
```

It contains the 2,027 clean rows from videos `42-44`. Video `41` is the
reference validation split and must remain diagnostic because it was used when
selecting the checkpoint.

Required evaluation groups include:

```text
reference_test_primary
reference_val_diagnostic
reference_test_pose_dropout_lt_50
reference_test_pose_dropout_lt_25
reference_test_top_pose_valid_ge_75
reference_test_top_pose_valid_lt_50
reference_test_bottom_pose_valid_ge_75
reference_test_both_pose_missing_lt_10
exact_label_video_<id>
```

Interpret decomposed metrics as:

- `accuracy`: both player side and stroke type are correct;
- `player_side_accuracy`: Top/Bottom class prefix is correct;
- `stroke_type_accuracy`: merged stroke type is correct, ignoring side;
- `top3_accuracy`: true side-aware class occurs in the top three predictions.

## Run Inference

Run this command:

```bash
python3 project/tools/phase10_run_bst_inference.py \
  --input-root project/outputs/integration/bst_tracked_phase09_41_44/inputs \
  --checkpoint project/weights/bst_shuttleset_finetuned_protocol.pt \
  --output-root project/outputs/integration/bst_tracked_phase09_41_44 \
  --device cuda \
  --gpu-id 0 \
  --batch-size 256 \
  --num-workers 0 \
  --num-threads 8
```

Expected outputs:

```text
project/outputs/integration/bst_tracked_phase09_41_44/
  phase10_predictions.csv
  phase10_structured_strokes.csv
  phase10_evaluation_by_group.csv
  phase10_evaluation_by_class.csv
  phase10_logits.npy
  phase10_probabilities.npy
  phase10_summary.json
```

The old Phase 09 primary test accuracy was `9.23%`. Compare the new
`reference_test_primary` result against that baseline, but do not assume the new
features improve accuracy until inference confirms it. The checkpoint was
trained on reference MMPose/TrackNet features, so a distribution mismatch can
remain even after temporal alignment and tracking improve.

`phase10_summary.json` performs this baseline comparison automatically using:

```text
project/outputs/integration/baseline_old_phase09/phase10_summary.json
```

## Optional Commands

Rebuild the prepared inputs:

```bash
python3 project/tools/phase10_prepare_bst_inputs.py \
  --feature-root project/outputs/features_yolo26x_bst_tracked_phase02_41_44 \
  --phase02-table project/outputs/tables/shuttleset_ground_truth_strokes.csv \
  --phase07-matches project/outputs/hit_frames/phase07_matches.csv \
  --reference-collated-root project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits \
  --output-root project/outputs/integration/bst_tracked_phase09_41_44/inputs \
  --seq-len 100 \
  --progress-every 500
```

Validate prepared inputs and checkpoint without inference:

```bash
python3 project/tools/phase10_run_bst_inference.py \
  --input-root project/outputs/integration/bst_tracked_phase09_41_44/inputs \
  --checkpoint project/weights/bst_shuttleset_finetuned_protocol.pt \
  --device cpu \
  --validate-only
```

## Validation Performed

- `phase10_prepare_bst_inputs.py` and `phase10_run_bst_inference.py` compile.
- Phase 10 integration unit tests pass.
- Five-clip tracked preparation smoke passed.
- Full 3,192-row CPU preparation passed.
- All prepared tensor shapes and finite-value checks passed.
- All 3,192 rows matched exact Phase 02 and reference BST labels.
- Protocol checkpoint loaded with no missing or unexpected model keys.
- BST inference was intentionally not run.

## Phase 11 Handoff

After inference, Phase 11 should consume:

```text
project/outputs/integration/bst_tracked_phase09_41_44/phase10_structured_strokes.csv
project/outputs/integration/bst_tracked_phase09_41_44/phase10_probabilities.npy
project/outputs/integration/bst_tracked_phase09_41_44/inputs/pos.npy
```

Compare the new tactical-mining output against
`project/outputs/mining/baseline_old_phase09/` to measure whether improved
temporal alignment and tracked features reduce the old prediction collapse.
