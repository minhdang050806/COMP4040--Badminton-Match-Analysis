# System Description

This project targets an end-to-end badminton analytics system:

TRACK A: Known ShuttleSet / benchmark path
Phase 00 setup
  -> Phase 01 dataset inventory
  -> Phase 02 ground-truth tactical table
  -> Phase 03 collate provided ShuttleSet features
  -> Phase 04 BST fine-tuning + stroke classification
  -> Phase 11 tactical mining on ground truth + predictions

TRACK B: Raw unseen-video path
raw MP4
  -> Phase 05 SA-CNN fine-tuning + rally filtering
  -> Phase 06 TrackNetV3 fine-tuning or verified inference + denoising
  -> Phase 07 hit-frame parameter tuning + stroke windows
  -> Phase 08 court calibration / homography
  -> Phase 09 pose, player position, feature normalization
  -> Phase 10 integrated inference with selected fine-tuned/tuned components
  -> Phase 11 tactical mining

```text
raw unseen broadcast video
  -> rally / shot-angle filtering with fine-tuned SA-CNN
  -> shuttle tracking with fine-tuned or verified TrackNetV3
  -> hit-frame detection with tuned event parameters
  -> court calibration and homography
  -> player pose and court-position extraction
  -> BST stroke classification with selected checkpoint
  -> structured rally table
  -> data-mining / tactical analysis
```

The current checkout is not yet a single integrated application. It is a staged research workspace with empty glue-code folders under `project/`, a complete ShuttleSet data copy, ShuttleSet BST weights, and upstream perception repositories under top-level `external_repos/`.

## Local Project Boundary

```text
project/
  1_video_prepreocessing/     intended video/rally preprocessing stage; currently empty
  2_rally_segmentation/       intended shuttle/hit-frame/rally segmentation glue; currently empty
  3_court_detection/          intended court/homography glue; currently empty
  4_stroke_classification/    intended BST inference/training glue; currently empty
  dataset/                    ShuttleSet raw videos, annotations, pre-extracted npy features
  weights/on_ShuttleSet/      ShuttleSet-trained BST and baseline checkpoints
  __guidance__/               source, system, data, and report documentation
    report/                   required phase reports

external_repos/
  Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/
  BST-Badminton-Stroke-type-Transformer/
  TrackNetV3/
  monotrack/
  mmpose/
  A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/
  badminton-db/
```

The `project/1_*` to `project/4_*` folders should be treated as planned integration points. At the time of this audit they contain no files, so the implemented behavior still lives in external repositories and the dataset.

## Phase Roadmap and Reporting Requirements

This section defines the execution phases for the project. The lower "Stage" sections describe the model components; the phases below describe how to set up, process data, run or validate each component, and move toward the final data-mining goal.

Every phase must produce a Markdown report under:

```text
project/__guidance__/report/
```

Use this naming scheme:

```text
project/__guidance__/report/phase_00_setup.md
project/__guidance__/report/phase_01_dataset_inventory.md
project/__guidance__/report/phase_02_ground_truth_tables.md
project/__guidance__/report/phase_03_bst_collation.md
project/__guidance__/report/phase_04_bst_inference.md
project/__guidance__/report/phase_05_rally_filtering.md
project/__guidance__/report/phase_06_shuttle_tracking.md
project/__guidance__/report/phase_07_hit_frame_detection.md
project/__guidance__/report/phase_08_court_homography.md
project/__guidance__/report/phase_09_pose_position.md
project/__guidance__/report/phase_10_integrated_inference.md
project/__guidance__/report/phase_11_tactical_mining.md
```

Each phase report must include:

- Objective.
- Source files and directories inspected.
- Data inputs and exact split/range used.
- Commands run, with working directory and environment.
- Generated outputs and their paths.
- Validation checks, including counts, schemas, tensor shapes, metrics, or qualitative QA.
- Assumptions and any `Unclear from current codebase: ...` items.
- Blockers and the next phase handoff.

Do not mark a phase complete until its report exists and points to its output artifacts. Machine validation and human signoff should be separate when the phase includes visual QA.

### All-44 Fine-Tuning Policy

Use the 44 local ShuttleSet raw videos as the supervised development corpus for every trainable or tunable component in Phases 04-07. The default protocol must still preserve a held-out evaluation split before producing a deployment checkpoint:

- Protocol run: build data from videos `1-44`, train on videos `1-34`, validate/select checkpoints or parameters on videos `35-39`, and keep videos `40-44` as the final test split.
- Final all-44 run: after the protocol report records validation and test metrics, train the deployment checkpoint on all available split folders (`train,val,test`) and label it as a deployment checkpoint, not an unbiased evaluation artifact.
- Every fine-tuning report must record the exact video ids, split assignment, seed, source labels, preprocessing, initial checkpoint, hyperparameters, metric table, best checkpoint path, and whether the checkpoint is protocol or final-all44.
- Do not mix test videos into model selection. If a phase uses all 44 videos to choose thresholds or weights, it must also say that no held-out ShuttleSet estimate remains for that component.
- Store project fine-tuned weights under `project/weights/` and run histories under `project/outputs/<component>_training.../`; do not overwrite upstream checkpoints in `external_repos/`.
- For components without a trainable local model, treat "fine-tuning" as parameter calibration against ShuttleSet annotations and record the selected parameters in a JSON/YAML artifact.

For every phase later than Phase 05, keep the same ShuttleSet protocol split whenever a phase trains, tunes, validates, calibrates against ShuttleSet labels, or reports ShuttleSet benchmark metrics:

- Train/calibration development videos: `1-34`.
- Validation/model-selection videos: `35-39`.
- Final untouched protocol-test videos: `40-44`.
- Phases 06 and 07 must use videos `1-34` for any training or parameter search, videos `35-39` for checkpoint/threshold selection, and videos `40-44` only for final reported test metrics.
- Phases 08 and 09 are not trainable in the default path, but any ShuttleSet calibration quality checks, normalization statistics, thresholds, or acceptance rules must be estimated from videos `1-34`, selected on videos `35-39`, and reported finally on videos `40-44`.
- Phase 10 must use the selected Phase 05-09 protocol artifacts when benchmarking on ShuttleSet and must report train/val/test coverage separately using the same video-id split. For a true unseen MP4 outside ShuttleSet, record it as deployment inference rather than part of the ShuttleSet protocol split.
- Phase 11 must keep ground-truth-only mining and model-prediction mining separate, and any prediction-based ShuttleSet results must report metrics/analysis by `1-34`, `35-39`, and `40-44` groups before any all-44 aggregate.

The four fine-tuned/tuned components on the all-44 path are:

- Phase 04: fine-tune BST stroke classification on collated ShuttleSet feature triples from all 44 videos.
- Phase 05: fine-tune SA-CNN rally/shot-angle filtering from positive and negative frames sampled from all 44 raw videos.
- Phase 06: fine-tune TrackNetV3 shuttle tracking only after a frame-level shuttle-coordinate training set is built or verified from local labels/features; otherwise document `Unclear from current codebase: frame-level TrackNet supervision for all 44 raw videos`.
- Phase 07: tune hit-frame event detection parameters against ShuttleSet `frame_num` annotations on all 44 videos; this phase is currently threshold calibration, not neural-network training.

The old runnable system was mostly inference-first: load an existing checkpoint, run a wrapper, and validate the output. That is acceptable for a baseline, but the all-44 development path needs the following extra work:

| Phase | Old behavior in this checkout | Needs fine-tune/tuning? | Current evidence | Required action |
| --- | --- | --- | --- | --- |
| Phase 04 BST | `project/tools/phase04_run_bst_inference.py` runs test-split inference from `project/weights/on_ShuttleSet/...pt`. | Yes, model fine-tuning. | Upstream `bst_main.py` has training logic, but project currently has only an inference wrapper. | Add/use a project-owned BST fine-tuning wrapper for the Phase 03 collated train/val/test splits, then run protocol and optional final-all44 checkpoints. |
| Phase 05 SA-CNN | `project/tools/phase05_rally_filtering.py` runs a pretrained `sacnn.pt` and threshold/smoothing logic. | Yes, model fine-tuning. | `phase05_build_sacnn_dataset.py`, `phase05_train_sacnn.py`, and `phase05_finetune_all44.sh` already exist. | Rebuild/verify all 44 video summaries, train protocol SA-CNN, evaluate videos `40-44`, then optionally train final-all44. |
| Phase 06 TrackNetV3 | `project/tools/phase06_shuttle_tracking.py` runs the pretrained TrackNetV3 checkpoint for inference and denoising. | Yes if frame-level shuttle labels are available; otherwise verified inference only. | Upstream `TrackNetV3/train.py` exists, but project has no all-44 TrackNet dataset builder yet. | Build/verify `Frame,Visibility,X,Y` supervision before training; if labels cannot be proven, keep inference and document the blocker. |
| Phase 07 hit frames | `project/tools/phase07_hit_frame_detection.py` applies deterministic trajectory rules. | Yes, parameter tuning, not neural fine-tuning. | The wrapper exposes thresholds such as `--prominence-px`, `--min-angle-degrees`, and `--tolerance-frames`. | Add a parameter-search wrapper over train/val videos, score against Phase 02 `frame_num_int`, and reserve videos `40-44` for final metrics. |
| Phase 08 court | Manual/video-scoped homography calibration. | No model fine-tuning. | No trainable court model is required in the current default path. | Keep human calibration and visual QA; monotrack stays optional future automation. |
| Phase 09 pose/position | MMPose inference plus feature normalization. | No fine-tuning unless pose labels are added. | No local player-pose labels or project trainer are present. | Validate pose quality and normalization against stored ShuttleSet feature distributions. |
| Phase 10 integration | Runs the selected components end to end. | No separate fine-tuning. | It consumes Phase 04-09 artifacts. | Use the selected fine-tuned/tuned checkpoints and report coverage/error propagation. |

### Phase 00: Setup and Environment Baseline

Goal: make the project runnable and reproducible before touching data or models.

Inputs:

- `project/__guidance__/source.md`
- `project/__guidance__/data.md`
- `project/__guidance__/systems.md`
- top-level `external_repos/`
- `project/weights/on_ShuttleSet/`
- `project/dataset/`

Setup tasks:

- Record OS, Python, CUDA, PyTorch, OpenCV, ffmpeg, gcc, cmake, and available GPU state.
- Decide whether the active environment is one shared environment or separate environments for BST, TrackNetV3, MMPose, and monotrack.
- Verify read access to `project/dataset/ShuttleSet`, `project/dataset/ShuttleSet_raw_videos`, and `project/weights/on_ShuttleSet`.
- Verify expected external repos exist:
  - `external_repos/BST-Badminton-Stroke-type-Transformer`
  - `external_repos/TrackNetV3`
  - `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis`
  - `external_repos/monotrack`
  - `external_repos/mmpose`
- Verify expected local checkpoints:
  - `external_repos/TrackNetV3/exp/model_best.pt`
  - `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt`
  - `project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt`

Outputs:

- Environment report with exact package/tool versions.
- Checkpoint availability table with path, role, and status.
- A decision on whether any dependency must be installed or compiled before later phases.

Validation:

- Version commands run successfully.
- All required source/data/checkpoint paths are either present or explicitly marked missing.
- No dependency installation or model download is assumed without evidence.

Required report:

```text
project/__guidance__/report/phase_00_setup.md
```

### Phase 01: Dataset Inventory and Provenance Lock

Goal: lock the dataset snapshot so later results can be traced to exact local files.

Inputs from `data.md`:

- `project/dataset/ShuttleSet_raw_videos/`
- `project/dataset/ShuttleSet/set/match.csv`
- `project/dataset/ShuttleSet/set/homography.csv`
- `project/dataset/ShuttleSet/set/<match_name>/set*.csv`
- all six feature variants under `project/dataset/ShuttleSet/`

Data processing:

- Count raw `.mp4` and `.info.json` files.
- Count match rows, homography rows, match subdirectories, set CSV files, and per-stroke rows.
- Count clips per feature variant and split.
- Verify that each clip has the full file triple:
  - `*_joints.npy`
  - `*_pos.npy`
  - `*_shuttle.npy`
- Sample tensor shapes from train/val/test for each variant.
- Record class folders for merged and original label sets.

Outputs:

- Dataset inventory table.
- Optional machine-readable manifest, for example `project/outputs/inventory/dataset_manifest.jsonl` if implementation creates outputs later.
- Explicit note that the current feature folders are non-collated per-clip files.

Validation:

- Raw video count should match 44 MP4 files and 44 info JSON files unless the dataset changed.
- `match.csv` and `homography.csv` should each match the current 44-row ShuttleSet snapshot.
- Feature variants should have split counts consistent with `data.md`: 25,741 train, 4,241 val, 3,499 test clips per variant, unless re-audited counts differ.
- Any mismatch between annotation rows and feature clips must be documented rather than silently ignored.

Required report:

```text
project/__guidance__/report/phase_01_dataset_inventory.md
```

### Phase 02: Ground-Truth Tactical Table Construction

Goal: create a clean analysis table from ShuttleSet annotations before using model predictions.

Inputs:

- `project/dataset/ShuttleSet/set/match.csv`
- `project/dataset/ShuttleSet/set/homography.csv`
- `project/dataset/ShuttleSet/set/<match_name>/set*.csv`

Data processing:

- Load all set CSV rows as stroke-level records.
- Add match metadata from `match.csv`: match id, tournament, round, date, winner, loser, duration, downcourt, and source URL.
- Add `set_id` from the filename.
- Preserve original stroke label text in `type`.
- Preserve frame-level fields: `time`, `frame_num`, `rally`, `ball_round`.
- Preserve tactical fields: score, player, server, hit/landing position, player/opponent position, win/loss reason, and point winner.
- Define stable keys:
  - `match_id`
  - `match_name`
  - `set_id`
  - `rally`
  - `ball_round`

Outputs:

- A normalized stroke-level table suitable for data mining.
- Recommended output path when implementation exists:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
```

Validation:

- Row count should reconcile with the audited `set*.csv` total.
- Required columns should be present for every row.
- Key uniqueness should be checked on `(match_id, set_id, rally, ball_round)`.
- Missing `type`, `frame_num`, or outcome fields should be counted.
- UTF-8 label preservation must be verified because stroke labels and class folders contain Chinese text.

Required report:

```text
project/__guidance__/report/phase_02_ground_truth_tables.md
```

### Phase 03: BST Feature Collation

Goal: convert the per-clip ShuttleSet feature folders into the packed layout required by BST inference and training scripts.

Primary input:

```text
project/dataset/ShuttleSet/merged_seq100_between_2_hits_with_max_limits/
```

Reference code:

- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/preparing_data/shuttleset_dataset.py`

Data processing:

- Read clip triples from `train`, `val`, and `test`.
- Convert `joints`, `pos`, and `shuttle` arrays to `float32`.
- Use `make_seq_len_same(seq_len=100)` to pad or downsample each clip.
- Use COCO bone pairs from `get_bone_pairs('coco')`.
- Generate:
  - `J_only`
  - `JnB_interp`
  - `JnB_bone`
  - `Jn2B`
  - `pos`
  - `shuttle`
  - `videos_len`
  - `labels`
- Store labels with the merged 25-class mapping.

Recommended output path:

```text
project/outputs/bst_collated/merged_seq100_between_2_hits_with_max_limits/
```

Expected packed layout:

```text
<collated_root>/train/JnB_bone.npy
<collated_root>/train/pos.npy
<collated_root>/train/shuttle.npy
<collated_root>/train/videos_len.npy
<collated_root>/train/labels.npy
<collated_root>/val/...
<collated_root>/test/...
```

Validation:

- Each split row count must match the source clip count.
- `JnB_bone` shape should be `(N, 100, 2, 36, 2)` for the 2D merged seq100 setup because COCO has 17 joints and 19 bone vectors.
- `pos` shape should be `(N, 100, 2, 2)`.
- `shuttle` shape should be `(N, 100, 2)`.
- `videos_len` should be length `N` and should not exceed 100.
- `labels` should be length `N` and within `[0, 24]` for merged classes.

Required report:

```text
project/__guidance__/report/phase_03_bst_collation.md
```

### Phase 04: BST Fine-Tuning and Stroke Classification on ShuttleSet

Goal: fine-tune BST on the collated ShuttleSet features from all 44 videos, run stroke classification on the held-out protocol split, and join predictions back to the ground-truth table.

Inputs:

- Collated output from Phase 03.
- Recommended checkpoint:

```text
project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt
```

Reference code:

- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/main_on_shuttleset/bst_infer.py`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/main_on_shuttleset/bst_main.py`
- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/model/bst.py`

Model setup:

- Model: `BST_CG_AP`
- Pose style: `JnB_bone`
- Sequence length: `100`
- Class count: `25`
- Pose input channels: `2`
- Batch size: start with `128` if memory allows; lower it for CPU or small GPU runs.
- Initial checkpoint for fine-tuning: start from the recommended ShuttleSet checkpoint when shapes match; otherwise train from the project-owned baseline config and report why initialization was skipped.
- Training config must keep the Phase 03 contract. Do not use upstream defaults blindly because `bst_main.py` defaults to `seq_len=30`; the all-44 path must use the collated `seq_len=100` features unless Phase 03 is intentionally rebuilt with another sequence length.

Fine-tuning data:

- Use all collated split folders from Phase 03, which are derived from the 44-video ShuttleSet snapshot.
- Keep the protocol split unchanged for model selection: train split for training, val split for early stopping/model selection, test split for final protocol reporting.
- Save protocol checkpoints under:

```text
project/weights/bst_shuttleset_finetuned_protocol.pt
project/outputs/bst_training_all44/
```

- After the protocol report exists, optionally produce a deployment checkpoint trained on `train,val,test` and save it separately:

```text
project/weights/bst_shuttleset_finetuned_all44.pt
project/outputs/bst_training_all44_final/
```

- Add or use a project-owned training wrapper before long runs so the command captures seed, split roots, sequence length, checkpoint initialization, device, and output paths. Do not make persistent edits inside the upstream repository just to change experiment constants.

Data processing:

- Load collated train/val/test splits through `Dataset_npy_collated`.
- Fine-tune `BST_CG_AP` with `JnB_bone`, `seq_len=100`, `n_classes=25`, deterministic seed, and logged optimizer/scheduler settings.
- Select the best protocol checkpoint from val macro-F1 or the documented primary metric.
- Load the collated test split through `Dataset_npy_collated`.
- Run model inference.
- Export at least:
  - clip id or source key
  - predicted class id
  - predicted merged stroke label
  - ground-truth class id if available
- If the implementation is changed to keep logits, also export confidence or softmax probabilities.
- Join predictions to Phase 02 table using clip id components: match id, set id, rally, and ball round.

Recommended outputs:

```text
project/weights/bst_shuttleset_finetuned_protocol.pt
project/outputs/bst_training_all44/training_summary.json
project/outputs/predictions/bst_test_predictions.csv
project/outputs/tables/shuttleset_strokes_with_bst_predictions.csv
```

Validation:

- Training, validation, and test row counts must match the Phase 03 collated split counts.
- Tensor shapes must match the Phase 03 contract before the first optimizer step: `JnB_bone=(N, 100, 2, 36, 2)`, `pos=(N, 100, 2, 2)`, `shuttle=(N, 100, 2)`, `labels=(N,)`.
- Prediction row count should equal the selected split clip count.
- Predicted class ids must be valid for the 25-class merged label set.
- If ground-truth labels are available, compute accuracy and confusion matrix on the selected split.
- Report macro-F1, per-class F1, top-1 accuracy, and the confusion matrix for the protocol test split before producing any final-all44 checkpoint.
- Confirm that the join to the tactical table does not duplicate rows unexpectedly.

Required report:

```text
project/__guidance__/report/phase_04_bst_inference.md
```

### Phase 05: SA-CNN Fine-Tuning and Rally Filtering for Raw Broadcast Video

Goal: fine-tune SA-CNN on positive and negative frames from all 44 ShuttleSet raw MP4s, then extract rally-like intervals for the true unseen-video path.

Inputs:

- `project/dataset/ShuttleSet_raw_videos/*.mp4` for validation.
- A future unseen MP4 for deployment testing.
- SA-CNN model:

```text
external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt
```

Reference code:

- `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/sacnn.py`
- `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/video_resolver.py`
- `project/tools/phase05_build_sacnn_dataset.py`
- `project/tools/phase05_train_sacnn.py`
- `project/tools/phase05_finetune_all44.sh`

Setup:

- Load only `SACNNContainer` and `ShotAngleQueue`.
- Avoid the full `VideoResolver` unless restoring its additional RallyProcessor dependencies.
- Decide frame sampling rate, output interval schema, and minimum/maximum interval filtering rules.
- Use the local all-44 helper as the default fine-tuning entry point; it builds an ImageFolder dataset from `--video-ids 1-44`, validates that 44 videos produced positive and negative images, then runs `phase05_train_sacnn.py`.
- Preserve the preprocessing contract from the local trainer: resize `(216, 384)`, center crop `(216, 216)`, `ToTensor()`, and ImageNet normalization.

Fine-tuning commands:

Protocol checkpoint, with videos `35-39` for validation and `40-44` held out:

```bash
GPU_ID=0 EPOCHS=20 BATCH_SIZE=64 NUM_WORKERS=4 \
  bash project/tools/phase05_finetune_all44.sh protocol
```

Final deployment checkpoint after the protocol report is complete:

```bash
GPU_ID=0 EPOCHS=20 BATCH_SIZE=64 NUM_WORKERS=4 \
  bash project/tools/phase05_finetune_all44.sh final-all44
```

Default fine-tuning outputs:

```text
project/dataset/sa_cnn_data_all44/
project/outputs/sacnn_training_all44/
project/weights/sacnn_shuttleset_finetuned_all44.pt
```

Data processing:

- Build positive frame samples from annotated rally/hit intervals derived from `project/dataset/ShuttleSet/set/<match_name>/set*.csv`.
- Build negative samples from complement intervals with guard frames around positives.
- Record `manifest.csv`, `video_summary.csv`, and `summary.json` for the all-44 SA-CNN dataset.
- Fine-tune from the upstream `sacnn.pt` checkpoint unless `--from-scratch` is explicitly documented as an ablation.
- Decode video metadata: fps, width, height, frame count.
- Sample frames at the selected rate.
- Predict shot-angle state for sampled frames.
- Smooth predictions through the queue.
- Convert state transitions into frame intervals.
- Optionally export rally clips with ffmpeg/OpenCV for later phases.

Recommended outputs:

```text
project/dataset/sa_cnn_data_all44/manifest.csv
project/dataset/sa_cnn_data_all44/video_summary.csv
project/outputs/sacnn_training_all44/training_summary.json
project/weights/sacnn_shuttleset_finetuned_all44.pt
project/outputs/rallies/<video_id>/rally_intervals.csv
project/outputs/rallies/<video_id>/clips/rally_<rally_id>.mp4
```

Validation:

- `video_summary.csv` must contain 44 rows with status `ok`, positive images `>0`, and negative images `>0`; otherwise do not train a final checkpoint.
- Training report must include train/val size, class counts, seed, best val F1, precision, recall, and loss history.
- Evaluate the fine-tuned checkpoint on held-out videos `40-44` before using final-all44 training for deployment.
- On ShuttleSet videos, compare detected rally windows against `set*.csv.frame_num` coverage.
- Count how many annotated hit frames fall inside predicted rally intervals.
- Visually inspect a small sample of accepted and rejected intervals.
- Keep human visual review separate from machine coverage metrics.

Required report:

```text
project/__guidance__/report/phase_05_rally_filtering.md
```

### Phase 06: TrackNetV3 Fine-Tuning, Shuttle Tracking, and Denoising

Goal: fine-tune or verify TrackNetV3 on all 44 ShuttleSet videos, then produce frame-level shuttle trajectories from rally clips.

Current implementation status: implemented through the project wrapper
`project/tools/phase06_shuttle_tracking.py`. The wrapper loads the local
TrackNetV3 checkpoint, selects accepted Phase 05 rally intervals, runs
frame-batched TrackNet inference, maps predictions back to original broadcast
coordinates, writes raw and denoised trajectories, and emits run-level manifest,
validation, and summary files under `project/outputs/shuttle/`.

Inputs:

- Accepted rally intervals from Phase 05:

```text
project/outputs/rallies/<video_id>/rally_intervals.csv
```

- ShuttleSet raw broadcast videos:

```text
project/dataset/ShuttleSet_raw_videos/
```

- TrackNetV3 checkpoint:

```text
external_repos/TrackNetV3/exp/model_best.pt
```

Reference code:

- `external_repos/TrackNetV3/train.py`
- `external_repos/TrackNetV3/dataset.py`
- `external_repos/TrackNetV3/predict.py`
- `external_repos/TrackNetV3/denoise.py`
- `project/tools/phase06_shuttle_tracking.py`

Setup:

- Confirm CUDA availability or set CPU fallback policy.
- Use `--device auto` for CUDA when available and CPU fallback otherwise, or set
  `--device cuda --gpu-id <id>` for a specific GPU.
- Standardized input roots default to `project/outputs/rallies/` and
  `project/dataset/ShuttleSet_raw_videos/`.
- Standardized output root defaults to `project/outputs/shuttle/`; do not use
  TrackNetV3's default `pred_result` directory for project phase outputs.
- Before fine-tuning, build a project-owned TrackNet dataset wrapper that writes the upstream expected layout:

```text
project/dataset/tracknet_all44/<split>/match<id>/frame/<rally_id>/<frame>.png
project/dataset/tracknet_all44/<split>/match<id>/csv/<rally_id>_ball.csv
```

- Split the TrackNet dataset by video id, not by individual frames or rallies: `train=1-34`, `val=35-39`, `test=40-44`. Frames from a held-out video must not appear in train or validation.
- The TrackNet CSV labels must contain `Frame,Visibility,X,Y` in the same frame coordinate system as the extracted PNGs. If local labels cannot provide frame-level shuttle coordinates for all 44 raw videos, stop before training and write `Unclear from current codebase: frame-level TrackNet supervision for all 44 raw videos`.
- Do not use Phase 05 rally intervals alone as TrackNet supervision. Rally intervals provide temporal crops, not shuttle centers.
- Because upstream `train.py` reads `data_dir = 'TrackNetV2_Dataset'` from `external_repos/TrackNetV3/utils.py`, prefer a project wrapper or temporary symlink for the dataset root instead of permanently editing upstream constants.
- Save fine-tuned checkpoints outside the upstream experiment folder:

```text
project/outputs/tracknet_training_all44/
project/weights/tracknetv3_shuttleset_finetuned_all44.pt
```

Data processing:

- For fine-tuning, extract labeled rally frames from all 44 raw MP4s at original fps, resize only inside the TrackNet dataset transform, and preserve a mapping back to original frame ids.
- Convert verified shuttle coordinates to TrackNet's `512x288` heatmap coordinate space and record the original video width/height for inverse projection checks.
- Use videos `1-34` for optimizer updates, videos `35-39` for checkpoint selection/early stopping, and videos `40-44` only for final TrackNet protocol metrics.
- Fine-tune from `external_repos/TrackNetV3/exp/model_best.pt` when model state names and shapes match; otherwise report the mismatch and train a clean protocol run from the local TrackNet architecture.
- Select Phase 05 tasks from accepted intervals unless `--include-rejected` is
  explicitly set.
- Resolve each `video_id` to the matching ShuttleSet raw MP4 and process the
  `[start_frame, end_frame]` interval directly from the broadcast video.
- Resize frames to TrackNet's `512x288` input size, use the checkpoint's
  `num_frame=3` temporal window, and threshold heatmaps into shuttle centers.
- Write `Frame,Visibility,X,Y,OriginalFrame,TimeSec` CSV files for raw
  predictions.
- Denoise each trajectory by dropping out-of-range points, removing jumps above
  `--max-jump-px`, and linearly interpolating short gaps up to
  `--interpolate-gap`.
- Preserve per-rally raw CSV, denoised CSV, and metadata JSON for comparison.
- Record fps, width, height, frame count, original frame offsets, device, model
  file, and output paths.

Implemented outputs:

```text
project/dataset/tracknet_all44/
project/outputs/tracknet_training_all44/training_summary.json
project/weights/tracknetv3_shuttleset_finetuned_all44.pt
project/outputs/shuttle/<video_id>/<rally_id>_ball_raw.csv
project/outputs/shuttle/<video_id>/<rally_id>_ball_denoised.csv
project/outputs/shuttle/<video_id>/<rally_id>_metadata.json
project/outputs/shuttle/phase06_shuttle_tracking_manifest.csv
project/outputs/shuttle/phase06_shuttle_tracking_validation.csv
project/outputs/shuttle/phase06_shuttle_tracking_summary.json
```

Validation:

- TrackNet training dataset report must include selected video ids, split counts, rally/frame counts, visible/invisible label counts, coordinate ranges, and decode failures.
- Run a smoke training job on a small subset before the all-44 protocol run, checking image tensor shape, heatmap shape, device placement, and loss decrease over a few batches.
- Protocol evaluation must report accuracy, precision, recall, detection coverage, and localization error on the held-out test videos.
- Count rows against the selected interval length.
- Check coordinate ranges against width and height.
- Compute raw and denoised detection coverage as the fraction of frames with
  `Visibility=1`.
- Count removed jump points, out-of-range points, and interpolated short-gap
  points.
- Sample trajectory overlays for visual QA.
- For ShuttleSet validation, compare trajectory-derived hit frames in Phase 07 against `set*.csv.frame_num`.

Required report:

```text
project/__guidance__/report/phase_06_shuttle_tracking.md
```

### Phase 07: Exact Phase 02 Label Windows

Goal: use exact Phase 02 `frame_num_int` labels as the authoritative stroke
window source for known ShuttleSet videos, while preserving Phase 06-compatible
rally IDs for Phase 09.

Active implementation:

```text
project/tools/phase07_build_ground_truth_windows.py
```

Inputs:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
project/outputs/shuttle/phase06_shuttle_tracking_manifest.csv
```

Data processing:

- Assign each exact Phase 02 labeled frame to the Phase 06 interval containing
  it. Annotation rally IDs are preserved in `phase07_matches.csv`, but Phase 06
  rally IDs are used in the window table so Phase 09 can resolve shuttle tasks.
- Use exact `frame_num_int` values as events.
- Build midpoint-bounded windows between adjacent labels inside each Phase 06
  interval.
- Preserve Phase 02 `stable_key`, `clip_id`, stroke label, player, and server.
- Write uncovered Phase 02 labels separately when no Phase 06 interval contains
  them.

Implemented outputs:

```text
project/outputs/hit_frames/phase07_hit_frame_predictions.csv
project/outputs/hit_frames/phase07_stroke_windows.csv
project/outputs/hit_frames/phase07_validation.csv
project/outputs/hit_frames/phase07_matches.csv
project/outputs/hit_frames/phase07_uncovered_phase02_labels.csv
project/outputs/hit_frames/phase07_hit_frame_detection_summary.json
project/outputs/hit_frames/<video_id>/<rally_id>_events.csv
```

Current `40-44` result: 3,737 exact windows from 3,884 Phase 02 labels, with
147 labels outside current Phase 06 intervals.

For true unseen videos without Phase 02 labels, use the optional trajectory
detector path:

```text
project/tools/phase07_hit_frame_detection.py
project/tools/phase07_tune_hit_frame_parameters.py
```

Do not mix trajectory-detector outputs into the active known-ShuttleSet
`project/outputs/hit_frames/` bundle.

Validation:

- Every exported event must have an exact match with `abs_error=0`.
- Window and match row counts must agree.
- Count uncovered labels explicitly.
- Phase 09 must resolve every window to a Phase 06 task and exact Phase 02
  `clip_id`.

Required report:

```text
project/__guidance__/report/phase_07_hit_frame_detection.md
```

### Phase 08: User-Assisted Court Calibration and Homography

Goal: generate the court calibration needed to normalize player positions for videos that do not already have ShuttleSet homography rows.

Primary decision: use user-assisted court annotation as the default unseen-video workflow. Most badminton broadcast matches use one stable tactical camera position for rally play, while camera transitions usually occur during breaks, replays, or non-rally segments. Because Phase 05 already filters rally-like intervals, one user-provided court calibration per unseen raw video is the practical default.

monotrack should be treated as optional future automation or a fallback experiment, not as the core Phase 08 dependency. This avoids the current monotrack build/runtime risk while preserving accuracy through explicit human calibration and visual QA.

Inputs:

- Full raw unseen video.
- Phase 05 accepted rally intervals.
- A representative rally frame selected from the unseen video.
- ShuttleSet `homography.csv` for known-video validation.
- Optional monotrack source for future automation only:

```text
external_repos/monotrack/court-detection/
```

Setup:

- Provide a lightweight annotation UI or script that shows a representative rally frame and asks the user to click standardized badminton court landmarks.
- Prefer 6-8 court line-intersection points when visible; allow a minimum of 4 non-collinear points only when the court is partially occluded.
- Define the canonical badminton court coordinate system used for homography projection.
- Decide whether the output represents the whole video or a named camera segment.
- Keep monotrack build steps out of the required path unless explicitly evaluating automatic court detection.

Data processing:

- Select a representative frame from an accepted Phase 05 rally interval, ideally one with clear court lines and both singles sidelines visible.
- Let the user click court landmarks once for the raw unseen video.
- Save the clicked image points, canonical court points, source frame index, and source timestamp.
- Compute a homography representation compatible with BST preprocessing.
- Store both raw user points and the computed homography matrix.
- Apply the same homography to all accepted rally intervals from the video by default.
- If a camera cut, zoom, replay angle, or major alignment drift is detected inside accepted rally intervals, create a new named calibration segment and ask the user to annotate that segment separately.
- For known ShuttleSet videos, compare computed homography/corners against `homography.csv` using the same video-id protocol split: develop acceptance rules on videos `1-34`, choose thresholds or review criteria on videos `35-39`, and report final quality on videos `40-44`.

Recommended outputs:

```text
project/outputs/court/<video_id>/court_points.json
project/outputs/court/<video_id>/homography.json
project/outputs/court/<video_id>/court_overlay.jpg
project/outputs/court/<video_id>/calibration_segments.csv
```

Validation:

- Output contains enough user-clicked points to solve a stable homography.
- Reprojection error on clicked points is reported.
- Overlay image shows plausible court alignment on the source calibration frame.
- Sample 1-3 frames from multiple accepted Phase 05 rally intervals and overlay the saved court to check that the one-homography assumption still holds.
- Homography projects key court points into stable normalized court coordinates.
- Later player feet from Phase 09 should mostly project inside or near the court; out-of-court rates should be counted.
- ShuttleSet validation should report deviation from provided `homography.csv` separately for videos `1-34`, `35-39`, and `40-44`; do not claim unseen-video readiness without visual QA.
- Human visual review is the authority for unseen-video court calibration. Machine checks can flag drift, but they should not silently replace user approval.

Required report:

```text
project/__guidance__/report/phase_08_court_homography.md
```

### Phase 09: Pose, Player Position, and BST Input Normalization

Goal: reproduce the `.npy` feature contract for newly generated stroke clips.

Inputs:

- Stroke windows from Phase 07.
- Homography outputs from Phase 08.
- MMPose source:

```text
external_repos/mmpose/
```

Reference code:

- `external_repos/BST-Badminton-Stroke-type-Transformer/stroke_classification/preparing_data/prepare_train_on_shuttleset.py`

Setup:

- Install or activate MMPose and its dependencies.
- Confirm whether `MMPoseInferencer('human')` can load weights from cache or requires download.
- Define player-selection rules for the two in-court players.

Data processing:

- Run pose inference on frames in each stroke window.
- Keep COCO-17 keypoints.
- Select two players, ideally using court membership after homography projection.
- Normalize joints relative to bounding boxes with center alignment, matching BST preprocessing.
- Normalize shuttle positions by video width and height.
- Convert player feet to court coordinates through homography and normalize to `[0, 1]`.
- If any pose confidence thresholds, player-selection thresholds, normalization statistics, or QA acceptance rules are estimated from ShuttleSet, estimate them on videos `1-34`, select them on videos `35-39`, and reserve videos `40-44` for final reporting.
- Save per-clip feature triples:

```text
<clip_id>_joints.npy
<clip_id>_pos.npy
<clip_id>_shuttle.npy
```

Recommended outputs:

```text
project/outputs/features/<video_id>/<clip_id>_joints.npy
project/outputs/features/<video_id>/<clip_id>_pos.npy
project/outputs/features/<video_id>/<clip_id>_shuttle.npy
project/outputs/features/<video_id>/<clip_id>_validity.npz
```

Validation:

- `joints` shape is `(T, 2, 17, 2)` for the 2D path.
- `pos` shape is `(T, 2, 2)`.
- `shuttle` shape is `(T, 2)`.
- Normalized shuttle values should usually be within `[0, 1]` when visible.
- Normalized player court positions should be plausible; out-of-court values should be counted.
- Pose and position validity must be stored separately per player; observed and recovered values must remain distinguishable.
- Compare generated features on ShuttleSet clips against the stored `.npy` feature distribution by split (`1-34`, `35-39`, `40-44`) before using unseen videos.

Required report:

```text
project/__guidance__/report/phase_09_pose_position.md
```

### Phase 10: Integrated BST Inference

Goal: convert arbitrary Phase 09 feature manifests into the fixed BST input
contract, run the selected Phase 04 checkpoint, and assemble structured
prediction rows for evaluation and Phase 11.

Active implementation:

```text
project/tools/phase10_prepare_bst_inputs.py
project/tools/phase10_run_bst_inference.py
```

Active tracked input:

```text
project/outputs/features_yolo26x_bst_tracked_phase02_41_44/
```

Prepared tracked integration input:

```text
project/outputs/integration/bst_tracked_phase09_41_44/inputs/
```

Data processing:

- Read every Phase 09 manifest triple without requiring class-organized source
  folders.
- Reuse Phase 03 sequence-length and `JnB_bone` construction logic.
- Preserve candidate-only windows and mark their label as `-1`.
- Join exact Phase 02 and reference BST labels only by exact `clip_id`.
- Run inference over all candidates but compute accuracy only on exact labels.
- Treat the reference test subset as the primary unbiased result; keep
  reference validation metrics diagnostic because that split selected the
  protocol checkpoint.
- Report full-class, player-side, and stroke-type metrics by separate Phase 09
  Top/Bottom pose-quality strata.
- Preserve all source paths and metadata in the structured prediction table.

Recommended outputs:

```text
project/outputs/integration/<run_name>/inputs/
project/outputs/integration/<run_name>/phase10_predictions.csv
project/outputs/integration/<run_name>/phase10_structured_strokes.csv
project/outputs/integration/<run_name>/phase10_evaluation_by_group.csv
project/outputs/integration/<run_name>/phase10_logits.npy
project/outputs/integration/<run_name>/phase10_probabilities.npy
project/outputs/integration/<run_name>/phase10_summary.json
```

Validation:

- Prepared and predicted row counts must equal the selected Phase 09 manifest
  count.
- Prepared arrays must be finite and match the BST sequence-100 contract.
- Candidate-only rows must not enter accuracy metrics.
- Primary accuracy must use `reference_test_primary`.
- Report quality-stratified metrics before using predictions for Phase 11.
- Keep `baseline_old_phase09` and `bst_tracked_phase09_41_44` integration
  outputs in separate roots.

Required report:

```text
project/__guidance__/report/phase_10_integrated_inference.md
```

### Phase 11: Tactical Mining and Final Analysis

Goal: answer the data-mining questions using the structured rally table.

Inputs:

- Ground-truth table from Phase 02.
- Optional BST prediction table from Phase 04.
- Optional integrated unseen-video table from Phase 10.

Data processing:

- Normalize labels and side/player fields.
- Build rally sequences sorted by `(match_id, set_id, rally, ball_round)`.
- Compute stroke distributions by player, side, tournament, and score state.
- Compute transition matrices and sequential patterns.
- Analyze serve plus third-shot patterns.
- Build hit, landing, player-position, and opponent-position heatmaps.
- Attribute point outcomes using `win_reason`, `lose_reason`, and `getpoint_player`.
- When using predictions, report ground-truth and predicted analyses separately before mixing them.
- For any ShuttleSet prediction-based tactical analysis, preserve the protocol groups: videos `1-34` for development/training-derived outputs, videos `35-39` for validation-selected outputs, and videos `40-44` as final test outputs. Do not tune mining thresholds or narrative conclusions on videos `40-44` before reporting test results.

Recommended outputs:

```text
project/outputs/mining/stroke_distribution.csv
project/outputs/mining/transition_matrix.csv
project/outputs/mining/player_profiles.csv
project/outputs/mining/patterns.csv
project/outputs/mining/figures/
```

Validation:

- All mining tables should reconcile to the source structured row count.
- Sequence order must be validated within each rally.
- Analyses using predictions must include model coverage and error context.
- Prediction-based metrics and tactical tables must be broken out by videos `1-34`, `35-39`, and `40-44` before presenting an all-44 summary.
- Claims about tactics should identify whether they come from ground-truth annotations or model predictions.

Required report:

```text
project/__guidance__/report/phase_11_tactical_mining.md
```
