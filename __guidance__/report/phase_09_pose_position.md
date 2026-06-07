# Phase 09: BST-Compatible Tracked Pose and Position Features

## Status

Implementation complete. Full YOLO inference has not been run.

The rebuilt Phase 09 has two separable changes:

1. `bst-compatible` temporal windows reproduce the upstream
   `between_2_hits_with_max_limits` construction.
2. Tracked YOLO26x features preserve available players, reject out-of-court
   detections, stabilize Top/Bottom ordering, recover short gaps, and store
   separate pose and position validity masks.

Do not overwrite these earlier baselines:

```text
project/outputs/features_yolo26x_41_44
project/outputs/features_yolo26x_tracked_phase02_40_44
```

Use this new root:

```text
project/outputs/features_yolo26x_bst_tracked_phase02_40_44
```

## Temporal Window Contract

The old midpoint ablation consumes the Phase 07 boundaries unchanged:

```text
midpoint(previous, current) -> midpoint(current, next)
```

The default `bst-compatible` strategy follows the upstream BST source:

```text
start = previous hit
end_exclusive = next hit + fps / 4

first hit fallback: current - fps / 2
last hit fallback: current + fps / 2
maximum context: current - 1.5 seconds -> current + 1.5 seconds + fps / 4
```

The extractor stores an inclusive `window_end_original`, so its frame count is:

```text
window_end_original - window_start_original + 1
```

Static analysis on the current exact Phase 02-derived `40-44` handoff:

| Source | Mean length | Median | Min | Max |
|---|---:|---:|---:|---:|
| Old generated features, 2,986 matches | 26.74 | not reported | not reported | not reported |
| Phase 07 midpoint windows, current 3,737 matches | 50.80 | 30 | 1 | 833 |
| New BST-compatible windows, current 3,737 matches | 57.96 | 58 | 7 | 97 |
| Reference BST features, current 3,737 matches | 58.70 | not reported | 6 | 98 |

The new temporal construction is close to the matched reference distribution
before pose inference. Its matched length MAE is `1.05` frames, versus `45.59`
for the midpoint ablation.

## Player Feature Contract

Player axis ordering is fixed:

```text
player 0 = Top player = far side = normalized court y <= 0.5
player 1 = Bottom player = near side = normalized court y > 0.5
```

For every frame:

- keep any available in-court detection;
- use bbox-bottom-center when ankle keypoints are missing;
- reject projected feet outside `[-0.2, 1.2]` in either coordinate;
- assign candidates by court side and previous court position;
- forget stale track memory after 10 missing frames so a player can be
  reacquired;
- preserve position when pose quality is insufficient;
- recover gaps independently for each player and feature type:
  - gap `<= 2`: forward-fill;
  - bounded gap `<= 10`: linear interpolation;
  - gap `> 10`: remain missing.

Outputs per clip:

```text
<clip_id>_joints.npy       # (T, 2, 17, 2)
<clip_id>_pos.npy          # (T, 2, 2)
<clip_id>_shuttle.npy      # (T, 2)
<clip_id>_validity.npz     # separate observed and recovered-valid masks
```

## Step 1: Window Ablation Without Inference

Run both commands. They do not load YOLO or perform inference.

```bash
python3 project/tools/phase09_pose_position_features.py \
  --video-id 40-44 \
  --window-strategy midpoint \
  --analyze-windows-only \
  --output-root project/outputs/phase09_window_ablation/midpoint

python3 project/tools/phase09_pose_position_features.py \
  --video-id 40-44 \
  --window-strategy bst-compatible \
  --analyze-windows-only \
  --output-root project/outputs/phase09_window_ablation/bst_compatible
```

Acceptance criterion:

```text
BST-compatible mean clip length should be close to the matched reference mean
of approximately 58.70 frames and should not contain midpoint-window outliers.
```

## Step 2: Full Two-GPU Inference

Run:

```bash
python3 project/tools/phase09_pose_position_features.py \
  --video-id 40-44 \
  --window-strategy bst-compatible \
  --output-root project/outputs/features_yolo26x_bst_tracked_phase02_40_44 \
  --devices 0,1 \
  --batch-size 128 \
  --half \
  --imgsz 640
```

There are two independent workers, one per GPU. Each clip contains at most
about 97 frames under the corrected window contract, so a batch size above 128
normally does not increase the effective YOLO batch. Increase `--batch-size`
only after confirming memory headroom and throughput.

## Step 3: Matched Reference Comparison

After inference completes, run:

```bash
python3 project/tools/phase09_compare_reference_features.py \
  --feature-root project/outputs/features_yolo26x_bst_tracked_phase02_40_44 \
  --reference-root project/dataset/ShuttleSet/merged_seq100_between_2_hits_with_max_limits \
  --output-root project/outputs/phase09_reference_comparison/bst_tracked_40_44
```

The comparison matches generated and reference features by exact `clip_id` and
reports:

- clip length and length absolute error;
- valid frame ratios for both players;
- pose and position zero rates;
- shuttle zero rate;
- player-position values outside `[-0.2, 1.2]`;
- Top/Bottom side consistency;
- temporal offset of the current hit and offset absolute error.

Expected files:

```text
project/outputs/phase09_reference_comparison/bst_tracked_40_44/
  phase09_matched_reference_comparison.csv
  phase09_matched_reference_comparison_summary.json
```

## Baseline Comparison

The comparison tool was smoke-tested against the old Phase 09 output and
matched exactly 2,986 clips:

| Metric | Old generated | Reference |
|---|---:|---:|
| Mean clip length | 26.74 | 58.95 |
| Both-pose valid frame rate | 49.43% | 93.92% |
| Both-position valid frame rate | 49.43% | 93.92% |
| Pose tensor zero rate | 50.65% | 6.29% |
| Position tensor zero rate | 50.57% | 6.08% |
| Player side consistency | 99.37% | 99.94% |

Use the old output only as the baseline. The new run should materially reduce
pose/position dropout while preserving near-perfect side consistency and
eliminating accepted positions outside the court tolerance.

## Acceptance Gates

Before handing the new root to Phase 10:

- all selected windows have complete feature bundles;
- mean clip length is close to the reference distribution;
- `player_pos_out_of_filter_range == 0`;
- matched comparison position out-of-tolerance rate is zero;
- Top/Bottom side consistency remains near 100%;
- both-player pose and position valid rates improve materially over the old
  baseline;
- visual QA confirms player identities do not swap across frames.

Phase 10 should consume the new root only after these checks pass.
