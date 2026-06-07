# Phase 07: Phase 02 Label Windows

## Status

For the known ShuttleSet videos currently covered by Phase 06, Phase 07 now
uses exact Phase 02 labels instead of trajectory-based hit-frame detection.

Authoritative generator:

```text
project/tools/phase07_build_ground_truth_windows.py
```

Authoritative output root:

```text
project/outputs/hit_frames/
```

The older trajectory detector remains available for experiments on unlabeled
videos:

```text
project/tools/phase07_hit_frame_detection.py
project/tools/phase07_tune_hit_frame_parameters.py
```

It is not the active Phase 09 window source for known ShuttleSet data.

## Inputs

Exact hit-frame labels and clip identities:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
```

Required Phase 02 fields:

```text
match_id
frame_num_int
stable_key
clip_id
set_id
rally_id
ball_round_id
stroke_type_ground_truth
player
server
```

Phase 06 intervals:

```text
project/outputs/shuttle/phase06_shuttle_tracking_manifest.csv
```

Phase 02 annotation rally IDs do not always match Phase 06 interval IDs.
Therefore, the generator assigns each exact labeled frame to the Phase 06
interval containing it. This preserves the `(video_id, rally_id)` lookup needed
by Phase 09.

## Window Construction

Within each Phase 06 interval:

- sort Phase 02 labels by `frame_num_int`;
- use the exact labeled frame as `event_frame_original`;
- use midpoints between adjacent labeled frames as stroke-window boundaries;
- clamp first and last windows to the Phase 06 interval;
- preserve the exact Phase 02 `clip_id`;
- emit exact matches with `abs_error=0`.

Labels outside all available Phase 06 intervals are not exported as windows,
because Phase 09 cannot process them without a corresponding Phase 06 task.
They are written to:

```text
project/outputs/hit_frames/phase07_uncovered_phase02_labels.csv
```

## Rebuild Command

```bash
rm -rf project/outputs/hit_frames

python3 project/tools/phase07_build_ground_truth_windows.py \
  --ground-truth-table project/outputs/tables/shuttleset_ground_truth_strokes.csv \
  --phase06-manifest project/outputs/shuttle/phase06_shuttle_tracking_manifest.csv \
  --output-root project/outputs/hit_frames \
  --context-frames 15
```

The selected videos are determined by the Phase 06 manifest. The current
manifest covers videos `40-44`.

## Outputs

```text
project/outputs/hit_frames/phase07_hit_frame_predictions.csv
project/outputs/hit_frames/phase07_stroke_windows.csv
project/outputs/hit_frames/phase07_matches.csv
project/outputs/hit_frames/phase07_validation.csv
project/outputs/hit_frames/phase07_uncovered_phase02_labels.csv
project/outputs/hit_frames/phase07_hit_frame_detection_summary.json
project/outputs/hit_frames/<video_id>/<phase06_rally_id>_events.csv
```

`phase07_stroke_windows.csv` and `phase07_matches.csv` preserve the existing
Phase 09-compatible schemas.

## Current Result

| Metric | Value |
| --- | ---: |
| Phase 06 rallies | 472 |
| Phase 02 labels on available videos | 3,884 |
| Exact exported events/windows | 3,737 |
| Exact matches | 3,737 |
| Match error | 0 frames |
| Labels outside Phase 06 intervals | 147 |

Per-video coverage:

| Video | Phase 02 labels | Exported windows |
| --- | ---: | ---: |
| 40 | 568 | 545 |
| 41 | 1,200 | 1,153 |
| 42 | 644 | 619 |
| 43 | 577 | 558 |
| 44 | 895 | 862 |

## Phase 09 Handoff

Phase 09 defaults to:

```text
project/outputs/hit_frames/phase07_stroke_windows.csv
project/outputs/hit_frames/phase07_matches.csv
```

These files now contain exact Phase 02-derived labels. Existing Phase 09
outputs generated from the previous detector windows are stale baselines and
must be regenerated before claiming the exact-label pipeline.

## Scope

This label-derived Phase 07 path is valid for known ShuttleSet
training/evaluation and baseline construction. A true unseen video has no Phase
02 labels, so deployment inference must use the trajectory detector or another
hit-frame model.
