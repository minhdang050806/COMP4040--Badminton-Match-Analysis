# Phase 06 Shuttle Tracking and Denoising

## Objective

Run TrackNetV3 shuttle tracking over the accepted Phase 05 rally intervals and
write frame-level raw and denoised shuttle trajectories for downstream
hit-frame detection.

## Source Files and Directories Inspected

- `project/tools/phase06_shuttle_tracking.py`
- `project/__guidance__/systems.md`
- `external_repos/TrackNetV3/`
- `external_repos/TrackNetV3/exp/model_best.pt`
- `project/outputs/rallies/<video_id>/rally_intervals.csv`
- `project/dataset/ShuttleSet_raw_videos/`
- `project/outputs/shuttle/`

## Data Inputs and Split/Range

The completed Phase 06 run used accepted Phase 05 intervals for ShuttleSet raw
videos `40`, `41`, `42`, `43`, and `44`.

| Video id | Accepted intervals processed |
|---:|---:|
| 40 | 80 |
| 41 | 113 |
| 42 | 91 |
| 43 | 85 |
| 44 | 103 |

Total processed tasks: `472` accepted rally intervals.

Total processed frames: `196792`.

The raw video source root was:

```text
project/dataset/ShuttleSet_raw_videos/
```

The Phase 05 interval source root was:

```text
project/outputs/rallies/
```

## Model and Inference Configuration

Checkpoint:

```text
external_repos/TrackNetV3/exp/model_best.pt
```

Recorded checkpoint parameters from the completed output summary:

```text
model_name=TrackNetV2
num_frame=3
input_type=2d
epochs=50
batch_size=4
learning_rate=0.001
tolerance=4
save_dir=exp
```

Recorded runtime device:

```text
cuda:1
```

`torch_cuda_available` was recorded as `true`.

## Commands Run

Working directory:

```text
/home/dang.cpm/__MY_SPACE__/VinUni/Data-Mining
```

The completed output summary records the script, model path, device, selected
tasks, and outputs. A reproducibility gap was found while implementing Phase 07:
the original aggregate summary did not persist every CLI argument. This has been
fixed in `project/tools/phase06_shuttle_tracking.py` for future Phase 06 runs by
adding a `runtime_config` block to the summary JSON. The completed trajectory
bundle predates that patch, so the exact run command below is the reproducible
command shape for the completed selected range:

```bash
python project/tools/phase06_shuttle_tracking.py \
  --video-id 40 \
  --video-id 41 \
  --video-id 42 \
  --video-id 43 \
  --video-id 44 \
  --max-rallies 0 \
  --device cuda \
  --gpu-id 1 \
  --output-root project/outputs/shuttle
```

Validation commands run while writing this report:

```bash
python project/tools/phase06_shuttle_tracking.py --help
python - <<'PY'
import csv, json
from pathlib import Path
root = Path("project/outputs/shuttle")
summary = json.loads((root / "phase06_shuttle_tracking_summary.json").read_text())
manifest = list(csv.DictReader((root / "phase06_shuttle_tracking_manifest.csv").open()))
validation = list(csv.DictReader((root / "phase06_shuttle_tracking_validation.csv").open()))
print(summary["task_count"], len(manifest), len(validation))
PY
```

## Generated Outputs

Run-level files:

```text
project/outputs/shuttle/phase06_shuttle_tracking_manifest.csv
project/outputs/shuttle/phase06_shuttle_tracking_validation.csv
project/outputs/shuttle/phase06_shuttle_tracking_summary.json
```

Per-rally files:

```text
project/outputs/shuttle/<video_id>/<rally_id>_ball_raw.csv
project/outputs/shuttle/<video_id>/<rally_id>_ball_denoised.csv
project/outputs/shuttle/<video_id>/<rally_id>_metadata.json
```

Trajectory schema:

```text
Frame,Visibility,X,Y,OriginalFrame,TimeSec
```

Manifest schema:

```text
video_id,rally_id,source_video,start_frame,end_frame,fps,width,height,frame_count,processed_frames,device,model_file,raw_csv,denoised_csv
```

Validation schema:

```text
video_id,rally_id,raw_rows,raw_visible_rows,raw_coverage,raw_out_of_range_visible_rows,denoised_rows,denoised_visible_rows,denoised_coverage,denoised_out_of_range_visible_rows,removed_jump_count,out_of_range_count,interpolated_count,runtime_sec
```

## Validation Checks

Run-level summary from `project/outputs/shuttle/phase06_shuttle_tracking_summary.json`:

| Metric | Value |
|---|---:|
| Tasks | 472 |
| Processed frames | 196792 |
| Raw visible rows | 170488 |
| Raw coverage | 0.866336 |
| Denoised visible rows | 179057 |
| Denoised coverage | 0.909879 |
| Removed jump points | 19384 |
| Interpolated points | 27953 |
| Out-of-range visible rows | 0 |
| Sum of per-rally runtime seconds | 4245.239 |

Coverage range over denoised per-rally outputs:

```text
min=0.186813
max=1.000000
zero-visible-denoised-rallies=0
```

Machine checks passed:

- Manifest and validation CSVs each contain `472` data rows.
- Total processed frames in the manifest reconcile with the summary JSON:
  `196792`.
- All denoised visible coordinates were within the recorded frame dimensions.
- Every processed rally produced raw CSV, denoised CSV, and metadata JSON files.
- The wrapper help command runs successfully, confirming the CLI entry point is
  importable in the active environment.

Visual QA artifact:

```text
project/outputs/visualizations/phase06_rally_comparison/phase06_rallies_side_by_side.jpg
```

The sheet compares video `40` rallies `7`, `60`, and `27`, selected near low,
median, and high denoised-coverage quantiles. Each rally is one chronological
row with sampled frames placed left to right.

The shuttle trails are generally coherent, but the final sampled frames expose
post-rally broadcast close-ups inside accepted Phase 05 intervals. This means
Phase 06 trajectory coverage is not sufficient evidence of correct rally
boundaries; Phase 05 interval-tail calibration remains necessary.

## Assumptions and Unclear Items

- The completed run selected held-out decodable videos `40` through `44`, which
  matches the Phase 05 engineering handoff. Video `09` has a small accepted
  Phase 05 output, but it was not part of the held-out Phase 06 production run.
- The script summary does not persist threshold, inference batch size,
  `max_jump_px`, or `interpolate_gap`. The defaults in
  `project/tools/phase06_shuttle_tracking.py` are `threshold=0.5`,
  `batch_size=1`, `max_jump_px=100.0`, and `interpolate_gap=5`. Future reruns
  will persist these values under `runtime_config`.
- The saved comparison sheet is representative visual QA, not an exhaustive
  human review of all Phase 06 intervals.

## Blockers and Next Phase Handoff

No machine-output blocker remains for Phase 07 on videos `40` through `44`.
Phase 07 can consume the denoised trajectories at:

```text
project/outputs/shuttle/<video_id>/<rally_id>_ball_denoised.csv
```

Before treating these trajectories as final research labels, perform visual QA
on a representative sample, especially rallies near the minimum denoised
coverage. Phase 07 should derive candidate hit frames from the denoised
trajectory and compare them against ShuttleSet `set*.csv.frame_num` annotations.
