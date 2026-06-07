# Phase 02: Ground-Truth Tactical Table Construction

## Objective

Build a normalized stroke-level ShuttleSet ground-truth table from the annotation CSVs, with match metadata, homography availability, stable join keys, and validation artifacts for downstream tactical mining and later BST prediction joins.

Status: complete for the current checkout.

## Inputs

Primary annotation files:

- `project/dataset/ShuttleSet/set/match.csv`
- `project/dataset/ShuttleSet/set/homography.csv`
- `project/dataset/ShuttleSet/set/<match_name>/set*.csv`

Phase 01 handoff:

- `project/outputs/inventory/phase01_set_csv_inventory.csv`
- `project/outputs/inventory/phase01_dataset_inventory_summary.json`

Reference docs:

- `project/__guidance__/data.md`
- `project/__guidance__/systems.md`
- `project/__guidance__/report/phase_01_dataset_inventory.md`

## Commands

Main implementation command:

```bash
python project/tools/phase02_build_ground_truth_tables.py
```

Validation commands:

```bash
python -m py_compile project/tools/phase02_build_ground_truth_tables.py
find project/outputs/tables -maxdepth 1 -type f -print
wc -l project/outputs/tables/*
```

Additional Python checks were run to verify row counts, stable-key uniqueness, homography coverage, missing required fields, label distribution, match summaries, and rally counts.

## Outputs

Implementation:

- `project/tools/phase02_build_ground_truth_tables.py`

Generated tables and summaries:

- `project/outputs/tables/shuttleset_ground_truth_strokes.csv`
- `project/outputs/tables/phase02_ground_truth_summary.json`
- `project/outputs/tables/phase02_label_distribution.csv`
- `project/outputs/tables/phase02_match_summary.csv`
- `project/outputs/tables/phase02_set_summary.csv`
- `project/outputs/tables/phase02_rally_summary.csv`
- `project/outputs/tables/phase02_missing_required_counts.csv`
- `project/outputs/tables/phase02_duplicate_keys.csv`
- `project/outputs/tables/phase02_homography_video_name_mismatches.csv`

Line counts:

| File | Lines | Data rows |
|---|---:|---:|
| `shuttleset_ground_truth_strokes.csv` | 36,483 | 36,482 |
| `phase02_ground_truth_summary.json` | 48 | JSON summary |
| `phase02_label_distribution.csv` | 20 | 19 |
| `phase02_match_summary.csv` | 45 | 44 |
| `phase02_set_summary.csv` | 105 | 104 |
| `phase02_rally_summary.csv` | 3,684 | 3,683 |
| `phase02_missing_required_counts.csv` | 10 | 9 |
| `phase02_duplicate_keys.csv` | 1 | 0 |
| `phase02_homography_video_name_mismatches.csv` | 4 | 3 |

## Data Processing

The implementation loads `match.csv`, `homography.csv`, and all 104 `set*.csv` files under the 44 match folders.

For every stroke row, the script adds:

- Stable identifiers:
  - `stable_key`: `match_id:set_id:rally_id:ball_round_id`
  - `clip_id`: `match_id_set_id_rally_id_ball_round_id`
  - `source_set_csv`
  - `source_row_index`
- Match metadata:
  - tournament
  - match round
  - match date
  - declared set count
  - duration
  - winner
  - loser
  - downcourt
  - source URL
- Homography fields:
  - `has_homography`
  - `homography_video`
  - `homography_matrix`
  - four court-corner coordinate pairs
- Preserved stroke fields:
  - `rally`, `ball_round`, `time`, `frame_num`
  - score, player, server
  - original `type`
  - `stroke_type_ground_truth`
  - hit, landing, player-location, opponent-location fields
  - outcome fields: `lose_reason`, `win_reason`, `getpoint_player`
- Derived fields:
  - integer-normalized `rally_id`, `ball_round_id`, `frame_num_int`
  - `score_state`
  - `has_outcome`

The stable key is unique across all output rows and is the primary key for Phase 02 outputs. The `clip_id` intentionally follows the observed feature filename pattern used by ShuttleSet feature clips, but Phase 01 confirmed that not every annotation row has a corresponding feature clip.

## Homography Join Decision

The script joins `homography.csv` by numeric `match_id`, not by video/folder name.

Reason: three homography rows have video-name spelling or spacing differences compared with `match.csv.video` and the set folder names:

| ID | Match video name | Homography video name |
|---:|---|---|
| 16 | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN_Indonesia_Masters_2020_SemiFinals` | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals` |
| 18 | `Viktor_AXELSEN_SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` | `Viktor_AXELSEN _SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` |
| 37 | `Viktor_Axelsen_Hans_Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals` | `Viktor_Axelsen_Hans-Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals` |

Joining by numeric id gives homography coverage for all 44 matches. The mismatch details are written to:

```text
project/outputs/tables/phase02_homography_video_name_mismatches.csv
```

## Validation

Summary from `phase02_ground_truth_summary.json`:

| Check | Result |
|---|---:|
| Output stroke rows | 36,482 |
| Unique stable keys | 36,482 |
| Duplicate stable keys | 0 |
| Match rows joined | 44 |
| Homography rows available | 44 |
| Set CSV files processed | 104 |
| Rows with any outcome field | 3,509 |
| Rows without outcome fields | 32,973 |
| Unique ground-truth stroke labels | 19 |
| Validation status | `passed` |

Required-field missing counts:

| Column | Missing rows |
|---|---:|
| `match_id` | 0 |
| `match_name` | 0 |
| `set_id` | 0 |
| `rally` | 0 |
| `ball_round` | 0 |
| `frame_num` | 0 |
| `player` | 0 |
| `server` | 0 |
| `stroke_type_ground_truth` | 0 |

Player/server distributions:

| Field | Distribution |
|---|---|
| `player` | `A`: 18,281; `B`: 18,201 |
| `server` | `1`: 3,659; `2`: 29,315; `3`: 3,508 |

Rally summary:

- 3,683 rally records were generated.
- Minimum strokes in a rally summary: 1.
- Maximum strokes in a rally summary: 65.
- Not every rally summary has an explicit terminal outcome field; the table preserves this as data rather than inferring outcomes.

UTF-8 preservation:

- The output contains 19 non-ASCII ground-truth stroke labels.
- Chinese stroke labels were preserved in `stroke_type_ground_truth` and `type`.

Top stroke labels:

| Stroke type | Count |
|---|---:|
| `放小球` | 6,290 |
| `挑球` | 5,330 |
| `擋小球` | 3,620 |
| `推球` | 2,925 |
| `長球` | 2,922 |
| `殺球` | 2,586 |
| `切球` | 2,144 |
| `發短球` | 2,051 |
| `點扣` | 1,648 |
| `未知球種` | 1,406 |

## Assumptions and Unclear Items

- Homography join by `match_id` is intentional because video-name joins would miss three known rows.
- `has_outcome=False` is expected for non-terminal strokes; it is not treated as missing data.
- Unclear from current codebase: why some rally summaries have no explicit terminal outcome field.
- Unclear from current codebase: whether `server=3` is a terminal/outcome code, a dataset-specific status code, or another annotation convention.
- Unclear from current codebase: the exact filtering rule that maps 36,482 annotation rows to 33,481 feature clips, so Phase 02 does not claim all rows are BST-feature-ready.

## Blockers

No blocker for Phase 03 collation.

Important handoff caveat: Phase 03 should use feature folders, not the full 36,482-row annotation table, because Phase 01 showed each feature variant has 33,481 clip branches.

## Next Phase Handoff

Use the Phase 02 table for data mining and later prediction joins:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
```

Use this stable key for table-level joins:

```text
match_id:set_id:rally_id:ball_round_id
```

Use this clip id pattern when joining to feature filenames or BST predictions:

```text
match_id_set_id_rally_id_ball_round_id
```

Phase 03 should collate:

```text
project/dataset/ShuttleSet/merged_seq100_between_2_hits_with_max_limits/
```

Phase 04 should join predictions back to:

```text
project/outputs/tables/shuttleset_ground_truth_strokes.csv
```
