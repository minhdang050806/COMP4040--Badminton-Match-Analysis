# Data Description

This project currently keeps ShuttleSet as the only active badminton dataset under `project/dataset/`.

```text
project/dataset/
  ShuttleSet/              annotations and pre-extracted feature tensors
  ShuttleSet_raw_videos/   44 raw broadcast match videos and metadata JSON files
```

Verified local size:

- `project/dataset`: about `33G`
- `project/dataset/ShuttleSet`: about `6.4G`
- `project/dataset/ShuttleSet_raw_videos`: about `27G`

BadmintonDB and TenniSet are present in some upstream repo code paths, but their datasets are not part of the active `project/dataset` copy.

## Dataset Roles

There are three practical data layers:

1. Raw videos: full broadcast MP4 matches. Use these for the true unseen-video pipeline and for perception validation.
2. Annotation CSVs: match metadata, homography, and per-stroke labels. Use these for immediate data mining and as ground truth.
3. Pre-extracted feature tensors: pose, player position, and shuttle arrays for BST stroke classification. Use these for model training/inference without rerunning perception.

For the 44 ShuttleSet matches, most perception work has already been solved in the provided annotations and `.npy` files. The raw videos are mainly needed for end-to-end reproduction, visualization, or testing the unseen-video pipeline.

## Raw Videos

Path:

```text
project/dataset/ShuttleSet_raw_videos/
```

Verified local inventory:

- 44 `.mp4` files
- 44 `.info.json` files
- about `27G`

File naming pattern:

```text
<id> - <player_A>_<player_B>_<event>_<round>.mp4
<id> - <player_A>_<player_B>_<event>_<round>.info.json
```

Example:

```text
21 - An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals.mp4
21 - An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals.info.json
```

Use cases:

- Stage 1 rally/shot-angle filtering.
- TrackNetV3 shuttle tracking validation.
- Court detection and homography validation.
- Qualitative review of mined patterns.

The `.info.json` sidecars come from the video acquisition process and should be treated as source metadata, not model inputs.

## ShuttleSet Annotation Directory

Path:

```text
project/dataset/ShuttleSet/set/
```

Verified local inventory:

- 44 match subdirectories
- 104 non-empty `set*.csv` files
- 36,482 per-stroke annotation rows across `set*.csv`
- `match.csv` with 44 rows
- `homography.csv` with 44 rows

### `match.csv`

Path:

```text
project/dataset/ShuttleSet/set/match.csv
```

Header:

```text
id,video,tournament,round,year,month,day,set,duration,winner,loser,downcourt,url
```

Role:

- Match-level metadata table.
- Join key for raw videos, homography rows, and per-stroke set files.
- Useful data-mining dimensions: tournament, round, year/month/day, winner, loser, number of sets, duration, and downcourt orientation.

Important fields:

- `id`: numeric ShuttleSet match id.
- `video`: canonical match folder/name used under `set/`.
- `set`: number of sets in the match.
- `duration`: match duration in minutes.
- `winner`, `loser`: player names.
- `downcourt`: orientation flag used by the dataset.
- `url`: source video URL.

### `homography.csv`

Path:

```text
project/dataset/ShuttleSet/set/homography.csv
```

Header:

```text
id,video,homography_matrix,upleft_x,upright_x,downleft_x,downright_x,upleft_y,upright_y,downleft_y,downright_y
```

Role:

- Per-match court calibration.
- `homography_matrix` is a serialized `3x3` matrix.
- Corner fields describe four court corners in camera/image coordinates.
- BST preprocessing uses this information to project player foot positions from camera coordinates into court coordinates and normalize them.

For the 44 bundled matches, this file means court detection is already solved. For unseen videos, the system must produce an equivalent homography record, likely from monotrack output.

### Per-Set Stroke CSVs

Path pattern:

```text
project/dataset/ShuttleSet/set/<match_name>/set1.csv
project/dataset/ShuttleSet/set/<match_name>/set2.csv
project/dataset/ShuttleSet/set/<match_name>/set3.csv
```

Not every match has three set files; the verified local count is 104 set CSVs across 44 matches.

Header:

```text
rally,ball_round,time,frame_num,roundscore_A,roundscore_B,player,server,type,aroundhead,backhand,hit_height,hit_area,hit_x,hit_y,landing_height,landing_area,landing_x,landing_y,lose_reason,win_reason,getpoint_player,flaw,player_location_area,player_location_x,player_location_y,opponent_location_area,opponent_location_x,opponent_location_y,db
```

Row meaning:

- One row is one annotated stroke.
- `rally`: rally id within the set.
- `ball_round`: stroke index within the rally.
- `time`: timestamp string.
- `frame_num`: hit frame in the source video.
- `roundscore_A`, `roundscore_B`: score at the stroke.
- `player`: hitting player side, usually `A` or `B`.
- `server`: server indicator.
- `type`: ground-truth stroke type, stored in Chinese labels.
- `aroundhead`, `backhand`: stroke attributes when annotated.
- `hit_*`: hit height/area/image-coordinate annotations.
- `landing_*`: landing height/area/image-coordinate annotations.
- `lose_reason`, `win_reason`, `getpoint_player`: rally outcome labels.
- `player_location_*`, `opponent_location_*`: player/opponent court-location annotations.
- `db`: dataset/source flag.

Use cases:

- Immediate tactical analysis without running any model.
- Ground-truth validation for hit-frame detection, stroke classification, and structured rally assembly.
- Joining predicted stroke labels back to true `type`.
- Building score-aware or outcome-aware mining tasks.

Important caveat:

- The annotation table has 36,482 stroke rows, while each feature variant has 33,481 clips.
- Unclear from current codebase: why 3,001 annotated strokes are not represented as feature clips. Possible causes include filtering, missing detections, unusable clips, or class/sequence constraints, but the current files do not prove which.

## Pre-Extracted Feature Variants

Path:

```text
project/dataset/ShuttleSet/
```

Feature variants present:

```text
merged_seq100_between_2_hits_with_max_limits/
merged_seq30/
merged_seq30_3d/
seq100_between_2_hits_with_max_limits/
seq30/
seq30_3d/
```

Each variant uses this layout:

```text
<variant>/
  train/
    <class_name>/
      <clip_id>_joints.npy
      <clip_id>_pos.npy
      <clip_id>_shuttle.npy
  val/
    <class_name>/
      ...
  test/
    <class_name>/
      ...
```

Each clip is represented by three files:

- `*_joints.npy`: player skeletons.
- `*_pos.npy`: normalized player court positions.
- `*_shuttle.npy`: normalized shuttle positions.

Clip id pattern observed:

```text
<match_id>_<set_id>_<rally_id>_<ball_round>
```

Example:

```text
13_1_4_3_joints.npy
13_1_4_3_pos.npy
13_1_4_3_shuttle.npy
```

The feature arrays are non-collated per-clip files. They are not yet packed into the large split-level arrays expected by `Dataset_npy_collated`.

### Split Counts

All six variants have the same split counts:

| Split | Clips | File triples | Notes |
|---|---:|---:|---|
| `train` | 25,741 | 25,741 joints + 25,741 pos + 25,741 shuttle | training split |
| `val` | 4,241 | 4,241 joints + 4,241 pos + 4,241 shuttle | validation split |
| `test` | 3,499 | 3,499 joints + 3,499 pos + 3,499 shuttle | test split |
| total | 33,481 | 100,443 `.npy` files per variant | before collation |

### Variant Summary

| Variant | Classes | Pose dim | Window meaning | Recommended use |
|---|---:|---:|---|---|
| `merged_seq100_between_2_hits_with_max_limits` | 25 | 2D | between-hit clips, intended seq_len 100 | primary for merged BST inference |
| `seq100_between_2_hits_with_max_limits` | 35 | 2D | between-hit clips, intended seq_len 100 | fine-grained labels |
| `merged_seq30` | 25 | 2D | shorter single-hit window, intended seq_len 30 | ablation / older setting |
| `seq30` | 35 | 2D | shorter single-hit window, intended seq_len 30 | fine-grained ablation |
| `merged_seq30_3d` | 25 | 3D | seq30 with 3D pose | avoid unless specifically studying 3D |
| `seq30_3d` | 35 | 3D | seq30 with 3D pose | avoid unless specifically studying 3D |

`source.md` notes the 3D pose path is buggy per the upstream author, so the recommended baseline is 2D `merged_seq100_between_2_hits_with_max_limits`.

### Tensor Shapes

Observed samples:

| Variant | Split sample | `joints` shape | `pos` shape | `shuttle` shape |
|---|---|---:|---:|---:|
| `merged_seq100_between_2_hits_with_max_limits` | train | `(44, 2, 17, 2)` | `(44, 2, 2)` | `(44, 2)` |
| `merged_seq100_between_2_hits_with_max_limits` | val | `(61, 2, 17, 2)` | `(61, 2, 2)` | `(61, 2)` |
| `merged_seq100_between_2_hits_with_max_limits` | test | `(57, 2, 17, 2)` | `(57, 2, 2)` | `(57, 2)` |
| `merged_seq30` | train | `(25, 2, 17, 2)` | `(25, 2, 2)` | `(25, 2)` |
| `merged_seq30` | val/test | `(30, 2, 17, 2)` | `(30, 2, 2)` | `(30, 2)` |
| `merged_seq30_3d` | train | `(25, 2, 17, 3)` | `(25, 2, 2)` | `(25, 2)` |

Shape contract:

```text
joints:  (T, 2, 17, D)
pos:     (T, 2, 2)
shuttle: (T, 2)
```

where:

- `T` is clip length before loader/collation padding or downsampling.
- `2` in the second axis is top/bottom or two-player ordering.
- `17` is COCO-17 keypoints.
- `D=2` for 2D variants and `D=3` for 3D variants.

Important detail:

- Variant names include `seq30` or `seq100`, but the stored per-clip files are not necessarily already padded to exactly that length.
- Padding/downsampling to the target sequence length happens in `Dataset_npy` or the collation function via `make_seq_len_same`.

## Class Labels

The labels are side-aware. `Top_...` and `Bottom_...` distinguish which side/player orientation the stroke belongs to.

### Merged 25-Class Set

Used by:

```text
merged_seq100_between_2_hits_with_max_limits
merged_seq30
merged_seq30_3d
```

Class structure:

- `未知球種`
- 12 `Top_...` stroke classes
- 12 `Bottom_...` stroke classes

Classes observed in `merged_seq100_between_2_hits_with_max_limits/train`:

```text
Bottom_切球
Bottom_勾球
Bottom_平球
Bottom_挑球
Bottom_推球
Bottom_撲球
Bottom_擋小球
Bottom_放小球
Bottom_殺球
Bottom_發短球
Bottom_發長球
Bottom_長球
Top_切球
Top_勾球
Top_平球
Top_挑球
Top_推球
Top_撲球
Top_擋小球
Top_放小球
Top_殺球
Top_發短球
Top_發長球
Top_長球
未知球種
```

### Original 35-Class Set

Used by:

```text
seq100_between_2_hits_with_max_limits
seq30
seq30_3d
```

Class structure:

- 17 `Top_...` stroke classes
- 17 `Bottom_...` stroke classes
- `未知球種`

Additional fine-grained classes compared with the merged set include:

```text
點扣
防守回挑
後場抽平球
過渡切球
防守回抽
```

## Collated Format Required by BST Inference

The current feature folders are per-clip. BST's `bst_infer.py` uses `Dataset_npy_collated`, which expects packed arrays under each split:

```text
<collated_root>/
  train/
    J_only.npy
    JnB_interp.npy
    JnB_bone.npy
    Jn2B.npy
    pos.npy
    shuttle.npy
    videos_len.npy
    labels.npy
  val/
    ...
  test/
    ...
```

No such packed files were found under `project/dataset/ShuttleSet` during this audit.

Collation behavior from upstream BST preprocessing:

- Load all clip-level `*_joints.npy`, `*_pos.npy`, and `*_shuttle.npy` files.
- Convert arrays to `float32`.
- Pad/downsample to target `seq_len`.
- Create COCO bone features and interpolated limb features.
- Save `J_only`, `JnB_interp`, `JnB_bone`, `Jn2B`, `pos`, `shuttle`, `videos_len`, and `labels`.

For the recommended checkpoint, the relevant packed feature should be:

```text
pose_style = JnB_bone
seq_len = 100
classes = 25 merged classes
source variant = project/dataset/ShuttleSet/merged_seq100_between_2_hits_with_max_limits
```

## Weights Related to the Dataset

Path:

```text
project/weights/on_ShuttleSet/
```

Verified local size:

- about `347M`

This directory contains ShuttleSet-trained BST and baseline checkpoints:

- `bst_0_*`
- `bst_CG_AP_*`
- `blockgcn_*`
- merged and non-merged variants
- 2D and 3D variants
- seq30 and between-hit seq100 variants

Recommended checkpoint from `source.md`:

```text
project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt
```

This matches the preferred data setting:

- ShuttleSet only
- merged 25-class labels
- between-hit seq100 features
- 2D `JnB_bone` pose representation
- `BST_CG_AP` model

## Data-Mining Entry Points

Fastest ground-truth path:

1. Read `project/dataset/ShuttleSet/set/match.csv`.
2. Read every `project/dataset/ShuttleSet/set/<match_name>/set*.csv`.
3. Join set rows to match metadata through `match.csv.video` and/or match id inferred from the raw video/feature clip id.
4. Use `type`, `player`, `server`, `frame_num`, court-location fields, score fields, and outcome fields for tactical analysis.

Feature/model path:

1. Use `merged_seq100_between_2_hits_with_max_limits` as the feature source.
2. Collate it into the packed BST format.
3. Run the recommended BST checkpoint.
4. Join predictions back to annotation rows by clip id fields: match id, set id, rally id, and ball round.
5. Compare predicted labels against `type`.

Raw-video path:

1. Use `ShuttleSet_raw_videos` for perception validation.
2. Use `match.csv` to map raw video id/name to annotations.
3. Use `homography.csv` as ground-truth court calibration for the known 44 videos.
4. Use `set*.csv.frame_num` as ground-truth hit frames.

## Caveats

- The dataset uses Chinese stroke labels; downstream code should preserve UTF-8 paths and labels.
- The raw feature directories contain many Unicode class folder names; avoid shell scripts that assume ASCII-only paths.
- The feature clips are already normalized. Re-normalizing them would change the BST input distribution.
- The stored feature clips are non-collated; `bst_infer.py` expects collated split arrays.
- `seq30`/`seq100` in a folder name is the intended model sequence length, not a guarantee that every stored per-clip file already has exactly that `T`.
- Unclear from current codebase: the exact filtering rule that maps 36,482 annotation rows to 33,481 feature clips.
- Unclear from current codebase: whether every raw video `.info.json` is needed after acquisition; keep them as provenance metadata.
