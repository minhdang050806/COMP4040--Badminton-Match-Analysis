# Project Source Inventory & System Plan

> **Ultimate goal:** Take a **raw, unseen badminton video** → run an AI perception pipeline →
> produce structured rally data → run **data-mining / tactical analysis** on top.

This document inventories everything currently available in the workspace (data, weights,
scripts) and maps it to the end-to-end pipeline, with an honest gap list for the "unseen video" goal.

## Finalized Stack
1. **ShuttleSet raw videos** (44 matches)
2. **ShuttleNet / BST** pretrained models (stroke classification)
3. **TrackNetV3** pretrained (shuttle tracking + hit-frame detection)
4. **monotrack C++** (court detection + homography)
5. **MMPose** (pose estimation)
6. **`sacnn.pt`** (shot-angle CNN — rally trimming only)
7. **Data-mining tactics** (self-implemented)

---

## 1. Target Pipeline

```
Raw video (unseen, full broadcast)
   │
   ├─ 1. Rally trimming             → sacnn.pt (shot-angle) → which segments are real rallies
   ├─ 2. Shuttle tracking           → TrackNetV3            → ball trajectory (Frame,Vis,X,Y)
   ├─ 3. Hit-frame segmentation     → TrackNetV3/event_detection.py → per-stroke clip cuts
   ├─ 4. Court detection + homography → monotrack C++       → camera → court coordinates
   ├─ 5. Player detection + pose    → MMPose                → skeletons (2 players, COCO-17)
   │        ▼
   │   [ normalize: joints in bbox · shuttle by resolution · position by court ]
   │        ▼
   ├─ 6. STROKE CLASSIFICATION      → BST                   → stroke type per clip
   │        ▼
   ├─ 7. Structured rally data: (player, stroke_type, court_position, timestamp, outcome)
   │        ▼
   └─ 8. DATA MINING / TACTICS      ← the goal (self-implemented)
```

Note: BST's README treats stages 1–5 as separate upstream models you must supply; only the
stroke classifier (stage 6) is the paper's own contribution. For the bundled ShuttleSet data,
all perception stages were already run by the authors and baked into `.npy` (see §2).

---

## 2. Data Available (`project/dataset/`)

**~33 GB total.** Only **ShuttleSet** (badminton) is kept — BadmintonDB / TenniSet were dropped.

### 2.1 Pre-extracted skeleton/shuttle/position features (`.npy`)
The heavy CV work (pose, shuttle, court) is **already done** and stored as `.npy`. Per clip =
3 files: `<id>_joints.npy` `(t,2,17,2)`, `<id>_pos.npy` `(t,2,2)`, `<id>_shuttle.npy` `(t,2)`.
Already normalized. Organized as `variant/<split>/<class>/<clip>`.

| Variant | seq_len | classes | clips | Use |
|---|---|---|---|---|
| `seq100_between_2_hits_with_max_limits` | 100 | 35 | 33,481 | **primary (fine-grained)** |
| `merged_seq100_between_2_hits_with_max_limits` | 100 | 25 | 33,481 | **primary (merged) — recommended** |
| `seq30` | 30 | 35 | — | single-hit window |
| `merged_seq30` | 30 | 25 | — | single-hit window |
| `seq30_3d`, `merged_seq30_3d` | 30 | 35/25 | — | 3D pose (buggy per author, avoid) |

Splits: train / val / test (≈ 25,741 / 4,241 / 3,499 for seq100).

> ⚠️ These are **non-collated** (per-clip files, read by `Dataset_npy`). BST training/inference
> scripts (`bst_main.py`, `bst_infer.py`) use **`Dataset_npy_collated`**, which needs big packed
> files (`JnB_bone.npy`, `pos.npy`, `shuttle.npy`, `videos_len.npy`, `labels.npy`) in one folder.
> **One collation step required** (last step in `prepare_train_on_shuttleset.py`). CPU-only, no video.

### 2.2 Raw videos (`project/dataset/ShuttleSet_raw_videos/`)
- **44 `.mp4` matches**, ~27 GB, plus `.info.json` per video.
- These are the original broadcast singles used to build ShuttleSet (pro players, BWF events).
- Naming: `<ID> - <match_name>.mp4`.

### 2.3 ShuttleSet annotations (`project/dataset/ShuttleSet/set/`)
- **`match.csv`** — 44 matches: `id, video, tournament, round, year, month, day, winner, loser, downcourt, url`. Useful **metadata join key** for data mining.
- **`homography.csv`** — pre-computed **homography matrix + 4 court corners per match** (45 rows). → court detection ALREADY solved for these 44 videos.
- **`<match>/set1.csv … set3.csv`** — per-stroke annotations. Each row = one stroke with:
  `rally, ball_round, frame_num, roundscore_A/B, player, server, type, hit_x/y, landing_x/y, lose_reason, win_reason, getpoint_player, player_location_x/y, opponent_location_x/y …`
  → **rally + hit segmentation ALREADY solved** for these 44 videos (`frame_num` = hit frame). Also a
  rich source of ground-truth tactical data on its own.

---

## 3. Weights Available

### 3.1 BST (stroke classification) — `project/weights/on_ShuttleSet/` (51 files)
- Trained **on ShuttleSet only**.
- Variants present: `bst_0` (backbone), `bst_CG_AP` (best), plus `blockgcn` baselines; merged & non-merged; 2D & 3D; `between_2_hits_seq_100` & `seq30`.
- **Key file (recommended):**
  `bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt` (7.2 MB)
  → matches `bst_infer.py` defaults (model `BST_CG_AP`, `pose_style=JnB_bone`, seq_len=100, 25 merged classes).
- ❌ No BadmintonDB / TenniSet weights (datasets dropped).

### 3.2 TrackNetV3 (shuttle tracking) — `external_repos/TrackNetV3/`
- **`exp/model_best.pt`** (174 MB) — present & in the path BST expects (`exp/model_best.pt`). ✅ ready.
- Architecture is actually TrackNetV2 + CBAM attention (= BST's "TrackNetV3 (using attention)"). 90.53% acc on small badminton set.
- Also provides **`event_detection.py`** — signal-based hit-frame detection on the trajectory (no extra weights).

### 3.3 sacnn.pt (rally trimming) — `external_repos/Automated-Hit-frame-Detection-.../src/models/weights/`
- ✅ **`sacnn.pt`** — Shot-Angle CNN. Reads each frame, decides if it's the standard overhead
  court camera → **trims which segments of a full broadcast are real rallies** (vs replays,
  crowd shots, score screens). This is the **only** asset kept from that repo; it fills a gap
  nothing else covers (TrackNetV3's `event_detection` assumes it's already given a rally clip).
- ⚠️ In the original repo, `sacnn` is wrapped in `video_resolver.py`, which also instantiates a
  `RallyProcessor` needing court/player kpRCNN + the `opt` transformer. To use sacnn standalone,
  **extract just the `SACNNContainer` + `ShotAngleQueue` logic** (drop the RallyProcessor wiring).
- ❌ All other weights from that repo (`court_kpRCNN.pth`, `kpRCNN.pth`, `opt.pt`, `scaler.pickle`)
  are **dropped** — redundant with TrackNetV3 (hit detection), MMPose (pose), monotrack (court).

### 3.4 MMPose (pose estimation)
- No local `.pth`; `MMPoseInferencer('human')` **auto-downloads** COCO weights on first run. ✅ effectively ready (needs install).

---

## 4. External Repos (`external_repos/`)

| Repo | Role in pipeline | Status |
|---|---|---|
| `BST-Badminton-Stroke-type-Transformer` | **Stage 6** stroke classifier + all data-prep scripts | Code ✅, weights ✅ (§3.1) |
| `TrackNetV3` | **Stages 2+3** shuttle tracking + `event_detection.py` (hit frames) | ✅ ready (weight in `exp/`) |
| `monotrack` | **Stage 4** court detection (C++) + 2D/3D trajectory analysis notebooks | ⚠️ C++ must be compiled |
| `mmpose` | **Stage 5** pose estimation | ✅ auto-downloads weights |
| `Automated-Hit-frame-Detection-...` | **Stage 1** rally trimming — keep `sacnn.pt` only (§3.3) | ⚠️ extract sacnn standalone |

### Key BST data-prep scripts (in `BST.../stroke_classification/preparing_data/`)
- `prepare_train_on_shuttleset.py` — the master preprocessing script. Contains:
  - `detect_shuttlecock_by_TrackNetV3_with_attension()` → subprocess-calls `TrackNetV3/predict.py`.
  - `MMPoseInferencer('human')` for pose; `detect_players_2d()` picks the 2 in-court players.
  - `get_court_info()` reads `homography.csv`; `normalize_shuttlecock / normalize_joints / normalize_position`.
  - Final **collation** step → packed `.npy`.
- `ShuttleSet/gen_my_dataset.py` — cuts raw video into per-stroke clips using `frame_num` from `set*.csv` (rally segmentation for ShuttleSet = pure ffmpeg/moviepy cut, NOT an AI model).

### Key inference entry
- `BST.../stroke_classification/main_on_shuttleset/bst_infer.py` — load weight → `infer()` → class IDs → `get_merged_stroke_types()` names.

### monotrack court detector — build feasibility
- `external_repos/monotrack/court-detection` — C++, CMake. `build.sh` → `mkdir build && cmake .. && make`.
- **Toolchain ready:** gcc 11.4, cmake 3.22, ffmpeg 4.4.2, 64 cores / 692 GB RAM (OpenCV-from-source builds fast).
- **Risk:** CMake `FetchContent`-builds **OpenCV 3.4.12 from source** (git clone). OpenCV 3.4.x predates
  gcc 11 → may need small patches (`-fpermissive`) or gcc-10. Code uses OpenCV-3-only modules
  (`opencv_superres`, `opencv_shape`, `CV_CAP_PROP_*`) → cannot swap to system OpenCV 4.
- **Input:** `.avi` (README); convert `.mp4` with ffmpeg if `VideoCapture` doesn't read it directly.
- **Output:** 6 court points (4 corners + 2 net poles) → compute homography yourself (`cv2.findHomography`).

---

## 5. Status Per Stage — for a NEW, UNSEEN video

| Stage | Tool | Ready for unseen video? | Action needed |
|---|---|---|---|
| 1. Rally trimming | `sacnn.pt` | ⚠️ Extract standalone | Pull `SACNNContainer` out of `video_resolver.py` |
| 2. Shuttle tracking | `TrackNetV3/predict.py` | ✅ Yes | Run `predict.py` (+ `denoise.py`) |
| 3. Hit-frame seg | `TrackNetV3/event_detection.py` | ✅ Yes (signal-based, no weights) | Run on denoised trajectory → cut clips |
| 4. Court detection | `monotrack/court-detection` (C++) | ⚠️ Build needed | `cmake .. && make`, then `detect video.avi out.txt` → homography |
| 5. Pose estimation | `mmpose` | ✅ Yes | `pip install mmpose mmdet`; auto-downloads weights |
| 6. Stroke classification | BST `bst_CG_AP` | ✅ Yes | Use key weight (§3.1); inputs must be normalized identically |
| 7. Structured data | custom code | ➖ to build | Assemble per-stroke records |
| 8. Data mining | custom code | ➖ to build | Tactical analysis (see §7) |

**Bottom line for unseen video:** Stages 2, 3, 5, 6 are runnable now. Remaining work:
**Stage 4 (court)** = compile monotrack C++; **Stage 1 (rally trim)** = extract sacnn standalone.
Stages 7–8 are yours to build.

> For the **bundled 44 ShuttleSet videos**, all perception stages are pre-solved (homography.csv +
> set*.csv + pre-extracted .npy) — you are effectively already at stage 6/7.

---

## 6. Critical Normalization Contract (do not skip)
BST inputs must be normalized exactly as in `prepare_train_on_shuttleset.py`, or inference silently degrades:
- **`normalize_shuttlecock`** — shuttle / video resolution → [0,1].
- **`normalize_joints`** — joints relative to player bbox top-left, divided by bbox diagonal, `center_align=True` → ~[-0.x, 0.x].
- **`normalize_position`** — feet → court coords via homography, midpoint, / court rectangle → [0,1].
- COCO 17 keypoints; bones via `get_bone_pairs()` (19 pairs); `pose_style=JnB_bone` → in_dim = (17 + 19)·2.

---

## 7. Stage 8 — Data Mining / Tactics (to build)
Once each shot is `(match, rally, player, stroke_type, court_pos, timestamp, outcome)`:
- **Sequential pattern mining** of stroke sequences (winning patterns, serve→3rd-shot combos).
- **Markov / transition models** of rally states.
- **Player / opponent profiling** (stroke distribution, court heatmaps, tendencies under score pressure).
- **Win/loss attribution** (which patterns lead to points — `win_reason` / `lose_reason` / `getpoint_player` already in set*.csv for ground-truth validation).
- Join predictions with `match.csv` (tournament, round, winner) for contextual analysis.

ShuttleSet's `set*.csv` is both a **ground-truth check** for the AI pipeline and a ready-made
tactical dataset to prototype stage 8 before the full perception pipeline is wired.

---

## 8. Recommended Path

**Track A — Fastest to data mining (use existing extracted data):**
1. Collate the `merged_seq100_between_2_hits` `.npy` (one CPU step).
2. Run BST (`bst_CG_AP` merged weight) → stroke predictions.
3. Build stages 7–8 on the structured output. *(No video/perception work.)*

**Track B — True "unseen video" pipeline (the ultimate goal):**
1. Pick a new match video.
2. Rally trimming with sacnn (extract standalone) (stage 1).
3. TrackNetV3 `predict.py` → `denoise.py` → `event_detection.py` (stages 2–3).
4. Compile monotrack → court homography (stage 4).
5. MMPose for skeletons (stage 5).
6. Normalize per §6 → BST (stage 6).
7. Stages 7–8 as in Track A.

**Immediate gaps to close for Track B:** (a) compile monotrack C++ (stage 4); (b) extract sacnn
standalone (stage 1); (c) glue scripts in `1_…`–`4_…` stage folders (currently empty);
(d) collation/normalization wiring for arbitrary video. Everything else is in place.

---

## 9. Quick Reference — what exists vs missing
- ✅ ShuttleSet `.npy` (6 variants, 33k clips), 44 raw videos, set CSVs + homography
- ✅ BST weights (ShuttleSet, incl. best `bst_CG_AP` merged)
- ✅ TrackNetV3 weight (`exp/model_best.pt`) + `event_detection.py` (hit frames)
- ✅ MMPose (auto-download)
- ✅ `sacnn.pt` (rally trimming — extract standalone)
- ⚠️ monotrack court detector = C++ source, must compile (OpenCV 3.4.12 from source)
- ➖ Stage folders `1_…`–`4_…` empty (glue code to write); stages 7–8 to build
- ❌ Dropped: `court_kpRCNN.pth`, `kpRCNN.pth`, `opt.pt`, `scaler.pickle`, A-New-Perspective repo,
  BadmintonDB / TenniSet data & weights
