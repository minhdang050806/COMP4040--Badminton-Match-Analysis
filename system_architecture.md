# Badminton Match Analysis — System Architecture

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Current Capabilities](#2-current-capabilities)
3. [Integration Plan](#3-integration-plan)
4. [Data Mining & Analytics](#4-data-mining--analytics)
5. [Future Extensions](#5-future-extensions)

---

## 1. High-Level Architecture

### 1.1 End-to-End Pipeline

```
RAW BADMINTON MATCH VIDEO (1280×720 MP4)
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — INGESTION & RALLY SEGMENTATION                         │
│                                                                    │
│  ┌─────────────────────────────────┐                              │
│  │  SACNN (Shot-Angle CNN)         │                              │
│  │  Every N-th frame (10% sample) │                              │
│  │  Input : (3, 216, 216) image    │                              │
│  │  Output: shot_angle ∈ {0, 1}   │                              │
│  └────────────┬────────────────────┘                              │
│               │                                                    │
│  ┌────────────▼────────────────────┐                              │
│  │  ShotAngleQueue (window=5)      │                              │
│  │  Detects 0→1 / 1→0 transitions │                              │
│  │  Output: rally [start, end]     │                              │
│  └────────────┬────────────────────┘                              │
└───────────────┼────────────────────────────────────────────────────┘
                │ rally boundary frames
                ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — VISION PIPELINE (per rally)                            │
│                                                                    │
│  ┌─────────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │ Court Keypoint  │   │  Human Keypoint  │   │  TrackNetV2  │  │
│  │ R-CNN           │   │  R-CNN           │   │  (10 frames) │  │
│  │ 6 court corners │   │  17 joints/person│   │  Shuttle x,y │  │
│  └────────┬────────┘   └────────┬─────────┘   └──────┬────────┘  │
│           │                     │                     │           │
│  ┌────────▼─────────────────────▼─────────────────────▼─────────┐ │
│  │  Court Geometry + Player Filtering + Trajectory Denoising    │ │
│  │  ─ Homography: camera → court coordinates                    │ │
│  │  ─ Bilateral filtering on shuttle path                       │ │
│  │  ─ Top/bottom player assignment by y-coordinate             │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ joints (T,2,17,2) + trajectory (T,2)
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — HIT-FRAME DETECTION                                    │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  OptimusPrime Transformer                                   │  │
│  │  Input : (T, 2, 17, 2) joint sequence (StandardScaler)     │  │
│  │  Output: direction ∈ {0,1,2,3} per frame                   │  │
│  │  Hit frame ← direction transition (0→1, 0→2, 1→2, 2→1)    │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│  ┌──────────────────────────▼──────────────────────────────────┐  │
│  │  Trajectory Event Detection (alternative / cross-check)     │  │
│  │  Peak detection on shuttle Y; angle change > 5°             │  │
│  │  Point-to-line distance > 2.5 px                            │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ hit_frame indices
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — STROKE CLASSIFICATION                                  │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  Preprocessing: .npy feature extraction per stroke clip   │    │
│  │  ─ Joints normalised within bounding box                  │    │
│  │  ─ Court position (homography → [0,1])                    │    │
│  │  ─ Shuttle normalised by frame resolution                 │    │
│  └──────────────────────┬────────────────────────────────────┘    │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐    │
│  │  BST (Badminton Stroke-type Transformer)                  │    │
│  │  ─ TCN encoder (pose + shuttle independently)             │    │
│  │  ─ Cross-modality attention (pose ← shuttle)              │    │
│  │  ─ Interactional encoder (player 1 ↔ player 2)           │    │
│  │  ─ MLP head → stroke class logits                        │    │
│  │  Output: 25 classes (ShuttleSet) / 18 (BadmintonDB)      │    │
│  └──────────────────────┬────────────────────────────────────┘    │
│                         │                                          │
│  ┌──────────────────────▼────────────────────────────────────┐    │
│  │  ViT Ensemble (5-fold) — per hit-frame image              │    │
│  │  ─ ViT_Hitter     : which player hit                      │    │
│  │  ─ ViT_Backhand   : forehand vs backhand                  │    │
│  │  ─ ViT_BallHeight : high / mid / low                      │    │
│  │  ─ ViT_BallType   : multi-class stroke type               │    │
│  │  ─ ViT_Winner     : rally outcome                         │    │
│  └──────────────────────┬────────────────────────────────────┘    │
└─────────────────────────┼──────────────────────────────────────────┘
                          │ per-stroke structured records
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — ANALYTICS & INSIGHTS                                   │
│                                                                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐   │
│  │ Player       │  │ Pattern       │  │ Tactical             │   │
│  │ Clustering   │  │ Mining        │  │ Profiling            │   │
│  │ (play style) │  │ (seq patterns)│  │ (offense/defense)    │   │
│  └──────────────┘  └───────────────┘  └──────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              OUTPUTS: dashboards, reports, CSV
```

### 1.2 Modular Breakdown

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| Ingestion | SACNN + ShotAngleQueue | Raw video | Rally [start, end] |
| Vision | Court/Human RCNN + TrackNetV2 | Rally frames | Joints (T,2,17,2), trajectory (T,2) |
| Hit detection | OptimusPrime + Event Detector | Joints + trajectory | Hit-frame indices |
| Feature extraction | MMPose + TrackNetV3 + Court Geo | Rally clips | .npy tensors |
| Stroke classification | BST | .npy tensors | Stroke class + probs |
| Attribute classification | ViT Ensemble + YOLOv8 | Hit-frame images | Hitter, BallType, Winner, Locations |
| Analytics | DM algorithms | Structured per-stroke CSV | Clusters, patterns, tactical insights |

### 1.3 Data Flow Between Components

```
Raw video
  │
  ├─[frames]──► SACNN ──► ShotAngleQueue ──► rally_segments[]
  │
  └─[rally frames]──► RallyProcessor
                          ├─[frames]──► Court Keypoint R-CNN ──► court_corners (6,2)
                          ├─[frames]──► Human Keypoint R-CNN ──► raw_joints (T,2,17,2)
                          │                 └─[filtered by court geo]
                          └─[joints]──► OptimusPrime ──► directions (T,) ──► hit_frames[]

  rally_clips[]
      │
      ├─[clips]──► TrackNetV2 ──► shuttle_trajectory (T,2)
      │                └─[trajectory]──► Event Detector ──► hit_frames[] (cross-check)
      │
      ├─[clips]──► MMPose ──► skeleton_sequence (T,17,2) per player
      │
      ├─[skeleton + shuttle + court]──► Normalisation ──► .npy tensors
      │                                     └──► BST ──► stroke_class, prob
      │
      └─[hit_frames]──► get_hitframe.py ──► cropped_images (720,720)
                             └──► ViT Ensemble ──► Hitter, Backhand, BallHeight, BallType, Winner
                             └──► YOLOv8 Pose ──► HitterXY, DefenderXY

  All above →
      Structured record per stroke:
      {rally_id, shot_seq, hit_frame, stroke_class, hitter, backhand, ball_height,
       ball_type, winner, hitter_xy, defender_xy, landing_xy, shuttle_trajectory}
          └──► CSV / DB ──► Data Mining & Analytics
```

---

## 2. Current Capabilities

### 2.1 What the Existing Code Already Supports

#### 2.1.1 BST — Badminton Stroke-type Transformer
`BST-Badminton-Stroke-type-Transformer/`

| Capability | Detail |
|---|---|
| Stroke classification | 25 classes (ShuttleSet), 18 (BadmintonDB), 6 (TenniSet) |
| Skeleton encoding | TCN on 17-joint COCO skeleton; supports J_only, JnB_bone, JnB_interp |
| Shuttle awareness | Cross-modality attention (pose queries attend to shuttle trajectory) |
| Player interaction | Interactional Transformer encoder fusing both players' representations |
| Variable-length handling | Attention masking via `video_len` |
| Data augmentation | `RandomTranslation` on normalised joints |
| Multi-dataset support | Distinct data loaders and training scripts per dataset |
| Baseline comparison | ST-GCN, BlockGCN, TemPose, ProtoGCN, SkateFormer |
| Evaluation | Per-class F1, confusion matrices, precision/recall |

**Where it fits:** Stage 4 (Stroke Classification), after skeleton and trajectory extraction.

**Limitations:**
- Requires pre-extracted `.npy` files — no end-to-end video input.
- 3D pose path broken (MMPose 3D API bug, disabled).
- Trained separately per dataset; no unified cross-dataset model.
- No real-time inference code; batch-only.

#### 2.1.2 Automated Hit-Frame Detection
`Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/`

| Capability | Detail |
|---|---|
| Rally segmentation | SACNN + ShotAngleQueue (window=5, threshold majority vote) |
| Court detection | Keypoint R-CNN (6 court corners + homography) |
| Court partitioning | 35-point grid for fine-grained position tracking |
| Player detection | Human Keypoint R-CNN filtered by court geometry |
| Player assignment | Top/bottom by y-coordinate |
| Shuttle direction prediction | OptimusPrime Transformer (4 tokens: none/to-top/to-bottom/transition) |
| Hit-frame detection | Direction transition detection (0→1, 0→2, 1→2, 2→1) |
| Rally export | MP4 clips + joints JSON + rally metadata JSON |

**Where it fits:** Stages 1–3 (Ingestion → Vision → Hit Detection).

**Limitations:**
- Runs rally processing only when rally ends; no true streaming/real-time.
- Player detection assumes exactly 2 players per court half.
- OptimusPrime rejects sequences with >60% zero predictions (may miss short rallies).
- No shuttlecock position tracking — direction only (no x,y coordinates).

#### 2.1.3 A New Perspective — Hitting Event Detection
`A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection/`

| Capability | Detail |
|---|---|
| Shuttlecock tracking | TrackNetV2 (10-frame grayscale stack, 288×512) |
| Trajectory denoising | Bilateral filtering, interpolation for missing frames |
| Hit-frame detection | Peak detection on Y + angle change > 5° |
| Stroke attribute classification | ViT-B_16 ensemble (5-fold): Hitter, Backhand, BallHeight, BallType, Winner |
| Player detection | YOLOv5 (conf=0.3, imgsz=2880, TTA augment) |
| Pose estimation | YOLOv8x-pose-p6 — 17 keypoints per player |
| Landing prediction | Separate regression heads for LandingX and LandingY |
| Output | Structured CSV: 15 columns per stroke event |

**Where it fits:** Stages 2–4 (Vision → Hit Detection → Stroke Attributes).

**Limitations:**
- Trajectory peak detection is fragile (heuristic thresholds, sensitive to noise).
- ViT models trained only on AICUP2023 competition data; limited generalisation.
- No court geometry integration (player positions are in pixel space, not court space).
- Landing prediction accuracy not explicitly benchmarked outside competition.

#### 2.1.4 Dataset
`Dataset/`

| Asset | Size | Content |
|---|---|---|
| KSeq_train_data.zip | 57 MB | Training video clips + annotations for AICUP/CodaLab |
| KSeq_test_dataset.zip | 18 MB | Test clips (unlabelled) |
| files-archive | 342 MB | Additional reference material |

### 2.2 Feature-to-Pipeline Map

```
Raw video ──────────────────────────────────── Automated Hit-frame Detection
Rally clips ────────────────────────────────── BST (after .npy prep)
                                           └── A New Perspective (hitting events)
Hit-frame images ───────────────────────────── ViT Ensemble
Player keypoints ───────────────────────────── BST, OptimusPrime
Shuttle trajectory ─────────────────────────── BST (cross-attention), Event Detector
Court geometry ─────────────────────────────── Court Keypoint R-CNN → position normalisation
Stroke labels ──────────────────────────────── BST output, ViT_BallType output
Player positions ───────────────────────────── YOLOv8 → HitterXY, DefenderXY
```

---

## 3. Integration Plan

### 3.1 Unified Pipeline Interface

The three projects currently share no code and have incompatible interfaces. The integration plan defines clear contracts between stages.

#### 3.1.1 Stage Interfaces (Data Contracts)

**Contract A — Rally Segment Record**
```python
@dataclass
class RallySegment:
    video_path: str
    rally_id: int
    start_frame: int
    end_frame: int
    clip_path: str           # exported MP4
    joints_path: str         # JSON: (T, 2, 17, 2)
    shuttle_csv_path: str    # TrackNetV2 output CSV
    court_corners: np.ndarray  # (6, 2) pixel coords
    homography: np.ndarray     # (3, 3) camera→court
```

**Contract B — Hit Event Record**
```python
@dataclass
class HitEvent:
    rally_id: int
    shot_seq: int            # 1-indexed within rally
    hit_frame: int           # absolute frame number
    direction_before: int    # OptimusPrime token
    direction_after: int
    joints_at_hit: np.ndarray   # (2, 17, 2) normalised
    shuttle_at_hit: np.ndarray  # (2,) normalised xy
    court_position: np.ndarray  # (2, 2) player court coords
```

**Contract C — Stroke Analysis Record**
```python
@dataclass
class StrokeRecord:
    # identity
    video_name: str
    rally_id: int
    shot_seq: int
    hit_frame: int
    # BST output
    stroke_class: int
    stroke_class_name: str
    stroke_probs: np.ndarray   # (n_classes,)
    # ViT outputs
    hitter: int                # 0 or 1
    backhand: int              # 0 or 1
    ball_height: int           # 0/1/2 (low/mid/high)
    ball_type: int
    winner: int
    # locations
    hitter_xy_court: np.ndarray    # (2,)
    defender_xy_court: np.ndarray  # (2,)
    landing_xy_court: np.ndarray   # (2,)
```

#### 3.1.2 Connecting the Modules

**Step 1: Adapt Automated Hit-Frame Detection to emit Contract A**

The existing `RallyProcessor` already outputs most fields into JSON files. Wrap it in a `SegmentationModule` class that:
- Reads `{video_name}.json` (rally start/end/count)
- Reads `rally_{n}.json` (joints, directions, hit_frames)
- Runs TrackNetV2 on each rally clip to produce shuttle CSV
- Runs Court Keypoint R-CNN to store homography per rally

**Step 2: Feature Extraction Adapter**

Build `FeatureExtractor` that takes Contract A and produces `.npy` tensors compatible with BST:

```python
class FeatureExtractor:
    def extract(self, rally: RallySegment) -> dict:
        # normalise joints within bounding box
        # transform court position via homography → [0,1]
        # normalise shuttle by frame resolution
        # collate to fixed seq_len (30/72/100) via stride+padding
        return {
            'human_pose': np.ndarray,   # (seq_len, 2, input_dim)
            'pos':        np.ndarray,   # (seq_len, 2, 2)
            'shuttle':    np.ndarray,   # (seq_len, 2)
            'video_len':  int,
        }
```

**Step 3: Stroke Classifier Wrapper**

Wrap BST inference into a stateless callable:

```python
class StrokeClassifier:
    def __init__(self, weights_path, n_classes=25, d_model=100):
        self.model = BST_CG_AP(in_dim=34, seq_len=30, n_class=n_classes, d_model=d_model)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def predict(self, features: dict) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            logits = self.model(...)
        return int(logits.argmax()), softmax(logits).numpy()
```

**Step 4: ViT Attribute Pipeline Adapter**

Wrap `get_hitframe.py` + ViT ensemble into a single callable that takes a `HitEvent` and outputs ViT predictions. Key change: add court-coordinate conversion for player positions (currently pixel-space only in the existing code).

**Step 5: Orchestrator**

```python
class BadmintonAnalysisPipeline:
    def __init__(self, config):
        self.segmenter    = SegmentationModule(config)
        self.extractor    = FeatureExtractor(config)
        self.classifier   = StrokeClassifier(config.bst_weights)
        self.vit_pipeline = ViTPipeline(config)

    def run(self, video_path: str) -> list[StrokeRecord]:
        rallies   = self.segmenter.process(video_path)
        records   = []
        for rally in rallies:
            features  = self.extractor.extract(rally)
            hit_events = self.segmenter.get_hit_events(rally)
            for event in hit_events:
                cls, probs = self.classifier.predict(features)
                attrs      = self.vit_pipeline.predict(rally, event)
                records.append(StrokeRecord(cls=cls, probs=probs, **attrs))
        return records
```

### 3.2 Required Refactoring

| Area | Current State | Required Change |
|---|---|---|
| `RallyProcessor` | Saves JSON files inline | Extract as `SegmentationModule`; decouple I/O |
| `prepare_train_on_shuttleset.py` | Monolithic script | Factor `FeatureExtractor` into reusable class |
| BST training scripts | Hardcoded paths | Accept config dict / YAML |
| ViT scripts | Notebook-style top-level code | Wrap in classes with `__call__` |
| TrackNetV2 | CLI only | Expose as Python API |
| Court homography | Computed ad hoc in `RallyProcessor` | Share `CourtGeometry` object across stages |

### 3.3 Configuration File (Proposed Unified YAML)

```yaml
video:
  source_dir: ./videos
  output_dir: ./outputs
  frame_sample_rate: 10  # process every 10th frame for SACNN

models:
  sacnn:       ./weights/sacnn.pt
  court_rcnn:  ./weights/court_kpRCNN.pth
  human_rcnn:  ./weights/kpRCNN.pth
  tracknet:    ./weights/tracknet.pt
  optimusprime: ./weights/opt.pt
  scaler:      ./weights/scaler.pickle
  bst:         ./weights/bst_cg_ap.pt
  vit_hitter:  ./weights/vit_hitter/
  vit_backhand: ./weights/vit_backhand/
  vit_ball_type: ./weights/vit_ball_type/
  yolov8_pose: ./weights/yolov8x-pose-p6.pt

bst:
  n_classes: 25
  seq_len: 30
  d_model: 100
  pose_style: JnB_bone

analytics:
  output_csv: ./outputs/strokes.csv
  clustering_n: 4
  min_sequence_len: 3
```

---

## 4. Data Mining & Analytics

The structured `StrokeRecord` CSV produced by the pipeline is the primary input to all data mining analyses. Each row is one stroke event with ~20 features.

### 4.1 Feature Engineering from Raw Model Outputs

Before applying DM algorithms, engineer a richer feature set from the raw model outputs:

#### 4.1.1 Per-Stroke Features

| Feature | Source | Type |
|---|---|---|
| `stroke_class` | BST output | Categorical (25) |
| `stroke_confidence` | `max(stroke_probs)` | Float [0,1] |
| `stroke_entropy` | `-sum(p*log(p))` on probs | Float |
| `is_backhand` | ViT_Backhand | Binary |
| `ball_height` | ViT_BallHeight | Ordinal {0,1,2} |
| `ball_type` | ViT_BallType | Categorical |
| `hitter_court_zone` | Quantise hitter_xy_court into 6×3 grid | Categorical (18) |
| `defender_court_zone` | Quantise defender_xy_court | Categorical (18) |
| `hitter_to_center_dist` | Euclidean distance from court center | Float |
| `defender_to_net_dist` | Distance from defender to net | Float |
| `landing_zone` | Quantise landing_xy_court into 6×3 grid | Categorical (18) |
| `shot_depth` | landing_xy[1] relative to net | Float |
| `shot_cross_court` | abs(landing_xy[0] - hitter_xy[0]) > threshold | Binary |
| `rally_shot_number` | `shot_seq` | Integer |
| `shuttle_speed_proxy` | Displacement between frames at hit | Float |

#### 4.1.2 Per-Rally Aggregated Features

| Feature | Computation |
|---|---|
| `rally_length` | Total shots in rally |
| `rally_duration_s` | (end_frame - start_frame) / fps |
| `p1_stroke_distribution` | Histogram over 25 stroke classes for player 1 |
| `p2_stroke_distribution` | Histogram for player 2 |
| `p1_backhand_rate` | Fraction of p1's shots that are backhand |
| `p1_avg_court_depth` | Mean y-coordinate of p1's hits |
| `rally_outcome` | `winner` from final shot |
| `rally_type` | Cluster label (from Section 4.3) |

#### 4.1.3 Sequence Features (for pattern mining)

Encode each shot as a symbol:
```
symbol = f"{stroke_class}_{hitter_court_zone}_{ball_height}"
```
This produces a string sequence per rally, e.g.:
```
rally_42: ["smash_FrontLeft_H", "lift_RearRight_H", "drop_FrontMid_L", ...]
```

---

### 4.2 Player Clustering — Play Style Grouping

**Goal:** Group players into archetypes (e.g., aggressive smasher, defensive retriever, net player, all-rounder) without manual labelling.

**Input features (one vector per player per match):**

```
f_player = [
  stroke_class_distribution,       # 25-dim histogram (L1-normalised)
  backhand_rate,                    # scalar
  avg_hitter_court_depth,           # scalar
  avg_landing_depth,                # scalar
  smash_rate,                       # scalar
  net_shot_rate,                    # scalar
  avg_shot_confidence,              # scalar
  win_rate_as_hitter,               # scalar
  rally_length_when_winner,         # scalar
]
# Total: ~34 dimensions
```

**Dimensionality reduction:** PCA to 10 components (retain ≥90% variance) or UMAP for visualisation to 2D.

**Algorithm:** K-Means (k=4 or determined by elbow + silhouette score)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = build_player_feature_matrix(stroke_records)   # (n_players, 34)
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=10).fit_transform(X_scaled)

# Elbow + silhouette sweep for k
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)

# Final model
km = KMeans(n_clusters=4, n_init=20, random_state=42)
player_clusters = km.fit_predict(X_pca)
```

**Expected cluster archetypes:**

| Cluster | Likely Profile | Key Features |
|---|---|---|
| 0 | Aggressive attacker | High smash_rate, front-court position, short rally_length_when_winner |
| 1 | Defensive retriever | High lift_rate, rear-court position, high rally_length |
| 2 | Net specialist | High net_shot_rate, short avg_landing_depth |
| 3 | All-rounder | Balanced stroke distribution, moderate everything |

**Output:** Player archetype label + cluster centroid feature importance (via PCA loading analysis).

---

### 4.3 Pattern Mining — Common Stroke Sequences

**Goal:** Discover frequent tactical patterns in stroke sequences (e.g., "smash → lift → drop → net-kill" is a common 4-shot pattern ending in a winner).

#### 4.3.1 Sequence Encoding

Encode each shot as a categorical symbol:
```
s_i = (stroke_class, hitter_court_zone, ball_height)
```
Each rally becomes a sequence of symbols `[s_1, s_2, ..., s_n]`.

#### 4.3.2 Frequent Sequence Mining

Use **PrefixSpan** (Prefix-Projected Sequential Pattern Mining) — efficient for variable-length sequences.

```python
from prefixspan import PrefixSpan   # pip install prefixspan

sequences = encode_rallies_as_sequences(stroke_records)   # list of lists
ps = PrefixSpan(sequences)
frequent_patterns = ps.frequent(min_support=0.05)   # 5% of rallies
# Returns: [(support, pattern), ...]
```

Alternatively, use **SPMF** (Sequential Pattern Mining Framework) for GSP or SPADE.

**Parameters:**
- `min_support`: 0.05 (tunable; lower for rare but tactically important patterns)
- `max_length`: 6 shots (limits pattern complexity)
- `gap_constraint`: max 2 shots between matched items (flexible matching)

**Discriminative patterns:** Apply **sequential rule mining** to find patterns with high confidence for winning outcomes:

```
IF [smash_FrontMid → lift_Rear → drop_Front] THEN Winner=1  (conf=0.78)
```

Use **RuleGrowth** algorithm (available in SPMF) for this.

**Output:** Top-k frequent patterns with support and confidence; tagged by outcome (winner/loser rally).

#### 4.3.3 Markov Chain Rally Model

Build a first-order Markov chain over stroke classes to model transition probabilities:

```python
# Build transition matrix (25×25)
T = np.zeros((n_classes, n_classes))
for rally in rallies:
    for i in range(len(rally.shots) - 1):
        T[rally.shots[i].stroke_class, rally.shots[i+1].stroke_class] += 1
T = T / T.sum(axis=1, keepdims=True)   # row-normalise

# Analyse: which strokes most likely follow a smash?
print(T[smash_idx].argsort()[-5:])   # top-5 transitions
```

This reveals the most likely response to each stroke — a tactical knowledge graph.

---

### 4.4 Tactical Analysis — Offensive vs Defensive Tendencies

**Goal:** Characterise each player's tactical style quantitatively and identify when they switch between modes.

#### 4.4.1 Offensive-Defensive Score

Define a scalar score per shot based on stroke type and court position:

```python
# Assign offensive weight per stroke class (domain knowledge)
offensive_weights = {
    'smash': +1.0,
    'push':  +0.7,
    'drop':  +0.5,
    'clear': -0.3,   # often defensive
    'lift':  -0.8,
    'lob':   -0.7,
    # ...
}

def offensive_score(shot: StrokeRecord) -> float:
    base = offensive_weights.get(shot.stroke_class_name, 0.0)
    # Bonus for net position
    base += 0.3 if shot.hitter_court_zone in FRONT_ZONES else 0.0
    # Penalty for rear-court defensive position
    base -= 0.2 if shot.hitter_court_zone in REAR_ZONES else 0.0
    return base
```

**Rally-level:** Smooth with a rolling window (window=3 shots) to produce an offensive momentum curve per rally. Peaks indicate attack phases; valleys indicate defensive retreating.

#### 4.4.2 Tactical Phase Segmentation (Hidden Markov Model)

Model each rally as a sequence of latent tactical states:

```
States: {ATTACK, NEUTRAL, DEFENCE}
Observations: offensive_score per shot (discretised into bins)
```

```python
from hmmlearn import hmm

scores = np.array([offensive_score(s) for s in rally_shots]).reshape(-1,1)
model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=100)
model.fit(scores)
tactical_states = model.predict(scores)
```

This segments each rally into phases, enabling analysis like:
- "Player A spends 65% of shots in ATTACK state vs Player B's 40%"
- "After entering DEFENCE state, player X returns to ATTACK in avg 2.3 shots"

#### 4.4.3 Court Heatmaps

Aggregate `hitter_xy_court` and `defender_xy_court` into 2D histograms (e.g., 30×15 grid matching badminton court dimensions):

```python
import scipy.ndimage

hitter_positions = np.array([(r.hitter_xy_court[0], r.hitter_xy_court[1])
                              for r in stroke_records if r.hitter == player_id])
heatmap, xedges, yedges = np.histogram2d(
    hitter_positions[:,0], hitter_positions[:,1],
    bins=[30, 15], range=[[0,1],[0,1]]
)
heatmap_smooth = scipy.ndimage.gaussian_filter(heatmap, sigma=1.5)
```

Overlaid on a court diagram: shows where a player tends to hit from and reveals positional tendencies (e.g., corner preference, T-zone dominance).

#### 4.4.4 Landing Zone Analysis

Cross-tabulate `hitter_court_zone × landing_zone` to reveal directional shot tendencies:

```python
import pandas as pd

df = pd.DataFrame([vars(r) for r in stroke_records])
landing_pivot = pd.crosstab(df['hitter_court_zone'], df['landing_zone'], normalize='index')
# landing_pivot[i,j] = P(land in zone j | hit from zone i)
```

This reveals: "When Player A hits from the left rear corner, 72% of shots land in the right front corner" (cross-court preference).

---

### 4.5 Suggested Algorithm Summary

| Analysis | Algorithm | Key Hyperparameters | Expected Output |
|---|---|---|---|
| Player clustering | K-Means + PCA | k=4, PCA components=10 | Archetype labels |
| Visualisation | UMAP | n_neighbors=15, min_dist=0.1 | 2D player map |
| Sequence patterns | PrefixSpan | min_support=0.05, max_len=6 | Frequent k-grams |
| Tactical rules | RuleGrowth | min_conf=0.6, min_sup=0.03 | Winning/losing patterns |
| Rally transitions | Markov Chain | order=1 | Transition matrix 25×25 |
| Tactical phases | HMM (Gaussian) | n_states=3 | Attack/Neutral/Defence labels |
| Position analysis | 2D Histogram | bins=30×15, Gaussian smoothing | Court heatmaps |
| Shot tendency | Cross-tabulation | Normalised row-wise | P(landing zone \| hitter zone) |

---

## 5. Future Extensions

### 5.1 Real-Time vs Offline Processing

**Current state:** All modules are offline (process full video, write to disk, next stage reads). Latency between raw video and final analytics is on the order of minutes per match.

**Path to real-time:**

| Bottleneck | Latency (est.) | Mitigation |
|---|---|---|
| SACNN (every 10th frame) | ~2ms/frame on GPU | Already fast; retain |
| TrackNetV2 (10-frame window) | ~10ms/window | Use TensorRT/ONNX export |
| Human Keypoint R-CNN | ~30ms/frame | Replace with YOLOv8-pose (faster) |
| OptimusPrime Transformer | ~5ms/frame | Quantise to FP16 |
| BST inference | ~3ms/stroke | Batch async |
| ViT Ensemble (5 models) | ~25ms/image | Distil to single ViT-S |

Real-time pipeline target: sub-200ms end-to-end per rally at 30 fps. Requires: TensorRT export of SACNN + TrackNetV2, async producer/consumer queue between stages, YOLOv8 replacing Keypoint R-CNN.

### 5.2 Scalability

**Multi-match processing:** The pipeline is stateless per video; parallelise with `ProcessPoolExecutor` or Celery workers.

**Database backend:** Replace CSV outputs with a relational DB (PostgreSQL):
- Table `matches(match_id, date, player_1, player_2)`
- Table `rallies(rally_id, match_id, start_frame, end_frame, winner)`
- Table `strokes(stroke_id, rally_id, shot_seq, all StrokeRecord fields)`

This enables cross-match analytics queries directly in SQL and scales to thousands of matches.

### 5.3 Model Improvements

**Shuttlecock tracking:** Replace TrackNetV2 with **TrackNetV3** (attention-based, already referenced in `prepare_train_on_shuttleset.py`) for more robust tracking under occlusion.

**Stroke classification:** Current BST uses 2D pose only (3D disabled). Lifting the 3D pose limitation by fixing the MMPose API call would add depth information and likely improve classification of strokes with similar 2D projections (e.g., smash vs drop from the same court position).

**Court detection:** The 6-point Court Keypoint R-CNN fails when the court is partially occluded (crowd, shadows). Replace with a line-segment detector + RANSAC homography, which is more robust to partial visibility.

**Multi-camera support:** Professional tournaments use multiple cameras. Add a view-selection module (classify camera angle — side/service-line/top) and train BST with view-conditioned embeddings.

### 5.4 Analytics Extensions

**Fatigue modelling:** Track how stroke distributions and court positions shift across game sets using a Bayesian change-point detection model. Identifies fatigue onset frames.

**Opponent exploitation:** Given player archetype labels and their transition matrices, compute the optimal counter-stroke sequence for each opponent type using a simple MDP (Markov Decision Process) with rewards proportional to win_rate.

**Automatic highlight generation:** Use `stroke_confidence` + `ball_type` (smash, net-kill) + rally outcome to score each shot for highlight worthiness; extract top-k clips automatically.

**Federated learning across clubs:** Club-specific player data is sensitive. Train a shared federation of BST models using FedAvg — clubs contribute gradients, not raw video — to build a global model without privacy risks.

---

## Appendix A — Key Model Specifications

### A.1 BST — Input/Output Specification

```
Inputs (all normalised):
  human_pose : (batch, seq_len=30, n_people=2, input_dim=34)
               input_dim = 17 joints × 2 coords (JnB_bone adds bones → 66)
  shuttle    : (batch, seq_len=30, 2)
  pos        : (batch, seq_len=30, n_people=2, 2)
  video_len  : (batch,)   — actual unpadded length

Internal dimensions:
  d_model = 100
  d_head  = 128 (cross-attention)
  n_head  = 6
  n_layers_temporal = 2
  n_layers_interact = 1
  TCN kernel_size = 5, dilation = [1, 3]

Output:
  logits : (batch, n_classes)
  For ShuttleSet: n_classes = 25
  For BadmintonDB: n_classes = 18
```

### A.2 OptimusPrime — Input/Output Specification

```
Inputs:
  joints : (batch, seq_len, n_people=2, n_joints=17, 2)
           pre-scaled with StandardScaler

Architecture:
  CoordinateEmbedding : (34,) → emb_size
  TransformerEncoder  : 8 layers, 8 heads, hidden=2048
  MLP Decoder         : emb_size → 4 tokens

Output:
  directions : (seq_len, batch, 4) — per-frame token probabilities
  Token mapping: 0=none, 1=to-top, 2=to-bottom, 3=transition
```

### A.3 SACNN — Input/Output Specification

```
Input:
  image : (batch, 3, 216, 216) — CenterCrop from (216, 384)
  Normalised with ImageNet μ/σ

Architecture:
  Conv2d(3→32, 3×3) → ReLU → MaxPool(2) → BatchNorm
  Conv2d(32→64, 3×3) → ReLU → MaxPool(2) → BatchNorm
  Conv2d(64→32, 3×3) → ReLU → MaxPool(2) → BatchNorm
  Linear(27×27×32 → 2) → ReLU → Dropout(0.1)

Output:
  logits : (batch, 2)   — rally / non-rally
```

### A.4 TrackNetV2 — Input/Output Specification

```
Input:
  frames : (batch, 10, 288, 512) — grayscale, divided by 255

Output:
  heatmap : (batch, 1, 288, 512)
  Threshold 0.5 → binary visibility map
  Extract centroid → (x, y) in 288×512 space → rescale to 1280×720
```

---

## Appendix B — Normalisation Reference

All normalisation must be applied consistently between training and inference.

| Feature | Formula | Range |
|---|---|---|
| Joint coordinates | `(joint - bbox_topleft) / bbox_diagonal` | ~[-0.5, 0.5] |
| Bone vectors | `joint_end - joint_start` (normalised joints) | ~[-1, 1] |
| Court position | `(pos_camera via H) → (x-border_L)/(border_R-border_L)` | [0, 1] |
| Shuttle position | `(x, y) / (video_width, video_height)` | [0, 1] |
| OptimusPrime joints | `sklearn.StandardScaler` (fit on training data) | ~N(0,1) |
| SACNN image | ImageNet mean/std per channel | ~N(0,1) |
| TrackNetV2 frames | divide by 255 | [0, 1] |
| ViT images | Resize to 480×480, ImageNet normalise | ~N(0,1) |

---

*Document generated from codebase analysis of `/home/dang.cpm/__MY_SPACE__/VinUni/Data-Mining/Video/`.*
