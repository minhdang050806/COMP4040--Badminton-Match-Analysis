# COMP4040 — Badminton Match Analysis

End-to-end system that ingests raw badminton match video and produces
per-stroke structured records plus tactical analytics. The codebase is
a refactored integration of three open-source repositories:

* **Automated-Hit-frame-Detection-for-Badminton-Match-Analysis** — SACNN
  rally segmentation, court & human keypoint R-CNN, OptimusPrime
  hit-frame transformer.
* **A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection** —
  TrackNetV2 shuttle tracking, ViT-B_16 attribute ensemble, YOLO landing
  prediction.
* **BST-Badminton-Stroke-type-Transformer** — BST stroke classifier
  (TCN + cross-attention + interactional encoder).

The original repos shared no code; this project unifies them behind one
configuration file and one set of stage contracts so the full pipeline
runs from `python run_pipeline.py`.

---

## 1. Architecture

The pipeline matches `Video/system_architecture.md` exactly: five stages
with explicit data contracts between them.

```
RAW MP4
   │
[Stage 1] Ingestion & Rally Segmentation       (SACNN + ShotAngleQueue)
   │  → RallySegment[]   (clip_path, start/end frames)
[Stage 2] Vision Pipeline                       (Court R-CNN + Human R-CNN + TrackNetV2)
   │  → RallySegment[]   (+ joints_path, shuttle_csv_path, court_corners)
[Stage 3] Hit-Frame Detection                   (OptimusPrime + trajectory cross-check)
   │  → HitEvent[]       (rally_id, shot_seq, hit_frame, joints_at_hit)
[Stage 4] Stroke Classification                 (Feature Extractor → BST → ViT Ensemble)
   │  → StrokeRecord[]   (CSV with class + ViT attributes + court positions)
[Stage 5] Analytics & Insights                  (clustering, pattern mining, tactical)
   ↓
strokes.csv  +  player_clusters.csv  +  frequent_patterns.json  +  heatmaps.npy
```

Stage data contracts live in [common/contracts.py](common/contracts.py).

---

## 2. Directory Layout

```
COMP4040--Badminton-Match-Analysis/
├── README.md
├── requirements.txt
├── run_pipeline.py                ← end-to-end orchestrator
├── configs/
│   └── pipeline.yaml              ← unified config (paths, weights, hparams)
├── common/                        ← shared contracts & helpers
│   ├── contracts.py               (RallySegment, HitEvent, StrokeRecord)
│   ├── config.py                  (YAML loader)
│   └── io.py                      (json/csv/jsonl helpers)
├── data_storage/                  ← gitignored data root
│   ├── raw_videos/                (input MP4s)
│   ├── intermediate/              (per-stage artefacts)
│   ├── outputs/                   (strokes.csv + analytics)
│   └── weights/                   (model checkpoints)
├── data_preparation/              ← preprocessing entry points
│   ├── unzip_kseq.py              (extract AICUP KSeq archives)
│   ├── prepare_optimusprime_dataset.py
│   └── prepare_bst_dataset.py     (dispatches to legacy preparing_data/*)
├── stage_1_ingestion_rally_segmentation/
│   ├── run.py                     ← stage entry point
│   ├── segmentation_module.py     (SegmentationModule + ShotAngleQueue)
│   ├── models/sacnn.py            (legacy)
│   └── utils/utils.py             (legacy)
├── stage_2_vision_pipeline/
│   ├── run.py
│   ├── vision_module.py
│   ├── court/rally_processor.py   (legacy court+human keypoint R-CNN)
│   └── shuttle_tracking/
│       ├── tracker.py             (Pythonic wrapper around predict10.py)
│       └── tracknetv2/            (legacy TrackNetV2 source)
├── stage_3_hit_frame_detection/
│   ├── run.py
│   ├── hit_detector.py
│   ├── optimusprime/transformer.py (legacy)
│   └── event_detection/event_detection_custom.py (legacy fallback)
├── stage_4_stroke_classification/
│   ├── run.py
│   ├── stroke_pipeline.py         (orchestrates feature extraction → BST → ViT)
│   ├── feature_extraction/
│   │   ├── extractor.py           (FeatureExtractor — Section 3.1.2 Step 2)
│   │   └── preparing_data/        (legacy training-data prep scripts)
│   ├── bst/
│   │   ├── classifier.py          (StrokeClassifier wrapper)
│   │   ├── model/bst.py           (legacy BST_CG_AP)
│   │   └── main_on_*/             (legacy training scripts, kept verbatim)
│   └── vit_ensemble/
│       ├── vit_pipeline.py        (ViTPipeline adapter)
│       ├── get_hitframe.py        (legacy)
│       ├── vit_<attr>.py          (legacy per-attribute submit scripts)
│       └── landing_*.py           (legacy YOLO landing prediction)
├── stage_5_analytics/
│   ├── run.py
│   ├── feature_engineering.py     (Section 4.1)
│   ├── clustering/player_clustering.py     (Section 4.2)
│   ├── pattern_mining/sequence_mining.py   (Section 4.3)
│   └── tactical/
│       ├── offensive_score.py     (Section 4.4.1)
│       ├── hmm_phases.py          (Section 4.4.2)
│       └── heatmaps.py            (Sections 4.4.3 & 4.4.4)
└── scripts/
    └── run_all.sh                 (convenience driver)
```

---

## 3. Setup

```bash
# clone & cd
cd COMP4040--Badminton-Match-Analysis

# Python ≥ 3.9 + CUDA-capable PyTorch recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place model checkpoints under `data_storage/weights/` (see
[data_storage/README.md](data_storage/README.md) for the file list and
sources). Reference paths in `configs/pipeline.yaml` if you store them
elsewhere.

---

## 4. Running the Pipeline

### 4.1 Full end-to-end run

```bash
# Drop your match MP4(s) into data_storage/raw_videos/
python run_pipeline.py
```

`run_pipeline.py` runs all five stages, persisting intermediates to
`data_storage/intermediate/` and writing the final CSV +
analytics artefacts to `data_storage/outputs/`.

### 4.2 Run a single stage

```bash
python -m stage_1_ingestion_rally_segmentation.run
python -m stage_2_vision_pipeline.run
python -m stage_3_hit_frame_detection.run
python -m stage_4_stroke_classification.run
python -m stage_5_analytics.run
```

### 4.3 Resume / re-run subsets

```bash
# Skip Stage 1 and 2 (load their outputs from disk) — useful for iterating on analytics
python run_pipeline.py --skip 1 2

# Only re-run analytics
python run_pipeline.py --only 5
```

### 4.4 Overriding the config

```bash
python run_pipeline.py --config configs/pipeline.yaml --video /data/match.mp4
```

---

## 5. Dataset Format

The system supports two dataset shapes:

### 5.1 Raw match video (production)

A 1280×720 MP4 dropped into `data_storage/raw_videos/`. Stage 1 will
segment it into per-rally MP4 clips. No annotations needed.

### 5.2 AICUP / KSeq labelled dataset (training)

The original challenge release lives in `Video/Dataset/`:

```
Dataset/
├── KSeq_train_data.zip         (57 MB — labelled rally clips + score_*.json)
├── KSeq_test_dataset.zip       (18 MB — unlabelled test clips)
└── files-archive               (extra reference material)
```

Each rally is stored as `score_<n>.json`:

```json
{
  "frames": [
    { "frame": 5926, "joint": [[x, y], …] },
    …
  ]
}
```

To prepare it for training the OptimusPrime / BST models:

```bash
# 1. unpack zips into data_storage/raw_videos/
python data_preparation/unzip_kseq.py

# 2. (Optional) fit a fresh StandardScaler for OptimusPrime
python data_preparation/prepare_optimusprime_dataset.py \
    --kseq_dir data_storage/raw_videos/KSeq_train_data \
    --scaler_out data_storage/weights/scaler.pickle

# 3. Build BST .npy tensors (legacy script)
python data_preparation/prepare_bst_dataset.py --dataset shuttleset
```

---

## 6. Stage Contracts (Cheat Sheet)

| Stage | Produces | Persisted at |
|---|---|---|
| 1 | `RallySegment[]` | `intermediate/rallies/<video>/rallies.json` + `…/clips/*.mp4` |
| 2 | enriches `RallySegment` (joints, shuttle, court) | `intermediate/joints/`, `intermediate/shuttle/` |
| 3 | `HitEvent[]` | `intermediate/hit_events/hit_events.json` |
| 4 | `StrokeRecord[]` | `outputs/strokes_csv/strokes.csv` |
| 5 | analytics artefacts | `outputs/analytics/*.csv` `*.json` `*.npy` |

Definitions live in [common/contracts.py](common/contracts.py).

---

## 7. Extension Guide

* **Plug in a different stroke classifier.** Replace
  `stage_4_stroke_classification/bst/classifier.py` with a new
  `StrokeClassifier` exposing `predict(features) -> (cls, probs, name)`.
* **Plug in a different shuttle tracker.** Re-implement
  `stage_2_vision_pipeline/shuttle_tracking/tracker.track_to_csv` to
  emit `(frame, visibility, x, y)` rows.
* **Add a new analytic.** Drop a module in `stage_5_analytics/<topic>/`
  and import it from `stage_5_analytics/__init__.py` + call it in
  `stage_5_analytics/run.py`.
* **Switch from CSV to a database.** `StrokePipeline.export_csv` is the
  only writer of `outputs/strokes_csv/strokes.csv`. Replace it with a DB
  upsert and update `stage_5_analytics/run.py` to read from the DB.
* **Run on a new dataset taxonomy** (BadmintonDB-18, TenniSet-6).
  Update `bst.n_classes` in `pipeline.yaml` and call
  `StrokeClassifier.set_class_names(...)` after construction.
* **Real-time / streaming.** The current pipeline is offline (per
  Section 5.1 of the architecture doc). To stream, replace
  `SegmentationModule.process` with a producer-consumer queue and
  switch Human R-CNN to YOLOv8-pose; the data contracts stay the same.

---

## 8. Citations

If you use this code, please cite the upstream projects whose work is
integrated here:

* Automated Hit-frame Detection — see `stage_1_…/CITATION.cff` (legacy).
* TrackNetV2 — Sun et al., 2020.
* BST — Badminton Stroke-type Transformer (project README).
* AICUP 2023 KSeq Dataset.
