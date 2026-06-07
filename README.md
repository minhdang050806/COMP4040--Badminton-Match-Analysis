# COMP4040 — Badminton Match Analysis

**Course:** COMP4040 Data Mining and Big Data Analytics  
**Institution:** VinUniversity, College of Engineering and Computer Science

**Team:**
| Name | Student ID | Email |
|---|---|---|
| Cao Pham Minh Dang | V202401280 | 24dang.cpm@vinuni.edu.vn |
| Pham Dinh Hieu | V202401287 | 24hieu.pd@vinuni.edu.vn |
| Pham Minh Hieu | V202200842 | 22hieu.pm@vinuni.edu.vn |

---

## Project Overview

This project builds a badminton analytics system that converts broadcast video into an ordered stream of predicted strokes and then mines that stream for tactical patterns.

The pipeline combines:

- **SA-CNN** rally-view filtering to isolate valid rally frames
- **TrackNetV3** shuttle tracking and trajectory denoising
- **ShuttleSet** hit-frame labels for stroke window construction and evaluation
- **YOLO26x-pose** player detection, tracking, and pose extraction
- A fine-tuned **Badminton-Stroke Transformer** (`BST_CG_AP`) for stroke classification
- Transition, pattern, spatial, and clustering tactical analyses

### Key Results (Videos 41–44, unseen evaluation set)

| Metric | Result |
|---|---:|
| Side-aware top-1 accuracy | **69.04%** |
| Top-3 accuracy | **87.89%** |
| Player-side accuracy | **93.05%** |
| Stroke-type accuracy (ignoring side) | **69.19%** |
| Supported-class macro-F1 | **66.64%** |

This is a substantial improvement over an earlier integrated baseline (8.41% all-labeled accuracy). The pipeline reduces the predicted `none` rate from 56.43% to 6.70% and recovers recognizable attack-defense sequences.

---

## Repository Structure

```
COMP4040--Badminton-Match-Analysis/
├── tools/                          # Python scripts — one per pipeline phase
│   ├── phase00_01_inventory.py
│   ├── phase02_build_ground_truth_tables.py
│   ├── phase03_collate_bst_features.py
│   ├── phase04_train_bst.py
│   ├── phase04_run_bst_inference.py
│   ├── phase05_rally_filtering.py
│   ├── phase05_train_sacnn.py
│   ├── phase05_finetune_all44.sh
│   ├── phase05_build_sacnn_dataset.py
│   ├── phase06_shuttle_tracking.py
│   ├── phase07_hit_frame_detection.py
│   ├── phase07_build_ground_truth_windows.py
│   ├── phase07_tune_hit_frame_parameters.py
│   ├── phase09_pose_position_features.py
│   ├── phase09_compare_reference_features.py
│   ├── phase10_prepare_bst_inputs.py
│   ├── phase10_run_bst_inference.py
│   ├── phase10_translate_outputs_to_english.py
│   ├── phase11_tactical_mining.py
│   ├── analyze_movement_strategy.py
│   ├── bst_label_names.py
│   ├── visualize_journal_report.py
│   ├── visualize_phase05_phase06.py
│   ├── visualize_phase09_features.py
│   └── visualize_video44_showcase.py
│
├── __guidance__/                   # Project documentation and reports
│   ├── COMP4040_course_project_and_rubric.md
│   ├── REPORT.md                   # Main project report
│   ├── systems.md                  # Full pipeline architecture
│   ├── data.md                     # Dataset documentation
│   ├── data_mining.md              # Mining methodology
│   ├── interpretation.md           # Results interpretation
│   ├── source.md                   # External sources and repos
│   ├── final_report.tex            # LaTeX report draft
│   ├── reference.bib               # Bibliography
│   ├── images/                     # Figures used in the report
│   └── report/                     # Per-phase execution reports
│       ├── phase_00_setup.md
│       ├── phase_01_dataset_inventory.md
│       ├── phase_02_ground_truth_tables.md
│       ├── phase_03_bst_collation.md
│       ├── phase_04_bst_inference.md
│       ├── phase_05_rally_filtering.md
│       ├── phase_06_shuttle_tracking.md
│       ├── phase_07_hit_frame_detection.md
│       ├── phase_09_pose_position.md
│       ├── phase_10_integrated_inference.md
│       └── phase_11_tactical_mining.md
│
└── outputs/                        # Pipeline outputs (text/CSV/JSON/images only)
    ├── bst_collated/               # Collated BST feature sets + metadata CSVs
    ├── bst_training_all44/         # BST training history, confusion matrix, metrics
    ├── features_yolo26x_41_44/     # Phase 09 feature manifests and summaries
    ├── hit_frames/                  # Per-rally hit-frame event CSVs
    ├── integration/                 # Phase 10 integrated prediction tables
    ├── inventory/                   # Dataset inventory tables
    ├── journal_report_figures/      # Publication-quality figures
    ├── mining/                      # Phase 11 tactical mining results
    ├── movement_strategy/           # Movement and strategy cluster outputs
    ├── phase09_window_ablation/     # Window-size ablation study results
    ├── predictions/                 # Final stroke prediction tables
    ├── rallies/                     # Per-video rally segmentation outputs
    ├── sacnn_training_protocol/     # SA-CNN training logs
    ├── shuttle/                     # TrackNetV3 shuttle tracking outputs
    ├── tables/                      # Ground-truth tables (Phase 02)
    └── visualizations/              # QA contact sheets and visualizations
```

> **Not included in this repo** (too large or binary):
> - Raw ShuttleSet videos and image crops (`dataset/`)
> - Pre-trained and fine-tuned model weights (`weights/`, `*.pt`)
> - Intermediate numpy feature arrays (`*.npy`, `*.npz`)
> - Raw broadcast video files

---

## Pipeline Architecture

The project has two tracks:

**Track A — ShuttleSet / benchmark path**
```
Phase 00: environment setup
  → Phase 01: dataset inventory
  → Phase 02: ground-truth tactical tables from ShuttleSet annotations
  → Phase 03: collate ShuttleSet BST features
  → Phase 04: BST fine-tuning + stroke classification
  → Phase 11: tactical mining on ground truth + predictions
```

**Track B — Raw unseen-video path**
```
raw MP4
  → Phase 05: SA-CNN fine-tuning + rally filtering
  → Phase 06: TrackNetV3 shuttle tracking + denoising
  → Phase 07: hit-frame parameter tuning + stroke windows
  → Phase 08: court calibration / homography
  → Phase 09: pose extraction, player position, feature normalization
  → Phase 10: integrated inference
  → Phase 11: tactical mining
```

---

## Running the Tools

Each `tools/phase*.py` script is self-contained and reads from `outputs/` produced by prior phases. Paths are currently hardcoded relative to the `project/` working directory — adjust the `ROOT` or `BASE` constants at the top of each script before running.

External model repos required (not included here):
- `BST-Badminton-Stroke-type-Transformer`
- `TrackNetV3`
- `mmpose` (YOLO26x-pose)
- `A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection`

See [`__guidance__/source.md`](__guidance__/source.md) for upstream repository links and setup notes.

---

## Guidance and Reports

- [`__guidance__/REPORT.md`](__guidance__/REPORT.md) — full project report with results and analysis
- [`__guidance__/systems.md`](__guidance__/systems.md) — pipeline architecture and design decisions
- [`__guidance__/report/`](__guidance__/report/) — per-phase execution logs (inputs, commands, outputs, validation)
- [`__guidance__/COMP4040_course_project_and_rubric.md`](__guidance__/COMP4040_course_project_and_rubric.md) — course rubric
