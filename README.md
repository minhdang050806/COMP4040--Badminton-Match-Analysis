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

The code is organized by **pipeline stage**, each stage a self-contained Python
package. Data flows top-to-bottom: perception stages produce features, `modeling`
classifies strokes, and `data_mining` analyses the resulting stroke stream.

```
COMP4040--Badminton-Match-Analysis/
├── data_prep/                      # Dataset inventory + ShuttleSet ground-truth tables
│   ├── inventory.py
│   └── ground_truth_tables.py
│
├── rally_filtering/                # SA-CNN rally-view filtering
│   ├── build_dataset.py            # Build SA-CNN ImageFolder dataset
│   ├── train.py                    # Fine-tune SA-CNN
│   ├── filter_rallies.py           # Apply filter to isolate rally frames
│   └── finetune_all44.sh           # End-to-end fine-tuning driver
│
├── shuttle_tracking/               # TrackNetV3 shuttle tracking + denoising
│   └── track.py
│
├── hit_detection/                  # Hit-frame detection + stroke windows
│   ├── detect.py
│   ├── tune_parameters.py
│   └── build_ground_truth_windows.py
│
├── pose_features/                  # YOLO-pose player position + feature extraction
│   ├── extract.py
│   └── compare_reference.py
│
├── modeling/                       # Badminton-Stroke Transformer (stroke classification)
│   ├── label_names.py              # Shared stroke-label helpers
│   ├── collate_features.py         # Collate ShuttleSet BST features
│   ├── train.py                    # Fine-tune BST
│   ├── inference.py                # BST inference (model + metrics)
│   ├── prepare_inputs.py           # Build integrated-pipeline BST inputs
│   ├── integrated_inference.py     # Run BST on the raw-video pipeline
│   └── translate_outputs.py        # Translate predicted labels to English
│
├── data_mining/                    # Tactical pattern mining
│   ├── tactical_mining.py          # Transitions, motifs, spatial, clustering
│   ├── movement_strategy.py        # Movement / strategy clustering
│   ├── validation.py               # Mining-result validation study
│   └── validation_figures.py       # Validation figures
│
├── visualization/                  # Figures and QA visualizations
│   ├── journal_report.py
│   ├── rally_and_shuttle.py
│   ├── feature_plots.py
│   ├── shuttle_tracking_rectangles.py
│   └── video44_showcase.py
│
├── tests/                          # Unit tests (run with pytest / unittest)
│   ├── test_movement_strategy.py
│   ├── test_pose_features.py
│   ├── test_modeling_integration.py
│   └── test_tactical_mining.py
│
├── __guidance__/                   # Project documentation and reports
│   ├── COMP4040_course_project_and_rubric.md
│   ├── REPORT.md                   # Main project report
│   ├── systems.md                  # Full pipeline architecture
│   ├── data.md                     # Dataset documentation
│   ├── data_mining.md              # Mining methodology
│   ├── interpretation.md           # Results interpretation
│   ├── source.md                   # External sources and repos
│   ├── final_report/               # LaTeX final report (modular)
│   │   ├── main.tex                # Entry point — \input{src/*}
│   │   ├── reference.bib           # Bibliography
│   │   ├── src/                    # Per-section .tex sources
│   │   ├── images/                 # Result figures used in the report
│   │   └── figures/                # Logos / static assets
│   ├── images/                     # Figures used in the markdown reports
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
├── outputs/                        # Generated pipeline artifacts (git-ignored)
├── requirements.txt                # Python dependencies
├── conftest.py                     # Puts repo root on sys.path for tests
└── .gitignore
```

> **Not committed to this repo** (generated, large, or binary — see `.gitignore`).
> Create these locally; the pipeline reads/writes them but they are never tracked:
> - `dataset/` — raw ShuttleSet videos, annotations, and image crops
> - `weights/` — pre-trained and fine-tuned model weights (`*.pt`, `*.pth`)
> - `outputs/` — all generated CSV / JSON / NPY tables and figures produced by the pipeline
> - Intermediate numpy feature arrays (`*.npy`, `*.npz`) and raw broadcast video

---

## Pipeline Architecture

The project has two tracks:

**Track A — ShuttleSet / benchmark path**
```
data_prep.inventory                 # dataset inventory
  → data_prep.ground_truth_tables   # ground-truth tactical tables from ShuttleSet
  → modeling.collate_features       # collate ShuttleSet BST features
  → modeling.train / modeling.inference   # BST fine-tuning + stroke classification
  → data_mining.tactical_mining     # tactical mining on ground truth + predictions
```

**Track B — Raw unseen-video path**
```
raw MP4
  → rally_filtering.*               # SA-CNN fine-tuning + rally filtering
  → shuttle_tracking.track          # TrackNetV3 shuttle tracking + denoising
  → hit_detection.*                 # hit-frame parameter tuning + stroke windows
  → (court calibration / homography)
  → pose_features.extract           # pose extraction, player position, normalization
  → modeling.prepare_inputs / modeling.integrated_inference   # integrated inference
  → data_mining.tactical_mining     # tactical mining
```

---

## Installation

```bash
git clone <repo-url>
cd COMP4040--Badminton-Match-Analysis

python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

> `torch` / `torchvision` are listed generically. For GPU support, install the
> build matching your CUDA version from <https://pytorch.org> before (or instead
> of) `pip install -r requirements.txt`.

### Local data layout

The pipeline reads datasets/weights and writes results into directories that are
**not** committed (they are listed in `.gitignore`). Create them locally:

```
COMP4040--Badminton-Match-Analysis/
├── dataset/    # ShuttleSet videos, annotations, crops  (you provide)
├── weights/    # *.pt / *.pth model checkpoints           (you provide)
└── outputs/    # produced by the tools as you run phases  (auto-created)
```

---

## Running the Pipeline

Each stage is a module inside its package and is self-contained — it reads from
`outputs/` produced by earlier stages. Run modules **from the repository root**
with `python -m <package>.<module>` so cross-package imports resolve, e.g.:

```bash
python -m data_prep.inventory                 # dataset inventory
python -m modeling.train --help               # see options for a stage
python -m data_mining.tactical_mining         # tactical pattern mining
```

> Data paths (`ROOT` / `DEFAULT_*` constants at the top of each module) are still
> hardcoded relative to the original working directory. Adjust them to point at
> your local `dataset/`, `weights/`, and `outputs/` before running a stage.

### Tests

```bash
pip install pytest          # already in requirements.txt
pytest                      # or: python -m unittest discover -s tests
```

`conftest.py` puts the repo root on `sys.path` so tests can import the pipeline
packages, e.g. `from data_mining.tactical_mining import cluster_profiles`.

External model repos required (not included here):
- `BST-Badminton-Stroke-type-Transformer`
- `TrackNetV3`
- `mmpose` (YOLO26x-pose)
- `A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection`

See [`__guidance__/source.md`](__guidance__/source.md) for upstream repository links and setup notes.

---

## Guidance and Reports

- [`__guidance__/final_report/main.tex`](__guidance__/final_report/main.tex) — LaTeX final report (compile with `pdflatex main.tex && bibtex main && pdflatex main.tex` twice)
- [`__guidance__/REPORT.md`](__guidance__/REPORT.md) — full project report with results and analysis
- [`__guidance__/systems.md`](__guidance__/systems.md) — pipeline architecture and design decisions
- [`__guidance__/report/`](__guidance__/report/) — per-phase execution logs (inputs, commands, outputs, validation)
- [`__guidance__/COMP4040_course_project_and_rubric.md`](__guidance__/COMP4040_course_project_and_rubric.md) — course rubric
