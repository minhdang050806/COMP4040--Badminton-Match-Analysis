# Phase 00: Setup and Environment Baseline

## Objective

Establish the runnable environment, local source-code boundaries, checkpoint availability, and setup blockers before executing any model, collation, training, or data-mining phase.

Status: baseline completed. Later GPU/model phases are not ready until the active Python environment can access CUDA and MMPose is installed or made importable.

## Inputs

Guidance files:

- `project/__guidance__/source.md`
- `project/__guidance__/data.md`
- `project/__guidance__/systems.md`
- `project/__guidance__/report/README.md`

Source repositories checked:

- `external_repos/BST-Badminton-Stroke-type-Transformer`
- `external_repos/TrackNetV3`
- `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis`
- `external_repos/monotrack`
- `external_repos/mmpose`

Data and weight roots checked:

- `project/dataset/ShuttleSet`
- `project/dataset/ShuttleSet_raw_videos`
- `project/weights/on_ShuttleSet`

## Commands

Main implementation command:

```bash
python project/tools/phase00_01_inventory.py
```

Validation and setup inspection commands run during this phase:

```bash
uname -a
python --version
which python
gcc --version
cmake --version
ffmpeg -version
nvidia-smi
python -m py_compile project/tools/phase00_01_inventory.py
```

Additional Python import/version checks were run for `torch`, `torchvision`, `numpy`, `pandas`, `cv2`, `mmpose`, `mmcv`, `mmengine`, `scipy`, `matplotlib`, `PIL`, `yaml`, and `sklearn`.

## Outputs

Generated artifacts:

- `project/tools/phase00_01_inventory.py`
- `project/outputs/inventory/phase00_environment_baseline.json`
- `project/outputs/inventory/phase00_external_repos.csv`
- `project/outputs/inventory/phase00_checkpoint_inventory.csv`

The same script also generated Phase 01 inventory artifacts; those are described in `phase_01_dataset_inventory.md`.

## Environment Snapshot

Host and tools:

| Item | Observed value |
|---|---|
| OS | Linux `5.15.0-164-generic` on x86_64 Ubuntu/glibc 2.35 |
| Python | `3.12.9` |
| Python executable | `/home/dang.cpm/miniconda3/bin/python` |
| Conda env | `base` |
| `CUDA_VISIBLE_DEVICES` | unset |
| gcc | `11.4.0` |
| cmake | `3.22.1` |
| ffmpeg | `4.4.2-0ubuntu0.22.04.1` |

Python package status in the active environment:

| Package | Status | Version / note |
|---|---:|---|
| `torch` | present | `2.7.1+cu126` |
| `torchvision` | present | `0.22.1+cu126` |
| `numpy` | present | `1.26.4` |
| `pandas` | present | `2.2.3` |
| `cv2` | present | `4.10.0` |
| `mmcv` | present | `2.1.0` |
| `mmengine` | present | `0.10.7` |
| `mmpose` | missing | source repo exists, package not importable |
| `scipy` | present | `1.17.1` |
| `matplotlib` | present | `3.10.1` |
| `PIL` | present | `11.3.0` |
| `yaml` | present | `6.0.2` |
| `sklearn` | present | `1.6.1` |

Torch/CUDA status:

| Check | Observed value |
|---|---:|
| `torch.cuda.is_available()` | `False` |
| `torch.cuda.device_count()` | `0` |
| `torch.version.cuda` | `12.6` |
| cuDNN version | `90501` |

GPU note:

- Direct `nvidia-smi` from the tool session detected eight NVIDIA GeForce RTX 3090 GPUs, driver `560.35.05`, CUDA `12.6`.
- At that moment GPU 2 had a Python process using about `2934 MiB`; other GPUs were effectively idle.
- Inside the inventory script subprocess, `nvidia-smi` returned an error that it could not communicate with the NVIDIA driver.
- Active Python/Torch also cannot initialize NVML and reports zero CUDA devices.

Interpretation: the machine has GPUs, but the current Python execution environment is not GPU-ready for model phases. Do not start TrackNetV3, MMPose, or large BST inference expecting CUDA until this is fixed.

## External Repository Inventory

| Path | Role | Present | Git head |
|---|---|---:|---|
| `external_repos/BST-Badminton-Stroke-type-Transformer` | BST code and data-prep scripts | yes | `2ef1797` |
| `external_repos/TrackNetV3` | shuttle tracking and hit-frame scripts | yes | `d1d96f3` |
| `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis` | SA-CNN rally filtering source | yes | `af7722384` |
| `external_repos/monotrack` | court detection source | yes | `39e9227` |
| `external_repos/mmpose` | pose estimation source | yes | `759b39c1` |

Directory sizes observed separately:

| Path | Size |
|---|---:|
| `external_repos/BST-Badminton-Stroke-type-Transformer` | `37M` |
| `external_repos/TrackNetV3` | `336M` |
| `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis` | `1.3G` |
| `external_repos/monotrack` | `67M` |
| `external_repos/mmpose` | `88M` |

## Checkpoint Inventory

| Path | Role | Status | Size bytes | SHA256 |
|---|---|---:|---:|---|
| `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt` | Phase 05 rally / shot-angle filtering | present | `346656` | `e29e7b11c5e308bcaddaee03633ae64da04d01fdf254a0216a71bfb1807c1653` |
| `external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/scaler.pickle` | legacy full Automated-Hit-frame pipeline scaler; not needed for standalone SA-CNN | present | `2090` | `eb8bf8b0353441db52123a348ca78c6875aef244c0e048fd8efeb6bb03b3b603` |
| `external_repos/TrackNetV3/exp/model_best.pt` | Phase 06 shuttle tracking | present | `181830533` | `ff3fc5687cc83cda19095116881d564ed7fe60de3d013fd4d6a801e88299dc68` |
| `project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt` | Phase 04 / Phase 10 BST_CG_AP stroke classification | present | `7514133` | `997f43e52b20cb47517132c71d3e147d7d717a2d9651cfbe7651a5003bd66809` |
| `project/weights/on_ShuttleSet` | all local ShuttleSet BST/baseline weights | present | directory | 51 `.pt` files |

## Setup Validation

Passed:

- Required project data roots exist.
- Required external source repositories exist.
- Required Phase 05, Phase 06, and Phase 04 checkpoint files exist.
- `project/tools/phase00_01_inventory.py` compiles with `python -m py_compile`.
- Core data-processing libraries are available: `numpy`, `pandas`, `cv2`, `scipy`, `matplotlib`, `sklearn`.

Blocked or risky:

- `torch.cuda.is_available()` is `False`; GPU model phases are blocked in the active Python environment.
- `mmpose` is not importable even though `external_repos/mmpose` exists.
- `matplotlib` emitted a warning because `/home/dang.cpm/.config/matplotlib` is not writable; set `MPLCONFIGDIR` to a writable path such as `/tmp/matplotlib-cache` for plotting-heavy phases.
- monotrack court detector source exists, but no `external_repos/monotrack/court-detection/build/bin/detect` executable was found.
- Active Python is `3.12.9`, while TrackNetV3 README states Python `3.7.9~3.9.4`, and the Automated-Hit-frame README states Python `3.8`. Use separate environments if those upstream stacks fail under Python 3.12.

## Assumptions and Unclear Items

- Unclear from current codebase: whether the intended final execution environment is the active `base` Conda environment or separate per-repo environments.
- Unclear from current codebase: whether MMPose weights are already cached elsewhere; only the source repo is present and the Python import is missing.
- Unclear from current codebase: whether monotrack was previously built in another location; no detector binary was found under the expected source tree.

## Blockers

Before Phase 04, 06, 09, or 10 model execution:

- Fix Torch CUDA visibility or explicitly run CPU-only with expected slowdowns.
- Install or expose MMPose in the active environment before Phase 09.
- Build monotrack before Phase 08.

## Next Phase Handoff

Phase 01 can proceed because it only needs filesystem, CSV, and NumPy access. The current environment is sufficient for dataset inventory and report generation.

Do not begin GPU-dependent inference as "ready" until a follow-up Phase 00 update shows `torch.cuda.is_available() == True` or documents a deliberate CPU-only execution plan.
