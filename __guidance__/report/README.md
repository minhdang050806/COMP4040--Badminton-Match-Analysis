# Phase Reports

Every execution phase in `project/__guidance__/systems.md` must produce one Markdown report in this directory.

Required files:

```text
phase_00_setup.md
phase_01_dataset_inventory.md
phase_02_ground_truth_tables.md
phase_03_bst_collation.md
phase_04_bst_inference.md
phase_05_rally_filtering.md
phase_06_shuttle_tracking.md
phase_07_hit_frame_detection.md
phase_08_court_homography.md
phase_09_pose_position.md
phase_10_integrated_inference.md
phase_11_tactical_mining.md
```

Use this template for each report:

```markdown
# Phase NN: Title

## Objective

## Inputs

## Commands

## Outputs

## Validation

## Assumptions and Unclear Items

## Blockers

## Next Phase Handoff
```

Do not mark a phase complete until its report lists the generated artifacts and validation results. If a phase needs visual review, keep machine checks and human signoff as separate statuses.
