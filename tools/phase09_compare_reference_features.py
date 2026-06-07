from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = ROOT / "project" / "outputs" / "features_yolo26x_tracked_phase02"
DEFAULT_REFERENCE_ROOT = ROOT / "project" / "dataset" / "ShuttleSet" / "merged_seq100_between_2_hits_with_max_limits"
DEFAULT_PHASE02 = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "phase09_reference_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def branch_metrics(
    joints: np.ndarray,
    positions: np.ndarray,
    shuttle: np.ndarray,
    validity_path: Path | None = None,
) -> dict[str, float | int]:
    if validity_path is not None and validity_path.exists():
        validity = np.load(validity_path)
        pose_valid = np.stack([validity["p1_pose_valid"], validity["p2_pose_valid"]], axis=1)
        position_valid = np.stack([validity["p1_position_valid"], validity["p2_position_valid"]], axis=1)
    else:
        pose_valid = np.any(joints != 0.0, axis=(2, 3))
        position_valid = np.any(positions != 0.0, axis=2)
    both_pose_valid = np.all(pose_valid, axis=1)
    both_position_valid = np.all(position_valid, axis=1)
    position_in_tolerance = (positions >= -0.2) & (positions <= 1.2)
    side_consistent = (
        (~position_valid[:, 0] | (positions[:, 0, 1] <= 0.5))
        & (~position_valid[:, 1] | (positions[:, 1, 1] > 0.5))
    )
    return {
        "length": len(positions),
        "both_pose_valid_rate": float(both_pose_valid.mean()),
        "both_position_valid_rate": float(both_position_valid.mean()),
        "pose_zero_rate": float((joints == 0.0).mean()),
        "position_zero_rate": float((positions == 0.0).mean()),
        "shuttle_zero_rate": float(np.all(shuttle == 0.0, axis=1).mean()),
        "position_out_of_tolerance_values": int(np.count_nonzero(position_valid[:, :, None] & ~position_in_tolerance)),
        "position_out_of_tolerance_rate": float(
            np.count_nonzero(position_valid[:, :, None] & ~position_in_tolerance)
            / max(int(np.count_nonzero(position_valid)) * 2, 1)
        ),
        "player_side_consistency_rate": float(side_consistent.mean()),
    }


def reference_branches(reference_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for path in (reference_root / split).glob("*/*_pos.npy"):
            output[path.name.removesuffix("_pos.npy")] = Path(str(path).removesuffix("_pos.npy"))
    return output


def reference_hit_offsets(phase02_path: Path, fps: float) -> dict[str, int]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv(phase02_path):
        groups.setdefault((row["match_id"], row["set_id"], row["rally_id"]), []).append(row)
    output: dict[str, int] = {}
    half_second = int(fps // 2)
    limit = int(fps * 3 // 2)
    for rows in groups.values():
        rows.sort(key=lambda row: int(float(row["ball_round_id"])))
        for index, row in enumerate(rows):
            current = int(float(row["frame_num_int"]))
            previous = int(float(rows[index - 1]["frame_num_int"])) if index > 0 else current - half_second
            start = max(previous, current - limit)
            output[row["clip_id"]] = current - start
    return output


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched generated Phase 09 features against reference BST features.")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--phase02-table", type=Path, default=DEFAULT_PHASE02)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-fps", type=float, default=30.0)
    parser.add_argument("--max-clips", type=int, default=0)
    args = parser.parse_args()

    manifest = read_csv(args.feature_root / "phase09_feature_manifest.csv")
    if args.max_clips > 0:
        manifest = manifest[: args.max_clips]
    references = reference_branches(args.reference_root)
    expected_offsets = reference_hit_offsets(args.phase02_table, args.reference_fps)
    event_frames = {row["clip_id"]: int(float(row["frame_num_int"])) for row in read_csv(args.phase02_table)}
    rows: list[dict[str, Any]] = []

    for generated in manifest:
        clip_id = generated["clip_id"]
        reference = references.get(clip_id)
        if reference is None:
            continue
        generated_joints = np.load(resolve_path(generated["joints_npy"]))
        generated_pos = np.load(resolve_path(generated["pos_npy"]))
        generated_shuttle = np.load(resolve_path(generated["shuttle_npy"]))
        reference_joints = np.load(str(reference) + "_joints.npy")
        reference_pos = np.load(str(reference) + "_pos.npy")
        reference_shuttle = np.load(str(reference) + "_shuttle.npy")
        validity_value = generated.get("validity_npz", "")
        validity_path = resolve_path(validity_value) if validity_value else None
        generated_metrics = branch_metrics(generated_joints, generated_pos, generated_shuttle, validity_path)
        reference_metrics = branch_metrics(reference_joints, reference_pos, reference_shuttle)
        if generated.get("event_offset_frames", "") != "":
            generated_offset = int(generated["event_offset_frames"])
        else:
            generated_offset = event_frames.get(clip_id, int(generated["window_start_original"])) - int(
                generated["window_start_original"]
            )
        expected_offset = expected_offsets.get(clip_id)
        row: dict[str, Any] = {
            "clip_id": clip_id,
            "video_id": generated["video_id"],
            "rally_id": generated["rally_id"],
            "window_strategy": generated.get("window_strategy", ""),
            "generated_event_offset_frames": generated_offset,
            "reference_expected_event_offset_frames": expected_offset if expected_offset is not None else "",
            "event_offset_abs_error": abs(generated_offset - expected_offset) if expected_offset is not None else "",
        }
        for key, value in generated_metrics.items():
            row[f"generated_{key}"] = value
        for key, value in reference_metrics.items():
            row[f"reference_{key}"] = value
        row["length_abs_error"] = abs(int(generated_metrics["length"]) - int(reference_metrics["length"]))
        rows.append(row)

    output_csv = args.output_root / "phase09_matched_reference_comparison.csv"
    write_csv(output_csv, rows)
    metric_names = [
        "length",
        "both_pose_valid_rate",
        "both_position_valid_rate",
        "pose_zero_rate",
        "position_zero_rate",
        "shuttle_zero_rate",
        "position_out_of_tolerance_values",
        "position_out_of_tolerance_rate",
        "player_side_consistency_rate",
    ]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if rows else "blocked_no_matched_clips",
        "generated_feature_root": str(args.feature_root),
        "reference_feature_root": str(args.reference_root),
        "generated_manifest_rows": len(manifest),
        "matched_clip_count": len(rows),
        "metrics": {
            name: {
                "generated_mean": mean(rows, f"generated_{name}"),
                "reference_mean": mean(rows, f"reference_{name}"),
            }
            for name in metric_names
        },
        "length_mae": mean(rows, "length_abs_error"),
        "event_offset_mae": mean(rows, "event_offset_abs_error"),
        "output_csv": str(output_csv),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "phase09_matched_reference_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
