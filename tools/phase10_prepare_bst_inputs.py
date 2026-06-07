from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

try:
    from phase03_collate_bst_features import get_bone_pairs, load_and_augment
    from bst_label_names import get_merged_stroke_types_english, translate_stroke_type
except ModuleNotFoundError:
    from project.tools.phase03_collate_bst_features import get_bone_pairs, load_and_augment
    from project.tools.bst_label_names import get_merged_stroke_types_english, translate_stroke_type


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = ROOT / "project" / "outputs" / "features_yolo26x_bst_tracked_phase02_41_44"
DEFAULT_PHASE02_TABLE = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_PHASE07_MATCHES = ROOT / "project" / "outputs" / "hit_frames" / "phase07_matches.csv"
DEFAULT_REFERENCE_COLLATED = ROOT / "project" / "outputs" / "bst_collated" / "merged_seq100_between_2_hits_with_max_limits"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44" / "inputs"
DEFAULT_SEQ_LEN = 100

METADATA_COLUMNS = [
    "row_index",
    "video_id",
    "rally_id",
    "event_rank",
    "clip_id",
    "feature_alias_source_clip_id",
    "feature_source",
    "window_strategy",
    "player_order",
    "event_offset_frames",
    "window_start_original",
    "window_end_original",
    "original_len",
    "video_len_after_collation",
    "joints_npy",
    "pos_npy",
    "shuttle_npy",
    "source_video",
    "validity_npz",
    "zero_pose_frames",
    "zero_pose_rate",
    "pose_missing_rate",
    "position_missing_rate",
    "p1_pose_valid_rate",
    "p2_pose_valid_rate",
    "p1_position_valid_rate",
    "p2_position_valid_rate",
    "both_pose_missing_rate",
    "both_position_missing_rate",
    "pose_tensor_zero_rate",
    "position_tensor_zero_rate",
    "shuttle_visible_frames",
    "shuttle_visible_rate",
    "player_pos_out_of_unit_range",
    "player_side_inconsistent_rate",
    "quality_group",
    "has_phase02_label",
    "stable_key",
    "match_id",
    "set_id",
    "gt_rally_id",
    "ball_round_id",
    "event_frame_original",
    "event_frame_source",
    "stroke_type_ground_truth",
    "player",
    "server",
    "true_label",
    "true_label_name",
    "reference_split",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [to_float(row[key]) for row in rows if row.get(key, "") != ""]
    return float(np.mean(values)) if values else None


def reference_labels(collated_root: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    class_names_english = get_merged_stroke_types_english()
    for split in ("train", "val", "test"):
        path = collated_root / f"{split}_metadata.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            label = to_int(row["label"], default=-1)
            output[row["clip_id"]] = {
                "true_label": row["label"],
                "true_label_name": class_names_english[label] if 0 <= label < len(class_names_english) else "",
                "reference_split": split,
            }
    return output


def quality_group(pose_missing_rate: float, has_phase02_label: bool) -> str:
    prefix = "exact_label" if has_phase02_label else "candidate_only"
    if pose_missing_rate == 1.0:
        return f"{prefix}_all_pose_missing"
    if pose_missing_rate >= 0.5:
        return f"{prefix}_pose_dropout_ge_50"
    if pose_missing_rate >= 0.25:
        return f"{prefix}_pose_dropout_25_50"
    return f"{prefix}_pose_dropout_lt_25"


def complement_mean(first: Any, second: Any, fallback: float) -> float:
    if first in (None, "") or second in (None, ""):
        return fallback
    return 1.0 - (to_float(first) + to_float(second)) / 2.0


def validate_source_contract(
    manifest: list[dict[str, str]],
    validation_by_clip: dict[str, dict[str, str]],
) -> dict[str, int]:
    clip_ids = [row["clip_id"] for row in manifest]
    duplicate_clip_ids = len(clip_ids) - len(set(clip_ids))
    missing_validation_rows = 0
    missing_validity_sidecars = 0
    event_outside_window = 0
    frame_count_mismatches = 0
    for row in manifest:
        quality = validation_by_clip.get(row["clip_id"])
        if quality is None:
            missing_validation_rows += 1
        elif row.get("frame_count", "") != "" and to_int(row["frame_count"]) != to_int(quality.get("frame_count")):
            frame_count_mismatches += 1
        validity_path = row.get("validity_npz", "")
        if validity_path and not resolve_path(validity_path).exists():
            missing_validity_sidecars += 1
        event = row.get("event_frame_original", "")
        if event != "" and not (
            to_int(row["window_start_original"]) <= to_int(event) <= to_int(row["window_end_original"])
        ):
            event_outside_window += 1
    checks = {
        "duplicate_clip_ids": duplicate_clip_ids,
        "missing_validation_rows": missing_validation_rows,
        "missing_validity_sidecars": missing_validity_sidecars,
        "event_outside_window": event_outside_window,
        "frame_count_mismatches": frame_count_mismatches,
    }
    if any(checks.values()):
        raise RuntimeError(f"Phase 09 source contract failed: {checks}")
    return checks


def expand_simultaneous_event_aliases(
    manifest: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    phase07_matches: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    grouped_manifest: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in manifest:
        key = (row["video_id"], row["rally_id"], row.get("event_frame_original", ""))
        grouped_manifest.setdefault(key, []).append(row)
    grouped_matches: dict[tuple[str, str, str], list[str]] = {}
    for row in phase07_matches:
        key = (row["video_id"], row["rally_id"], row["predicted_frame_original"])
        grouped_matches.setdefault(key, []).append(row["clip_id"])
    validation_by_clip = {row["clip_id"]: row for row in validation_rows}

    expanded: list[dict[str, str]] = []
    expanded_validation: dict[str, dict[str, str]] = {}
    aliased_rows = 0
    repaired_groups = 0
    for key, rows in grouped_manifest.items():
        rows.sort(key=lambda row: to_int(row["event_rank"]))
        target_clip_ids = grouped_matches.get(key, [])
        if len(rows) == 1:
            row = rows[0].copy()
            row["feature_alias_source_clip_id"] = ""
            expanded.append(row)
            if row["clip_id"] in validation_by_clip:
                expanded_validation[row["clip_id"]] = validation_by_clip[row["clip_id"]]
            continue
        if len(target_clip_ids) != len(rows):
            raise RuntimeError(
                f"Cannot repair simultaneous Phase 09 event {key}: manifest_rows={len(rows)}, "
                f"phase07_matches={len(target_clip_ids)}"
            )
        source_clip_id = rows[-1]["clip_id"]
        source_validation = validation_by_clip[source_clip_id]
        source_frame_count = to_int(source_validation["frame_count"])
        canonical = next((row for row in reversed(rows) if to_int(row["frame_count"]) == source_frame_count), rows[-1])
        for original, target_clip_id in zip(rows, target_clip_ids):
            row = canonical.copy()
            row["clip_id"] = target_clip_id
            row["event_rank"] = original["event_rank"]
            row["feature_alias_source_clip_id"] = source_clip_id
            expanded.append(row)
            quality = source_validation.copy()
            quality["clip_id"] = target_clip_id
            quality["event_rank"] = original["event_rank"]
            expanded_validation[target_clip_id] = quality
            aliased_rows += 1
        repaired_groups += 1
    expanded.sort(key=lambda row: (to_int(row["video_id"]), to_int(row["rally_id"]), to_int(row["event_rank"])))
    return expanded, expanded_validation, {
        "simultaneous_event_groups_repaired": repaired_groups,
        "rows_using_shared_feature_alias": aliased_rows,
    }


def create_arrays(output_root: Path, count: int, seq_len: int) -> dict[str, np.memmap]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "JnB_bone": open_memmap(
            output_root / "JnB_bone.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, seq_len, 2, 36, 2),
        ),
        "pos": open_memmap(output_root / "pos.npy", mode="w+", dtype=np.float32, shape=(count, seq_len, 2, 2)),
        "shuttle": open_memmap(output_root / "shuttle.npy", mode="w+", dtype=np.float32, shape=(count, seq_len, 2)),
        "videos_len": open_memmap(output_root / "videos_len.npy", mode="w+", dtype=np.int64, shape=(count,)),
        "labels": open_memmap(output_root / "labels.npy", mode="w+", dtype=np.int64, shape=(count,)),
    }


def validate_arrays(output_root: Path, count: int, seq_len: int) -> dict[str, Any]:
    expected = {
        "JnB_bone": (count, seq_len, 2, 36, 2),
        "pos": (count, seq_len, 2, 2),
        "shuttle": (count, seq_len, 2),
        "videos_len": (count,),
        "labels": (count,),
    }
    arrays: dict[str, Any] = {}
    for name, shape in expected.items():
        path = output_root / f"{name}.npy"
        array = np.load(path, mmap_mode="r")
        arrays[name] = {
            "path": relative(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "shape_ok": tuple(array.shape) == shape,
            "finite": bool(np.isfinite(array).all()),
            "size_bytes": path.stat().st_size,
        }
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare arbitrary Phase 09 triples for Phase 10 BST inference.")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--phase02-table", type=Path, default=DEFAULT_PHASE02_TABLE)
    parser.add_argument("--phase07-matches", type=Path, default=DEFAULT_PHASE07_MATCHES)
    parser.add_argument("--reference-collated-root", type=Path, default=DEFAULT_REFERENCE_COLLATED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-clips", type=int, default=0, help="CPU preparation smoke limit; 0 means all.")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    manifest_path = args.feature_root / "phase09_feature_manifest.csv"
    validation_path = args.feature_root / "phase09_feature_validation.csv"
    feature_summary_path = args.feature_root / "phase09_feature_summary.json"
    manifest = read_csv(manifest_path)
    validation_rows = read_csv(validation_path)
    manifest, validation_by_clip, alias_repair = expand_simultaneous_event_aliases(
        manifest,
        validation_rows,
        read_csv(args.phase07_matches),
    )
    if args.max_clips > 0:
        manifest = manifest[: args.max_clips]
    if not manifest:
        raise RuntimeError("No Phase 09 feature rows selected.")

    source_contract = validate_source_contract(manifest, validation_by_clip)
    phase02_by_clip = {row["clip_id"]: row for row in read_csv(args.phase02_table)}
    reference_by_clip = reference_labels(args.reference_collated_root)
    arrays = create_arrays(args.output_root, len(manifest), args.seq_len)
    bone_pairs = get_bone_pairs()
    metadata_rows: list[dict[str, Any]] = []

    for index, row in enumerate(manifest):
        clip_id = row["clip_id"]
        branch = Path(str(resolve_path(row["joints_npy"]))[: -len("_joints.npy")])
        j_only, _, jnb_bone, _, pos, shuttle, video_len, original_len = load_and_augment(
            branch,
            args.seq_len,
            bone_pairs,
        )
        del j_only
        arrays["JnB_bone"][index] = jnb_bone
        arrays["pos"][index] = pos
        arrays["shuttle"][index] = shuttle
        arrays["videos_len"][index] = video_len

        quality = validation_by_clip.get(clip_id, {})
        phase02 = phase02_by_clip.get(clip_id, {})
        reference = reference_by_clip.get(clip_id, {})
        label = to_int(reference.get("true_label"), default=-1)
        arrays["labels"][index] = label
        tensor_zero_rate = to_float(quality.get("pose_tensor_zero_rate", quality.get("zero_pose_rate")))
        pose_missing_rate = complement_mean(
            quality.get("p1_pose_valid_rate"),
            quality.get("p2_pose_valid_rate"),
            tensor_zero_rate,
        )
        position_missing_rate = complement_mean(
            quality.get("p1_position_valid_rate"),
            quality.get("p2_position_valid_rate"),
            to_float(quality.get("position_tensor_zero_rate")),
        )
        zero_pose_frames = int(round(pose_missing_rate * original_len))
        shuttle_visible_rate = to_float(quality.get("shuttle_visible_rate"))
        window_start = to_int(row["window_start_original"])
        window_end = to_int(row["window_end_original"])
        event_frame = row.get("event_frame_original", "") or phase02.get("frame_num_int", "")
        event_frame_source = "phase09_exact" if row.get("event_frame_original", "") != "" else "phase02_exact"
        if event_frame == "":
            event_frame = int(round((window_start + window_end) / 2))
            event_frame_source = "window_midpoint_estimate"
        metadata_rows.append(
            {
                "row_index": index,
                "video_id": row["video_id"],
                "rally_id": row["rally_id"],
                "event_rank": row["event_rank"],
                "clip_id": clip_id,
                "feature_alias_source_clip_id": row.get("feature_alias_source_clip_id", ""),
                "feature_source": row["feature_source"],
                "window_strategy": row.get("window_strategy", ""),
                "player_order": row.get("player_order", ""),
                "event_offset_frames": row.get("event_offset_frames", to_int(event_frame) - window_start),
                "window_start_original": row["window_start_original"],
                "window_end_original": row["window_end_original"],
                "original_len": original_len,
                "video_len_after_collation": video_len,
                "joints_npy": row["joints_npy"],
                "pos_npy": row["pos_npy"],
                "shuttle_npy": row["shuttle_npy"],
                "source_video": row["source_video"],
                "validity_npz": row.get("validity_npz", ""),
                "zero_pose_frames": zero_pose_frames,
                "zero_pose_rate": pose_missing_rate,
                "pose_missing_rate": pose_missing_rate,
                "position_missing_rate": position_missing_rate,
                "p1_pose_valid_rate": quality.get("p1_pose_valid_rate", ""),
                "p2_pose_valid_rate": quality.get("p2_pose_valid_rate", ""),
                "p1_position_valid_rate": quality.get("p1_position_valid_rate", ""),
                "p2_position_valid_rate": quality.get("p2_position_valid_rate", ""),
                "both_pose_missing_rate": quality.get("both_pose_missing_rate", ""),
                "both_position_missing_rate": quality.get("both_position_missing_rate", ""),
                "pose_tensor_zero_rate": tensor_zero_rate,
                "position_tensor_zero_rate": quality.get("position_tensor_zero_rate", ""),
                "shuttle_visible_frames": int(round(shuttle_visible_rate * original_len)),
                "shuttle_visible_rate": shuttle_visible_rate,
                "player_pos_out_of_unit_range": quality.get(
                    "player_pos_out_of_filter_range",
                    quality.get("player_pos_out_of_unit_range", ""),
                ),
                "player_side_inconsistent_rate": quality.get("player_side_inconsistent_rate", ""),
                "quality_group": quality_group(pose_missing_rate, bool(phase02)),
                "has_phase02_label": bool(phase02),
                "stable_key": phase02.get("stable_key", ""),
                "match_id": phase02.get("match_id", ""),
                "set_id": phase02.get("set_id", ""),
                "gt_rally_id": phase02.get("rally_id", ""),
                "ball_round_id": phase02.get("ball_round_id", ""),
                "event_frame_original": event_frame,
                "event_frame_source": event_frame_source,
                "stroke_type_ground_truth": translate_stroke_type(phase02.get("stroke_type_ground_truth", "")),
                "player": phase02.get("player", ""),
                "server": phase02.get("server", ""),
                "true_label": label,
                "true_label_name": reference.get("true_label_name", ""),
                "reference_split": reference.get("reference_split", ""),
            }
        )
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"Prepared {index + 1}/{len(manifest)} Phase 10 inputs")

    for array in arrays.values():
        array.flush()

    metadata_path = args.output_root / "phase10_input_metadata.csv"
    write_csv(metadata_path, metadata_rows, METADATA_COLUMNS)
    validation = validate_arrays(args.output_root, len(manifest), args.seq_len)
    labeled_rows = sum(to_int(row["true_label"], -1) >= 0 for row in metadata_rows)
    phase02_rows = sum(str(row["has_phase02_label"]).lower() == "true" for row in metadata_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "project/tools/phase10_prepare_bst_inputs.py",
        "status": "passed"
        if all(item["shape_ok"] and item["finite"] for item in validation.values())
        else "review",
        "feature_root": relative(args.feature_root),
        "feature_manifest": relative(manifest_path),
        "feature_validation": relative(validation_path),
        "feature_summary": relative(feature_summary_path) if feature_summary_path.exists() else "",
        "phase02_table": relative(args.phase02_table),
        "phase07_matches": relative(args.phase07_matches),
        "reference_collated_root": relative(args.reference_collated_root),
        "output_root": relative(args.output_root),
        "seq_len": args.seq_len,
        "selected_feature_rows": len(manifest),
        "phase02_matched_rows": phase02_rows,
        "rows_with_bst_reference_label": labeled_rows,
        "candidate_only_rows": len(manifest) - phase02_rows,
        "evaluation_coverage": {
            "reference_test_all": sum(row["reference_split"] == "test" for row in metadata_rows),
            "reference_test_clean": sum(
                row["reference_split"] == "test" and row["feature_alias_source_clip_id"] == ""
                for row in metadata_rows
            ),
            "reference_val_all": sum(row["reference_split"] == "val" for row in metadata_rows),
            "reference_val_clean": sum(
                row["reference_split"] == "val" and row["feature_alias_source_clip_id"] == ""
                for row in metadata_rows
            ),
        },
        "source_contract": source_contract,
        "simultaneous_event_alias_repair": alias_repair,
        "window_strategy_counts": {
            strategy: sum(row["window_strategy"] == strategy for row in metadata_rows)
            for strategy in sorted({row["window_strategy"] for row in metadata_rows})
        },
        "player_order_counts": {
            order: sum(row["player_order"] == order for row in metadata_rows)
            for order in sorted({row["player_order"] for row in metadata_rows})
        },
        "quality_means": {
            key: mean(metadata_rows, key)
            for key in (
                "pose_missing_rate",
                "position_missing_rate",
                "p1_pose_valid_rate",
                "p2_pose_valid_rate",
                "p1_position_valid_rate",
                "p2_position_valid_rate",
                "both_pose_missing_rate",
                "both_position_missing_rate",
                "pose_tensor_zero_rate",
                "position_tensor_zero_rate",
                "shuttle_visible_rate",
                "player_side_inconsistent_rate",
            )
        },
        "quality_group_counts": {
            group: sum(row["quality_group"] == group for row in metadata_rows)
            for group in sorted({row["quality_group"] for row in metadata_rows})
        },
        "arrays": validation,
        "outputs": {
            "metadata": relative(metadata_path),
            "summary": relative(args.output_root / "phase10_prepare_summary.json"),
        },
    }
    (args.output_root / "phase10_prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": relative(args.output_root), "rows": len(manifest), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
