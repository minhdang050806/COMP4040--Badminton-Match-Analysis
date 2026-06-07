from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import phase07_hit_frame_detection as p7


DEFAULT_OUTPUT_ROOT = p7.ROOT / "project" / "outputs" / "hit_frames_tuning_all44"

SEARCH_COLUMNS = [
    "candidate_id",
    "split",
    "min_distance_frames",
    "prominence_px",
    "peak_radius",
    "min_angle_degrees",
    "min_step_px",
    "serve_slope_px_per_frame",
    "tolerance_frames",
    "rally_count",
    "predicted_events",
    "ground_truth_hits",
    "matched_ground_truth",
    "missed_ground_truth",
    "extra_predictions",
    "precision",
    "recall",
    "f1",
    "mean_abs_error",
]


@dataclass(frozen=True)
class ParameterCandidate:
    candidate_id: int
    min_distance_frames: int
    prominence_px: float
    peak_radius: int
    min_angle_degrees: float
    min_step_px: float
    serve_slope_px_per_frame: float

    def as_runtime_kwargs(self) -> dict[str, int | float]:
        return {
            "min_distance_frames": self.min_distance_frames,
            "prominence_px": self.prominence_px,
            "peak_radius": self.peak_radius,
            "min_angle_degrees": self.min_angle_degrees,
            "min_step_px": self.min_step_px,
            "serve_slope_px_per_frame": self.serve_slope_px_per_frame,
        }


@dataclass(frozen=True)
class TuneTask:
    split: str
    video_id: str
    rally_id: str
    start_frame: int
    end_frame: int
    points: list[p7.TrajectoryPoint]
    gt_rows: list[dict[str, str]]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_video_ids(value: str) -> set[str]:
    ids: set[str] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            ids.update(str(video_id) for video_id in range(int(start), int(end) + 1))
        else:
            ids.add(str(int(item)))
    return ids


def split_for_video(video_id: str, train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> str:
    if video_id in train_ids:
        return "train"
    if video_id in val_ids:
        return "val"
    if video_id in test_ids:
        return "test"
    return "ignored"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def shuttle_roots(args: argparse.Namespace) -> list[Path]:
    return args.shuttle_root or [p7.DEFAULT_SHUTTLE_ROOT]


def load_tasks(args: argparse.Namespace) -> list[TuneTask]:
    manifest_paths = [root / "phase06_shuttle_tracking_manifest.csv" for root in shuttle_roots(args)]
    missing_manifest_paths = [path for path in manifest_paths if not path.exists()]
    if missing_manifest_paths:
        formatted = "\n".join(str(path) for path in missing_manifest_paths)
        raise FileNotFoundError(f"Missing Phase 06 manifest(s):\n{formatted}")
    if not args.ground_truth_table.exists():
        raise FileNotFoundError(f"Missing ground-truth table: {args.ground_truth_table}")

    train_ids = parse_video_ids(args.train_video_ids)
    val_ids = parse_video_ids(args.val_video_ids)
    test_ids = parse_video_ids(args.test_video_ids)
    selected_video_ids = {str(video_id) for video_id in args.video_id}
    gt_by_match = p7.load_ground_truth(args.ground_truth_table)

    counts_by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    tasks: list[TuneTask] = []
    for manifest_path in manifest_paths:
        for row in p7.read_csv(manifest_path):
            video_id = row["video_id"]
            if selected_video_ids and video_id not in selected_video_ids:
                continue
            split = split_for_video(video_id, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
            if split == "ignored":
                continue
            if args.max_rallies_per_split > 0 and counts_by_split[split] >= args.max_rallies_per_split:
                continue

            denoised_csv = Path(row["denoised_csv"])
            if not denoised_csv.is_absolute():
                denoised_csv = p7.ROOT / denoised_csv
            start_frame = p7.to_int(row["start_frame"])
            end_frame = p7.to_int(row["end_frame"])
            gt_rows = p7.ground_truth_in_interval(
                gt_by_match.get(video_id, []),
                start_frame=start_frame,
                end_frame=end_frame,
            )
            tasks.append(
                TuneTask(
                    split=split,
                    video_id=video_id,
                    rally_id=row["rally_id"],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    points=p7.load_trajectory(denoised_csv),
                    gt_rows=gt_rows,
                )
            )
            counts_by_split[split] += 1

    if not tasks:
        raise RuntimeError("No Phase 06 trajectories selected for Phase 07 tuning.")
    return tasks


def build_candidates(args: argparse.Namespace) -> list[ParameterCandidate]:
    grid = itertools.product(
        parse_csv_ints(args.min_distance_frames),
        parse_csv_floats(args.prominence_px),
        parse_csv_ints(args.peak_radius),
        parse_csv_floats(args.min_angle_degrees),
        parse_csv_floats(args.min_step_px),
        parse_csv_floats(args.serve_slope_px_per_frame),
    )
    candidates = [
        ParameterCandidate(
            candidate_id=index,
            min_distance_frames=min_distance_frames,
            prominence_px=prominence_px,
            peak_radius=peak_radius,
            min_angle_degrees=min_angle_degrees,
            min_step_px=min_step_px,
            serve_slope_px_per_frame=serve_slope_px_per_frame,
        )
        for index, (
            min_distance_frames,
            prominence_px,
            peak_radius,
            min_angle_degrees,
            min_step_px,
            serve_slope_px_per_frame,
        ) in enumerate(grid, start=1)
    ]
    if not candidates:
        raise ValueError("Parameter grid is empty.")
    return candidates


def empty_stats(candidate: ParameterCandidate, split: str, tolerance_frames: int) -> dict[str, Any]:
    row = {
        "candidate_id": candidate.candidate_id,
        "split": split,
        "tolerance_frames": tolerance_frames,
        "rally_count": 0,
        "predicted_events": 0,
        "ground_truth_hits": 0,
        "matched_ground_truth": 0,
        "missed_ground_truth": 0,
        "extra_predictions": 0,
        "precision": "0.000000",
        "recall": "0.000000",
        "f1": "0.000000",
        "mean_abs_error": "",
    }
    row.update(candidate.as_runtime_kwargs())
    return row


def evaluate_candidate(
    candidate: ParameterCandidate,
    tasks: list[TuneTask],
    split: str,
    tolerance_frames: int,
) -> dict[str, Any]:
    split_tasks = [task for task in tasks if task.split == split]
    if not split_tasks:
        return empty_stats(candidate, split=split, tolerance_frames=tolerance_frames)

    predicted_events = 0
    ground_truth_hits = 0
    matched_ground_truth = 0
    missed_ground_truth = 0
    extra_predictions = 0
    errors: list[int] = []
    for task in split_tasks:
        events = p7.detect_events(task.points, **candidate.as_runtime_kwargs())
        matches, matched_predictions, missed = p7.greedy_match(
            video_id=task.video_id,
            rally_id=task.rally_id,
            events=events,
            gt_rows=task.gt_rows,
            tolerance_frames=tolerance_frames,
        )
        predicted_events += len(events)
        ground_truth_hits += len(task.gt_rows)
        matched_ground_truth += len(matches)
        missed_ground_truth += missed
        extra_predictions += max(len(events) - matched_predictions, 0)
        errors.extend(p7.to_int(match["abs_error"]) for match in matches)

    precision = matched_ground_truth / predicted_events if predicted_events else 0.0
    recall = matched_ground_truth / ground_truth_hits if ground_truth_hits else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    row = {
        "candidate_id": candidate.candidate_id,
        "split": split,
        "tolerance_frames": tolerance_frames,
        "rally_count": len(split_tasks),
        "predicted_events": predicted_events,
        "ground_truth_hits": ground_truth_hits,
        "matched_ground_truth": matched_ground_truth,
        "missed_ground_truth": missed_ground_truth,
        "extra_predictions": extra_predictions,
        "precision": f"{precision:.6f}",
        "recall": f"{recall:.6f}",
        "f1": f"{f1:.6f}",
        "mean_abs_error": f"{mean(errors):.3f}" if errors else "",
    }
    row.update(candidate.as_runtime_kwargs())
    return row


def best_validation_row(rows: list[dict[str, Any]], primary_metric: str) -> dict[str, Any] | None:
    val_rows = [row for row in rows if row["split"] == "val" and int(row["rally_count"]) > 0]
    if not val_rows:
        return None

    def score(row: dict[str, Any]) -> tuple[float, float, float, int]:
        mean_abs_error = row["mean_abs_error"]
        mae = float(mean_abs_error) if mean_abs_error not in {"", None} else 1e9
        return (
            float(row[primary_metric]),
            -mae,
            -float(row["extra_predictions"]),
            -int(row["candidate_id"]),
        )

    return max(val_rows, key=score)


def selected_params_from_row(row: dict[str, Any]) -> dict[str, int | float]:
    return {
        "min_distance_frames": int(row["min_distance_frames"]),
        "prominence_px": float(row["prominence_px"]),
        "peak_radius": int(row["peak_radius"]),
        "min_angle_degrees": float(row["min_angle_degrees"]),
        "min_step_px": float(row["min_step_px"]),
        "serve_slope_px_per_frame": float(row["serve_slope_px_per_frame"]),
        "tolerance_frames": int(row["tolerance_frames"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 07 parameter search over Phase 06 shuttle trajectories.")
    parser.add_argument(
        "--shuttle-root",
        type=Path,
        action="append",
        default=[],
        help="Phase 06 output root. Repeat for train/val/test roots. Defaults to project/outputs/shuttle.",
    )
    parser.add_argument("--ground-truth-table", type=Path, default=p7.DEFAULT_GT_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", default=[], help="Optional video id filter. Repeatable.")
    parser.add_argument("--train-video-ids", default="1-34")
    parser.add_argument("--val-video-ids", default="35-39")
    parser.add_argument("--test-video-ids", default="40-44")
    parser.add_argument("--max-rallies-per-split", type=int, default=0)
    parser.add_argument("--min-distance-frames", default="12,15,18")
    parser.add_argument("--prominence-px", default="8,12,16")
    parser.add_argument("--peak-radius", default="7")
    parser.add_argument("--min-angle-degrees", default="135,145,155")
    parser.add_argument("--min-step-px", default="5")
    parser.add_argument("--serve-slope-px-per-frame", default="5")
    parser.add_argument("--tolerance-frames", type=int, default=15)
    parser.add_argument("--primary-metric", choices=["precision", "recall", "f1"], default="f1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rallies_per_split < 0:
        raise ValueError("--max-rallies-per-split must be >= 0")

    tasks = load_tasks(args)
    candidates = build_candidates(args)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for split in ("train", "val", "test"):
            rows.append(
                evaluate_candidate(
                    candidate,
                    tasks=tasks,
                    split=split,
                    tolerance_frames=args.tolerance_frames,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    search_path = args.output_root / "phase07_parameter_search.csv"
    write_csv(search_path, rows, SEARCH_COLUMNS)

    selected = best_validation_row(rows, primary_metric=args.primary_metric)
    split_counts = {
        split: len([task for task in tasks if task.split == split])
        for split in ("train", "val", "test")
    }
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(p7.ROOT)),
        "phase06_manifests": [
            str(root / "phase06_shuttle_tracking_manifest.csv")
            for root in shuttle_roots(args)
        ],
        "ground_truth_table": str(args.ground_truth_table),
        "output_root": str(args.output_root),
        "candidate_count": len(candidates),
        "primary_metric": args.primary_metric,
        "split_counts": split_counts,
        "parameter_grid": {
            "min_distance_frames": parse_csv_ints(args.min_distance_frames),
            "prominence_px": parse_csv_floats(args.prominence_px),
            "peak_radius": parse_csv_ints(args.peak_radius),
            "min_angle_degrees": parse_csv_floats(args.min_angle_degrees),
            "min_step_px": parse_csv_floats(args.min_step_px),
            "serve_slope_px_per_frame": parse_csv_floats(args.serve_slope_px_per_frame),
            "tolerance_frames": args.tolerance_frames,
        },
        "outputs": {
            "parameter_search": str(search_path),
            "selected_params_protocol": str(args.output_root / "selected_params_protocol.json"),
        },
    }
    if selected is None:
        summary["selection_status"] = "blocked_missing_validation_split"
        summary["selected_params"] = None
        summary["reason"] = (
            "No validation-split Phase 06 trajectories were available. "
            "Generate Phase 06 trajectories for videos 35-39 before protocol selection."
        )
    else:
        summary["selection_status"] = "selected_from_validation_split"
        summary["selected_candidate_id"] = int(selected["candidate_id"])
        summary["selected_params"] = selected_params_from_row(selected)
        summary["validation_metrics"] = selected

    (args.output_root / "selected_params_protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
