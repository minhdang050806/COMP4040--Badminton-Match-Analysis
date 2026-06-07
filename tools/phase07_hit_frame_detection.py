from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHUTTLE_ROOT = ROOT / "project" / "outputs" / "shuttle"
DEFAULT_GT_TABLE = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
# Keep optional trajectory-detector outputs separate from the authoritative
# Phase 02-label-derived Phase 07 bundle under project/outputs/hit_frames.
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "hit_frames_detected"

EVENT_COLUMNS = [
    "Frame",
    "OriginalFrame",
    "TimeSec",
    "Visibility",
    "X",
    "Y",
    "event",
    "event_rank",
    "event_reason",
]

PREDICTION_COLUMNS = [
    "video_id",
    "rally_id",
    "event_rank",
    "Frame",
    "OriginalFrame",
    "TimeSec",
    "X",
    "Y",
    "event_reason",
]

WINDOW_COLUMNS = [
    "video_id",
    "rally_id",
    "event_rank",
    "event_frame_local",
    "event_frame_original",
    "window_start_local",
    "window_end_local",
    "window_start_original",
    "window_end_original",
    "context_start_original",
    "context_end_original",
    "prev_event_frame_original",
    "next_event_frame_original",
]

VALIDATION_COLUMNS = [
    "video_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "processed_frames",
    "trajectory_visible_rows",
    "trajectory_coverage",
    "predicted_events",
    "ground_truth_hits",
    "matched_predictions",
    "matched_ground_truth",
    "missed_ground_truth",
    "extra_predictions",
    "tolerance_frames",
    "mean_abs_error",
    "median_abs_error",
    "max_abs_error",
    "accuracy_at_tolerance",
]

MATCH_COLUMNS = [
    "video_id",
    "rally_id",
    "predicted_frame_original",
    "ground_truth_frame_num",
    "abs_error",
    "stable_key",
    "clip_id",
    "set_id",
    "gt_rally_id",
    "ball_round_id",
    "stroke_type_ground_truth",
    "player",
    "server",
]


@dataclass(frozen=True)
class TrajectoryPoint:
    index: int
    frame: int
    original_frame: int
    time_sec: str
    visible: bool
    x: float
    y: float


@dataclass(frozen=True)
class EventCandidate:
    original_frame: int
    local_frame: int
    point_index: int
    score: float
    reason: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str | int | float, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: str | int | float, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_trajectory(path: Path) -> list[TrajectoryPoint]:
    points: list[TrajectoryPoint] = []
    for index, row in enumerate(read_csv(path)):
        points.append(
            TrajectoryPoint(
                index=index,
                frame=to_int(row["Frame"]),
                original_frame=to_int(row.get("OriginalFrame", row["Frame"])),
                time_sec=row.get("TimeSec", ""),
                visible=to_int(row["Visibility"]) == 1,
                x=to_float(row["X"]),
                y=to_float(row["Y"]),
            )
        )
    return points


def local_prominence(values: list[float], index: int, radius: int) -> float:
    left = max(0, index - radius)
    right = min(len(values), index + radius + 1)
    if right - left <= 1:
        return 0.0
    baseline = min(min(values[left:index] or [values[index]]), min(values[index + 1 : right] or [values[index]]))
    return values[index] - baseline


def sparse_peak_candidates(
    visible_points: list[TrajectoryPoint],
    min_distance_frames: int,
    prominence_px: float,
    radius: int,
) -> list[EventCandidate]:
    if len(visible_points) < 3:
        return []
    ys = [p.y for p in visible_points]
    candidates: list[EventCandidate] = []
    for i in range(1, len(visible_points) - 1):
        prev_y = ys[i - 1]
        cur_y = ys[i]
        next_y = ys[i + 1]
        if cur_y < prev_y or cur_y < next_y:
            continue
        prominence = local_prominence(ys, i, radius=radius)
        if prominence < prominence_px:
            continue
        point = visible_points[i]
        candidates.append(
            EventCandidate(
                original_frame=point.original_frame,
                local_frame=point.frame,
                point_index=point.index,
                score=prominence,
                reason="y_peak",
            )
        )
    return suppress_close_candidates(candidates, min_distance_frames=min_distance_frames)


def angle_degrees(a: TrajectoryPoint, b: TrajectoryPoint, c: TrajectoryPoint) -> float:
    v1 = (b.x - a.x, b.y - a.y)
    v2 = (c.x - b.x, c.y - b.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def kink_candidates(
    visible_points: list[TrajectoryPoint],
    min_distance_frames: int,
    min_angle_degrees: float,
    min_step_px: float,
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    for i in range(1, len(visible_points) - 1):
        a = visible_points[i - 1]
        b = visible_points[i]
        c = visible_points[i + 1]
        if b.original_frame - a.original_frame > min_distance_frames:
            continue
        if c.original_frame - b.original_frame > min_distance_frames:
            continue
        step_before = math.hypot(b.x - a.x, b.y - a.y)
        step_after = math.hypot(c.x - b.x, c.y - b.y)
        if min(step_before, step_after) < min_step_px:
            continue
        angle = angle_degrees(a, b, c)
        if angle < min_angle_degrees:
            continue
        candidates.append(
            EventCandidate(
                original_frame=b.original_frame,
                local_frame=b.frame,
                point_index=b.index,
                score=angle,
                reason="trajectory_kink",
            )
        )
    return suppress_close_candidates(candidates, min_distance_frames=min_distance_frames)


def serve_candidate(visible_points: list[TrajectoryPoint], slope_px_per_frame: float) -> list[EventCandidate]:
    for current, nxt in zip(visible_points, visible_points[1:]):
        frame_gap = max(nxt.original_frame - current.original_frame, 1)
        upward_speed = (current.y - nxt.y) / frame_gap
        if upward_speed >= slope_px_per_frame:
            return [
                EventCandidate(
                    original_frame=current.original_frame,
                    local_frame=current.frame,
                    point_index=current.index,
                    score=upward_speed,
                    reason="serve_drop",
                )
            ]
    return []


def suppress_close_candidates(candidates: list[EventCandidate], min_distance_frames: int) -> list[EventCandidate]:
    selected: list[EventCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(abs(candidate.original_frame - kept.original_frame) >= min_distance_frames for kept in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.original_frame)


def detect_events(
    points: list[TrajectoryPoint],
    min_distance_frames: int,
    prominence_px: float,
    peak_radius: int,
    min_angle_degrees: float,
    min_step_px: float,
    serve_slope_px_per_frame: float,
) -> list[EventCandidate]:
    visible_points = [point for point in points if point.visible and point.x > 0 and point.y > 0]
    candidates: list[EventCandidate] = []
    candidates.extend(serve_candidate(visible_points, slope_px_per_frame=serve_slope_px_per_frame))
    candidates.extend(
        sparse_peak_candidates(
            visible_points,
            min_distance_frames=min_distance_frames,
            prominence_px=prominence_px,
            radius=peak_radius,
        )
    )
    candidates.extend(
        kink_candidates(
            visible_points,
            min_distance_frames=min_distance_frames,
            min_angle_degrees=min_angle_degrees,
            min_step_px=min_step_px,
        )
    )
    return suppress_close_candidates(candidates, min_distance_frames=min_distance_frames)


def event_rows(points: list[TrajectoryPoint], events: list[EventCandidate]) -> list[dict[str, Any]]:
    event_by_frame = {event.original_frame: event for event in events}
    rank_by_frame = {event.original_frame: rank for rank, event in enumerate(events, start=1)}
    rows: list[dict[str, Any]] = []
    for point in points:
        event = event_by_frame.get(point.original_frame)
        rows.append(
            {
                "Frame": point.frame,
                "OriginalFrame": point.original_frame,
                "TimeSec": point.time_sec,
                "Visibility": int(point.visible),
                "X": point.x,
                "Y": point.y,
                "event": 1 if event else 0,
                "event_rank": rank_by_frame.get(point.original_frame, ""),
                "event_reason": event.reason if event else "",
            }
        )
    return rows


def prediction_rows(video_id: str, rally_id: str, events: list[EventCandidate], points_by_original: dict[int, TrajectoryPoint]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, event in enumerate(events, start=1):
        point = points_by_original[event.original_frame]
        rows.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "event_rank": rank,
                "Frame": event.local_frame,
                "OriginalFrame": event.original_frame,
                "TimeSec": point.time_sec,
                "X": point.x,
                "Y": point.y,
                "event_reason": event.reason,
            }
        )
    return rows


def build_windows(
    video_id: str,
    rally_id: str,
    events: list[EventCandidate],
    start_frame: int,
    end_frame: int,
    context_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    originals = [event.original_frame for event in events]
    for index, event in enumerate(events):
        prev_event = originals[index - 1] if index > 0 else ""
        next_event = originals[index + 1] if index + 1 < len(originals) else ""
        midpoint_start = start_frame if prev_event == "" else int(round((int(prev_event) + event.original_frame) / 2))
        midpoint_end = end_frame if next_event == "" else int(round((event.original_frame + int(next_event)) / 2))
        rows.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "event_rank": index + 1,
                "event_frame_local": event.local_frame,
                "event_frame_original": event.original_frame,
                "window_start_local": max(0, midpoint_start - start_frame),
                "window_end_local": max(0, midpoint_end - start_frame),
                "window_start_original": midpoint_start,
                "window_end_original": midpoint_end,
                "context_start_original": max(start_frame, event.original_frame - context_frames),
                "context_end_original": min(end_frame, event.original_frame + context_frames),
                "prev_event_frame_original": prev_event,
                "next_event_frame_original": next_event,
            }
        )
    return rows


def load_ground_truth(path: Path) -> dict[str, list[dict[str, str]]]:
    by_match: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return by_match
    for row in read_csv(path):
        by_match[row["match_id"]].append(row)
    return by_match


def ground_truth_in_interval(rows: list[dict[str, str]], start_frame: int, end_frame: int) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if start_frame <= to_int(row.get("frame_num_int", row.get("frame_num", "")), default=-1) <= end_frame
    ]


def greedy_match(
    video_id: str,
    rally_id: str,
    events: list[EventCandidate],
    gt_rows: list[dict[str, str]],
    tolerance_frames: int,
) -> tuple[list[dict[str, Any]], int, int]:
    unmatched_gt = set(range(len(gt_rows)))
    matches: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: item.original_frame):
        best_index = None
        best_error = tolerance_frames + 1
        for index in list(unmatched_gt):
            gt_frame = to_int(gt_rows[index].get("frame_num_int", gt_rows[index].get("frame_num", "")), default=-1)
            error = abs(event.original_frame - gt_frame)
            if error < best_error:
                best_error = error
                best_index = index
        if best_index is None or best_error > tolerance_frames:
            continue
        row = gt_rows[best_index]
        unmatched_gt.remove(best_index)
        matches.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "predicted_frame_original": event.original_frame,
                "ground_truth_frame_num": to_int(row.get("frame_num_int", row.get("frame_num", ""))),
                "abs_error": best_error,
                "stable_key": row.get("stable_key", ""),
                "clip_id": row.get("clip_id", ""),
                "set_id": row.get("set_id", ""),
                "gt_rally_id": row.get("rally_id", ""),
                "ball_round_id": row.get("ball_round_id", ""),
                "stroke_type_ground_truth": row.get("stroke_type_ground_truth", row.get("type", "")),
                "player": row.get("player", ""),
                "server": row.get("server", ""),
            }
        )
    matched_predictions = len(matches)
    missed_ground_truth = len(unmatched_gt)
    return matches, matched_predictions, missed_ground_truth


def summarize_errors(errors: list[int]) -> tuple[str, str, str]:
    if not errors:
        return "", "", ""
    return f"{mean(errors):.3f}", f"{median(errors):.3f}", str(max(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 07 hit-frame detection from Phase 06 shuttle trajectories.")
    parser.add_argument("--shuttle-root", type=Path, default=DEFAULT_SHUTTLE_ROOT)
    parser.add_argument("--ground-truth-table", type=Path, default=DEFAULT_GT_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Optional selected_params_protocol.json from phase07_tune_hit_frame_parameters.py.",
    )
    parser.add_argument("--video-id", action="append", default=[], help="Optional video id filter. Repeatable.")
    parser.add_argument("--rally-id", action="append", default=[], help="Optional Phase 06 rally id filter. Repeatable.")
    parser.add_argument("--max-rallies", type=int, default=0, help="Maximum manifest rows to process; 0 means all selected.")
    parser.add_argument("--min-distance-frames", type=int, default=15)
    parser.add_argument("--prominence-px", type=float, default=12.0)
    parser.add_argument("--peak-radius", type=int, default=7)
    parser.add_argument("--min-angle-degrees", type=float, default=145.0)
    parser.add_argument("--min-step-px", type=float, default=5.0)
    parser.add_argument("--serve-slope-px-per-frame", type=float, default=5.0)
    parser.add_argument("--context-frames", type=int, default=15)
    parser.add_argument("--tolerance-frames", type=int, default=15)
    return parser.parse_args()


def apply_params_file(args: argparse.Namespace) -> None:
    if args.params_file is None:
        return
    if not args.params_file.exists():
        raise FileNotFoundError(f"Missing Phase 07 params file: {args.params_file}")
    payload = json.loads(args.params_file.read_text(encoding="utf-8"))
    if payload.get("selection_status") not in {None, "selected_from_validation_split"}:
        raise ValueError(
            f"Params file is not a selected protocol artifact: selection_status={payload.get('selection_status')!r}"
        )
    selected_params = payload.get("selected_params", payload)
    for name in [
        "min_distance_frames",
        "prominence_px",
        "peak_radius",
        "min_angle_degrees",
        "min_step_px",
        "serve_slope_px_per_frame",
        "tolerance_frames",
    ]:
        if name in selected_params:
            setattr(args, name, selected_params[name])


def main() -> None:
    args = parse_args()
    apply_params_file(args)
    manifest_path = args.shuttle_root / "phase06_shuttle_tracking_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase 06 manifest: {manifest_path}")
    if args.max_rallies < 0:
        raise ValueError("--max-rallies must be >= 0")
    if args.min_distance_frames < 1:
        raise ValueError("--min-distance-frames must be >= 1")

    selected_video_ids = {str(video_id) for video_id in args.video_id}
    selected_rally_ids = {str(rally_id) for rally_id in args.rally_id}
    manifest = []
    for row in read_csv(manifest_path):
        if selected_video_ids and row["video_id"] not in selected_video_ids:
            continue
        if selected_rally_ids and row["rally_id"] not in selected_rally_ids:
            continue
        manifest.append(row)
        if args.max_rallies > 0 and len(manifest) >= args.max_rallies:
            break
    if not manifest:
        raise RuntimeError("No Phase 06 trajectories selected for Phase 07.")

    gt_by_match = load_ground_truth(args.ground_truth_table)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_predictions: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    event_count_by_video: dict[str, int] = defaultdict(int)
    gt_count_by_video: dict[str, int] = defaultdict(int)
    matched_count_by_video: dict[str, int] = defaultdict(int)

    for manifest_row in manifest:
        video_id = manifest_row["video_id"]
        rally_id = manifest_row["rally_id"]
        start_frame = to_int(manifest_row["start_frame"])
        end_frame = to_int(manifest_row["end_frame"])
        denoised_csv = Path(manifest_row["denoised_csv"])
        if not denoised_csv.is_absolute():
            denoised_csv = ROOT / denoised_csv
        points = load_trajectory(denoised_csv)
        points_by_original = {point.original_frame: point for point in points}
        events = detect_events(
            points,
            min_distance_frames=args.min_distance_frames,
            prominence_px=args.prominence_px,
            peak_radius=args.peak_radius,
            min_angle_degrees=args.min_angle_degrees,
            min_step_px=args.min_step_px,
            serve_slope_px_per_frame=args.serve_slope_px_per_frame,
        )
        video_dir = args.output_root / video_id
        event_csv = video_dir / f"{rally_id}_events.csv"
        write_csv(event_csv, event_rows(points, events), EVENT_COLUMNS)

        predictions = prediction_rows(video_id, rally_id, events, points_by_original)
        windows = build_windows(
            video_id=video_id,
            rally_id=rally_id,
            events=events,
            start_frame=start_frame,
            end_frame=end_frame,
            context_frames=args.context_frames,
        )
        all_predictions.extend(predictions)
        all_windows.extend(windows)

        gt_rows = ground_truth_in_interval(gt_by_match.get(video_id, []), start_frame=start_frame, end_frame=end_frame)
        matches, matched_predictions, missed_ground_truth = greedy_match(
            video_id=video_id,
            rally_id=rally_id,
            events=events,
            gt_rows=gt_rows,
            tolerance_frames=args.tolerance_frames,
        )
        match_rows.extend(matches)
        errors = [to_int(row["abs_error"]) for row in matches]
        mean_abs_error, median_abs_error, max_abs_error = summarize_errors(errors)
        visible_rows = sum(1 for point in points if point.visible)
        processed_frames = len(points)
        validation_rows.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "processed_frames": processed_frames,
                "trajectory_visible_rows": visible_rows,
                "trajectory_coverage": f"{visible_rows / processed_frames:.6f}" if processed_frames else "0.000000",
                "predicted_events": len(events),
                "ground_truth_hits": len(gt_rows),
                "matched_predictions": matched_predictions,
                "matched_ground_truth": len(matches),
                "missed_ground_truth": missed_ground_truth,
                "extra_predictions": max(len(events) - matched_predictions, 0),
                "tolerance_frames": args.tolerance_frames,
                "mean_abs_error": mean_abs_error,
                "median_abs_error": median_abs_error,
                "max_abs_error": max_abs_error,
                "accuracy_at_tolerance": f"{len(matches) / len(gt_rows):.6f}" if gt_rows else "",
            }
        )
        event_count_by_video[video_id] += len(events)
        gt_count_by_video[video_id] += len(gt_rows)
        matched_count_by_video[video_id] += len(matches)

    write_csv(args.output_root / "phase07_hit_frame_predictions.csv", all_predictions, PREDICTION_COLUMNS)
    write_csv(args.output_root / "phase07_stroke_windows.csv", all_windows, WINDOW_COLUMNS)
    write_csv(args.output_root / "phase07_validation.csv", validation_rows, VALIDATION_COLUMNS)
    write_csv(args.output_root / "phase07_matches.csv", match_rows, MATCH_COLUMNS)

    all_errors = [to_int(row["abs_error"]) for row in match_rows]
    mean_abs_error, median_abs_error, max_abs_error = summarize_errors(all_errors)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "phase06_manifest": str(manifest_path),
        "ground_truth_table": str(args.ground_truth_table),
        "runtime_config": {
            "shuttle_root": str(args.shuttle_root),
            "output_root": str(args.output_root),
            "video_ids": [str(video_id) for video_id in args.video_id],
            "rally_ids": [str(rally_id) for rally_id in args.rally_id],
            "max_rallies": args.max_rallies,
            "min_distance_frames": args.min_distance_frames,
            "prominence_px": args.prominence_px,
            "peak_radius": args.peak_radius,
            "min_angle_degrees": args.min_angle_degrees,
            "min_step_px": args.min_step_px,
            "serve_slope_px_per_frame": args.serve_slope_px_per_frame,
            "context_frames": args.context_frames,
            "tolerance_frames": args.tolerance_frames,
        },
        "rally_count": len(manifest),
        "predicted_events": len(all_predictions),
        "stroke_windows": len(all_windows),
        "ground_truth_hits_in_intervals": int(sum(int(row["ground_truth_hits"]) for row in validation_rows)),
        "matched_ground_truth": len(match_rows),
        "missed_ground_truth": int(sum(int(row["missed_ground_truth"]) for row in validation_rows)),
        "extra_predictions": int(sum(int(row["extra_predictions"]) for row in validation_rows)),
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
        "max_abs_error": max_abs_error,
        "accuracy_at_tolerance": (
            len(match_rows) / int(sum(int(row["ground_truth_hits"]) for row in validation_rows))
            if validation_rows and int(sum(int(row["ground_truth_hits"]) for row in validation_rows)) > 0
            else 0.0
        ),
        "by_video": {
            video_id: {
                "predicted_events": event_count_by_video[video_id],
                "ground_truth_hits": gt_count_by_video[video_id],
                "matched_ground_truth": matched_count_by_video[video_id],
            }
            for video_id in sorted(event_count_by_video, key=lambda item: int(item) if item.isdigit() else item)
        },
        "outputs": {
            "predictions": str(args.output_root / "phase07_hit_frame_predictions.csv"),
            "windows": str(args.output_root / "phase07_stroke_windows.csv"),
            "validation": str(args.output_root / "phase07_validation.csv"),
            "matches": str(args.output_root / "phase07_matches.csv"),
        },
    }
    (args.output_root / "phase07_hit_frame_detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
