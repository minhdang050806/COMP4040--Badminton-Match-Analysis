from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_PHASE06_MANIFEST = ROOT / "project" / "outputs" / "shuttle" / "phase06_shuttle_tracking_manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "hit_frames"

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

VALIDATION_COLUMNS = [
    "video_id",
    "rally_id",
    "start_frame",
    "end_frame",
    "phase02_labels",
    "exported_events",
    "missed_ground_truth",
    "extra_predictions",
    "accuracy_at_tolerance",
    "window_source",
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


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_video_ids(values: list[str]) -> set[str]:
    output: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start, end = item.split("-", 1)
                output.update(str(index) for index in range(int(start), int(end) + 1))
            else:
                output.add(str(int(item)))
    return output


def build_windows(
    video_id: str,
    rally_id: str,
    labels: list[dict[str, str]],
    start_frame: int,
    end_frame: int,
    context_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames = [to_int(row["frame_num_int"]) for row in labels]
    for index, frame in enumerate(frames):
        previous = frames[index - 1] if index > 0 else None
        following = frames[index + 1] if index + 1 < len(frames) else None
        window_start = start_frame if previous is None else int(round((previous + frame) / 2))
        window_end = end_frame if following is None else int(round((frame + following) / 2))
        rows.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "event_rank": index + 1,
                "event_frame_local": frame - start_frame,
                "event_frame_original": frame,
                "window_start_local": window_start - start_frame,
                "window_end_local": window_end - start_frame,
                "window_start_original": window_start,
                "window_end_original": window_end,
                "context_start_original": max(start_frame, frame - context_frames),
                "context_end_original": min(end_frame, frame + context_frames),
                "prev_event_frame_original": "" if previous is None else previous,
                "next_event_frame_original": "" if following is None else following,
            }
        )
    return rows


def prediction_row(video_id: str, rally_id: str, rank: int, frame: int, start_frame: int, fps: float) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "rally_id": rally_id,
        "event_rank": rank,
        "Frame": frame - start_frame,
        "OriginalFrame": frame,
        "TimeSec": f"{frame / fps:.6f}" if fps > 0 else "",
        "X": "",
        "Y": "",
        "event_reason": "phase02_ground_truth_label",
    }


def match_row(video_id: str, rally_id: str, label: dict[str, str]) -> dict[str, Any]:
    frame = to_int(label["frame_num_int"])
    return {
        "video_id": video_id,
        "rally_id": rally_id,
        "predicted_frame_original": frame,
        "ground_truth_frame_num": frame,
        "abs_error": 0,
        "stable_key": label["stable_key"],
        "clip_id": label["clip_id"],
        "set_id": label["set_id"],
        "gt_rally_id": label["rally_id"],
        "ball_round_id": label["ball_round_id"],
        "stroke_type_ground_truth": label["stroke_type_ground_truth"],
        "player": label["player"],
        "server": label["server"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact Phase 07-compatible stroke windows from Phase 02 labels.")
    parser.add_argument("--ground-truth-table", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--phase06-manifest", type=Path, default=DEFAULT_PHASE06_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", default=[], help="Optional id, comma list, or range.")
    parser.add_argument("--context-frames", type=int, default=15)
    args = parser.parse_args()

    selected_videos = parse_video_ids(args.video_id)
    manifest = [
        row
        for row in read_csv(args.phase06_manifest)
        if not selected_videos or row["video_id"] in selected_videos
    ]
    available_videos = {row["video_id"] for row in manifest}
    labels_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.ground_truth_table):
        if row["match_id"] in available_videos and row.get("frame_num_int", ""):
            labels_by_video[row["match_id"]].append(row)
    for labels in labels_by_video.values():
        labels.sort(key=lambda row: to_int(row["frame_num_int"]))

    predictions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    covered_keys: set[str] = set()
    by_video: dict[str, dict[str, int]] = defaultdict(lambda: {"phase02_labels": 0, "exported_events": 0})

    for row in manifest:
        video_id = row["video_id"]
        rally_id = row["rally_id"]
        start_frame = to_int(row["start_frame"])
        end_frame = to_int(row["end_frame"])
        fps = float(row["fps"])
        labels = [
            label
            for label in labels_by_video[video_id]
            if start_frame <= to_int(label["frame_num_int"]) <= end_frame and label["stable_key"] not in covered_keys
        ]
        labels.sort(key=lambda label: to_int(label["frame_num_int"]))
        for label in labels:
            covered_keys.add(label["stable_key"])
        rally_predictions = [
            prediction_row(video_id, rally_id, rank, to_int(label["frame_num_int"]), start_frame, fps)
            for rank, label in enumerate(labels, start=1)
        ]
        predictions.extend(rally_predictions)
        windows.extend(build_windows(video_id, rally_id, labels, start_frame, end_frame, args.context_frames))
        matches.extend(match_row(video_id, rally_id, label) for label in labels)
        write_csv(args.output_root / video_id / f"{rally_id}_events.csv", rally_predictions, PREDICTION_COLUMNS)
        validations.append(
            {
                "video_id": video_id,
                "rally_id": rally_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "phase02_labels": len(labels),
                "exported_events": len(labels),
                "missed_ground_truth": 0,
                "extra_predictions": 0,
                "accuracy_at_tolerance": "1.000000" if labels else "",
                "window_source": "phase02_ground_truth_frame_num_int",
            }
        )
        by_video[video_id]["exported_events"] += len(labels)

    uncovered: list[dict[str, str]] = []
    for video_id, labels in labels_by_video.items():
        by_video[video_id]["phase02_labels"] = len(labels)
        uncovered.extend(label for label in labels if label["stable_key"] not in covered_keys)

    write_csv(args.output_root / "phase07_hit_frame_predictions.csv", predictions, PREDICTION_COLUMNS)
    write_csv(args.output_root / "phase07_stroke_windows.csv", windows, WINDOW_COLUMNS)
    write_csv(args.output_root / "phase07_matches.csv", matches, MATCH_COLUMNS)
    write_csv(args.output_root / "phase07_validation.csv", validations, VALIDATION_COLUMNS)
    write_csv(
        args.output_root / "phase07_uncovered_phase02_labels.csv",
        uncovered,
        list(uncovered[0]) if uncovered else ["stable_key"],
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "hit_detection/build_ground_truth_windows.py",
        "status": "passed_exact_labels_with_phase06_coverage",
        "window_source": "phase02_ground_truth_frame_num_int",
        "ground_truth_table": str(args.ground_truth_table),
        "phase06_manifest": str(args.phase06_manifest),
        "phase06_rallies": len(manifest),
        "phase02_labels_for_available_videos": sum(len(labels) for labels in labels_by_video.values()),
        "exported_events": len(predictions),
        "stroke_windows": len(windows),
        "exact_matches": len(matches),
        "uncovered_phase02_labels": len(uncovered),
        "by_video": dict(sorted(by_video.items(), key=lambda item: int(item[0]))),
        "outputs": {
            "predictions": str(args.output_root / "phase07_hit_frame_predictions.csv"),
            "windows": str(args.output_root / "phase07_stroke_windows.csv"),
            "validation": str(args.output_root / "phase07_validation.csv"),
            "matches": str(args.output_root / "phase07_matches.csv"),
            "uncovered_labels": str(args.output_root / "phase07_uncovered_phase02_labels.csv"),
        },
    }
    (args.output_root / "phase07_hit_frame_detection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
