from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
# Phase 07's authoritative known-ShuttleSet bundle is derived from exact Phase 02 labels.
DEFAULT_WINDOWS = ROOT / "project" / "outputs" / "hit_frames" / "phase07_stroke_windows.csv"
DEFAULT_MATCHES = ROOT / "project" / "outputs" / "hit_frames" / "phase07_matches.csv"
DEFAULT_PHASE06_MANIFEST = ROOT / "project" / "outputs" / "shuttle" / "phase06_shuttle_tracking_manifest.csv"
DEFAULT_HOMOGRAPHY_CSV = ROOT / "project" / "dataset" / "ShuttleSet" / "set" / "homography.csv"
DEFAULT_POSE_WEIGHTS = ROOT / "project" / "weights" / "yolo26x-pose.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "features_yolo26x_tracked_phase02"
DEFAULT_REFERENCE_COLLATED = ROOT / "project" / "outputs" / "bst_collated" / "merged_seq100_between_2_hits_with_max_limits"

MANIFEST_COLUMNS = [
    "video_id",
    "rally_id",
    "event_rank",
    "clip_id",
    "feature_source",
    "window_strategy",
    "player_order",
    "event_frame_original",
    "prev_event_frame_original",
    "next_event_frame_original",
    "event_offset_frames",
    "window_start_original",
    "window_end_original",
    "frame_count",
    "joints_shape",
    "pos_shape",
    "shuttle_shape",
    "joints_npy",
    "pos_npy",
    "shuttle_npy",
    "validity_npz",
    "source_video",
    "worker_device",
]

VALIDATION_COLUMNS = [
    "video_id",
    "rally_id",
    "event_rank",
    "clip_id",
    "frame_count",
    "valid_shape",
    "p1_pose_observed_rate",
    "p2_pose_observed_rate",
    "p1_pose_valid_rate",
    "p2_pose_valid_rate",
    "p1_position_observed_rate",
    "p2_position_observed_rate",
    "p1_position_valid_rate",
    "p2_position_valid_rate",
    "both_pose_missing_frames",
    "both_pose_missing_rate",
    "both_position_missing_frames",
    "both_position_missing_rate",
    "pose_recovered_values",
    "position_recovered_values",
    "shuttle_visible_rate",
    "pose_tensor_zero_rate",
    "position_tensor_zero_rate",
    "player_pos_out_of_filter_range",
    "rejected_out_of_range_candidates",
    "player_side_inconsistent_frames",
    "player_side_inconsistent_rate",
    "problem",
    "fix",
]


@dataclass(frozen=True)
class StrokeWindow:
    video_id: str
    rally_id: str
    event_rank: str
    event_frame_original: int
    window_start_original: int
    window_end_original: int
    prev_event_frame_original: int | None
    next_event_frame_original: int | None


@dataclass(frozen=True)
class Phase06Task:
    source_video: Path
    denoised_csv: Path
    width: int
    height: int
    fps: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class HomographyInfo:
    matrix: np.ndarray
    border_left: float
    border_right: float
    border_up: float
    border_down: float


@dataclass(frozen=True)
class Candidate:
    keypoints: np.ndarray  # (17, 2), camera pixels
    bbox: np.ndarray  # (4,), xyxy camera pixels
    court_position: np.ndarray  # (2,), normalized court xy
    pose_valid: bool
    area: float


@dataclass
class FeatureBundle:
    joints: np.ndarray  # (T, 2, 17, 2)
    positions: np.ndarray  # (T, 2, 2)
    shuttle: np.ndarray  # (T, 2)
    pose_observed: np.ndarray  # (T, 2)
    position_observed: np.ndarray  # (T, 2)
    pose_valid: np.ndarray  # (T, 2), includes recovered values
    position_valid: np.ndarray  # (T, 2), includes recovered values
    rejected_out_of_range_candidates: int = 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def load_windows(args: argparse.Namespace) -> list[StrokeWindow]:
    selected_videos = parse_video_ids(args.video_id)
    selected_rallies = set(args.rally_id)
    windows: list[StrokeWindow] = []
    for row in read_csv(args.stroke_windows):
        if selected_videos and row["video_id"] not in selected_videos:
            continue
        if selected_rallies and row["rally_id"] not in selected_rallies:
            continue
        windows.append(
            StrokeWindow(
                video_id=row["video_id"],
                rally_id=row["rally_id"],
                event_rank=row["event_rank"],
                event_frame_original=to_int(row["event_frame_original"]),
                window_start_original=to_int(row["window_start_original"]),
                window_end_original=to_int(row["window_end_original"]),
                prev_event_frame_original=(
                    to_int(row["prev_event_frame_original"]) if row.get("prev_event_frame_original", "") != "" else None
                ),
                next_event_frame_original=(
                    to_int(row["next_event_frame_original"]) if row.get("next_event_frame_original", "") != "" else None
                ),
            )
        )
    if args.max_clips > 0:
        windows = windows[: args.max_clips]
    if not windows:
        raise RuntimeError("No Phase 09 windows selected.")
    if args.worker_count > 1:
        windows = [window for index, window in enumerate(windows) if index % args.worker_count == args.worker_index]
    return windows


def load_match_clip_ids(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.exists():
        return {}
    output: dict[tuple[str, str, int], str] = {}
    for row in read_csv(path):
        output[(row["video_id"], row["rally_id"], to_int(row["predicted_frame_original"]))] = row["clip_id"]
    return output


def clip_id_for_window(window: StrokeWindow, matched_ids: dict[tuple[str, str, int], str]) -> str:
    return matched_ids.get(
        (window.video_id, window.rally_id, window.event_frame_original),
        f"{window.video_id}_{window.rally_id}_event_{window.event_rank}",
    )


def load_phase06_tasks(path: Path) -> dict[tuple[str, str], Phase06Task]:
    tasks: dict[tuple[str, str], Phase06Task] = {}
    for row in read_csv(path):
        tasks[(row["video_id"], row["rally_id"])] = Phase06Task(
            source_video=resolve_path(row["source_video"]),
            denoised_csv=resolve_path(row["denoised_csv"]),
            width=to_int(row["width"]),
            height=to_int(row["height"]),
            fps=to_float(row["fps"], 30.0),
            start_frame=to_int(row["start_frame"]),
            end_frame=to_int(row["end_frame"]),
        )
    return tasks


def rebuild_temporal_window(window: StrokeWindow, task: Phase06Task, strategy: str) -> StrokeWindow:
    if strategy == "midpoint":
        return window
    if strategy != "bst-compatible":
        raise ValueError(f"Unsupported window strategy: {strategy}")

    # Matches BST ShuttleSet/gen_my_dataset.py: previous hit -> next hit + fps/4,
    # with half-second rally-edge fallback and 1.5-second maximum context.
    half_second = int(task.fps // 2)
    extension = int(half_second // 2)
    limit = int(task.fps * 3 // 2)
    current = window.event_frame_original
    start = window.prev_event_frame_original if window.prev_event_frame_original is not None else current - half_second
    end_exclusive = (
        window.next_event_frame_original + extension
        if window.next_event_frame_original is not None
        else current + half_second
    )
    start = max(start, current - limit, 0)
    end_exclusive = min(end_exclusive, current + limit + extension)
    if end_exclusive <= start:
        end_exclusive = start + 1
    return StrokeWindow(
        video_id=window.video_id,
        rally_id=window.rally_id,
        event_rank=window.event_rank,
        event_frame_original=current,
        window_start_original=start,
        window_end_original=end_exclusive - 1,
        prev_event_frame_original=window.prev_event_frame_original,
        next_event_frame_original=window.next_event_frame_original,
    )


def project_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_xy.astype(np.float64), np.ones((1, points_xy.shape[1]))], axis=0)
    projected = matrix @ homogeneous
    return projected[:2] / projected[2:3]


def load_homographies(path: Path) -> dict[str, HomographyInfo]:
    output: dict[str, HomographyInfo] = {}
    for row in read_csv(path):
        matrix = np.asarray(ast.literal_eval(row["homography_matrix"]), dtype=np.float64)
        corners = np.asarray(
            [
                [to_float(row["upleft_x"]), to_float(row["upright_x"]), to_float(row["downleft_x"]), to_float(row["downright_x"])],
                [to_float(row["upleft_y"]), to_float(row["upright_y"]), to_float(row["downleft_y"]), to_float(row["downright_y"])],
            ],
            dtype=np.float64,
        )
        court = project_points(matrix, corners)
        output[row["id"]] = HomographyInfo(
            matrix=matrix,
            border_left=float(court[0, 0]),
            border_right=float(court[0, 1]),
            border_up=float(court[1, 0]),
            border_down=float(court[1, 2]),
        )
    return output


def normalize_court_points(points_xy: np.ndarray, homography: HomographyInfo) -> np.ndarray:
    court = project_points(homography.matrix, points_xy)
    x = (court[0] - homography.border_left) / (homography.border_right - homography.border_left)
    y = (court[1] - homography.border_up) / (homography.border_down - homography.border_up)
    return np.stack([x, y], axis=-1).astype(np.float32)


def normalize_joints(keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    diagonal = max(float(np.linalg.norm(bbox[2:] - bbox[:2])), 1.0)
    center = (bbox[:2] + bbox[2:]) / 2.0
    output = np.zeros_like(keypoints, dtype=np.float32)
    valid = np.any(keypoints > 0.0, axis=1)
    output[valid] = (keypoints[valid] - center) / diagonal
    return output


def person_feet_camera(keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    ankles = keypoints[[15, 16]]
    valid = np.any(ankles > 0.0, axis=1)
    if np.any(valid):
        return ankles[valid].mean(axis=0)
    return np.asarray([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)


def yolo_people(result: Any, keypoint_conf: float) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes_obj = getattr(result, "boxes", None)
    keypoints_obj = getattr(result, "keypoints", None)
    if boxes_obj is None or keypoints_obj is None or keypoints_obj.xy is None:
        return []
    boxes = boxes_obj.xyxy.detach().cpu().numpy()
    points = keypoints_obj.xy.detach().cpu().numpy()
    confidence = (
        keypoints_obj.conf.detach().cpu().numpy()
        if getattr(keypoints_obj, "conf", None) is not None
        else np.ones(points.shape[:2], dtype=np.float32)
    )
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for bbox, keypoints, scores in zip(boxes, points, confidence):
        clean = keypoints.astype(np.float32).copy()
        clean[scores < keypoint_conf] = 0.0
        output.append((clean, bbox.astype(np.float32)))
    return output


def court_candidates(
    people: list[tuple[np.ndarray, np.ndarray]],
    homography: HomographyInfo,
    court_margin: float,
    min_valid_keypoints: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for keypoints, bbox in people:
        foot = person_feet_camera(keypoints, bbox).reshape(2, 1)
        position = normalize_court_points(foot, homography)[0]
        if not (
            -court_margin <= position[0] <= 1.0 + court_margin
            and -court_margin <= position[1] <= 1.0 + court_margin
        ):
            continue
        area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 0.0)
        pose_valid = int(np.count_nonzero(np.any(keypoints != 0.0, axis=1))) >= min_valid_keypoints
        candidates.append(Candidate(keypoints, bbox, position, pose_valid, area))
    return candidates


def court_candidates_with_stats(
    people: list[tuple[np.ndarray, np.ndarray]],
    homography: HomographyInfo,
    court_margin: float,
    min_valid_keypoints: int,
) -> tuple[list[Candidate], int]:
    candidates = court_candidates(people, homography, court_margin, min_valid_keypoints)
    return candidates, len(people) - len(candidates)


def choose_track_candidate(
    candidates: list[Candidate],
    previous_position: np.ndarray | None,
    side: int,
    side_split: float,
    max_track_distance: float,
) -> Candidate | None:
    side_candidates = [
        candidate
        for candidate in candidates
        if (candidate.court_position[1] <= side_split if side == 0 else candidate.court_position[1] > side_split)
    ]
    if not side_candidates:
        return None
    if previous_position is not None:
        distances = [float(np.linalg.norm(candidate.court_position - previous_position)) for candidate in side_candidates]
        best_index = int(np.argmin(distances))
        if distances[best_index] <= max_track_distance:
            return side_candidates[best_index]
        return None
    expected_y = side_split * 0.5 if side == 0 else side_split + (1.0 - side_split) * 0.5
    return max(
        side_candidates,
        key=lambda candidate: candidate.area - abs(float(candidate.court_position[1]) - expected_y) * 10000.0,
    )


def track_candidates(
    candidates_by_frame: list[list[Candidate]],
    side_split: float,
    max_track_distance: float,
    max_track_gap: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(candidates_by_frame)
    joints = np.zeros((frame_count, 2, 17, 2), dtype=np.float32)
    positions = np.zeros((frame_count, 2, 2), dtype=np.float32)
    pose_observed = np.zeros((frame_count, 2), dtype=bool)
    position_observed = np.zeros((frame_count, 2), dtype=bool)
    previous_positions: list[np.ndarray | None] = [None, None]
    missing_ages = [max_track_gap + 1, max_track_gap + 1]

    for frame_index, candidates in enumerate(candidates_by_frame):
        for player_index in range(2):
            previous_position = previous_positions[player_index] if missing_ages[player_index] <= max_track_gap else None
            selected = choose_track_candidate(
                candidates,
                previous_position,
                player_index,
                side_split,
                max_track_distance,
            )
            if selected is None:
                missing_ages[player_index] += 1
                continue
            positions[frame_index, player_index] = selected.court_position
            position_observed[frame_index, player_index] = True
            previous_positions[player_index] = selected.court_position
            missing_ages[player_index] = 0
            if selected.pose_valid:
                joints[frame_index, player_index] = normalize_joints(selected.keypoints, selected.bbox)
                pose_observed[frame_index, player_index] = True
    return joints, positions, pose_observed, position_observed


def false_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, valid in enumerate(mask):
        if not valid and start is None:
            start = index
        elif valid and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def recover_short_gaps(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover each player independently; <=2 forward-fill, <=10 bounded linear interpolation."""
    recovered_values = values.copy()
    valid = observed.copy()
    for player_index in range(2):
        for start, end in list(false_runs(observed[:, player_index])):
            gap = end - start
            previous = start - 1 if start > 0 and observed[start - 1, player_index] else None
            following = end if end < len(observed) and observed[end, player_index] else None
            if gap <= 2 and previous is not None:
                recovered_values[start:end, player_index] = recovered_values[previous, player_index]
                valid[start:end, player_index] = True
            elif gap <= 10 and previous is not None and following is not None:
                for offset, frame_index in enumerate(range(start, end), start=1):
                    alpha = offset / (gap + 1)
                    recovered_values[frame_index, player_index] = (
                        (1.0 - alpha) * recovered_values[previous, player_index]
                        + alpha * recovered_values[following, player_index]
                    )
                valid[start:end, player_index] = True
    return recovered_values, valid


def iter_video_frames(video_path: Path, frame_numbers: Iterable[int]) -> Iterable[tuple[int, np.ndarray | None]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    try:
        for frame_number in frame_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            yield frame_number, frame if ok else None
    finally:
        capture.release()


def load_shuttle(path: Path, width: int, height: int) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for row in read_csv(path):
        frame = to_int(row.get("OriginalFrame", row.get("Frame")))
        if to_int(row.get("Visibility")) == 1 and width > 0 and height > 0:
            output[frame] = np.asarray([to_float(row["X"]) / width, to_float(row["Y"]) / height], dtype=np.float32)
        else:
            output[frame] = np.zeros((2,), dtype=np.float32)
    return output


def build_yolo_model(weights: Path) -> Any:
    if not weights.exists():
        raise FileNotFoundError(f"Missing YOLO pose weights: {weights}")
    matplotlib_root = ROOT / "project" / "outputs" / ".matplotlib"
    ultralytics_root = ROOT / "project" / "outputs" / ".ultralytics"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    ultralytics_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_root))
    from ultralytics import YOLO

    return YOLO(str(weights))


def infer_clip(
    window: StrokeWindow,
    task: Phase06Task,
    homography: HomographyInfo,
    model: Any,
    args: argparse.Namespace,
) -> tuple[FeatureBundle, list[dict[str, str]]]:
    frame_numbers = list(range(window.window_start_original, window.window_end_original + 1))
    decoded = list(iter_video_frames(task.source_video, frame_numbers))
    valid_indices = [index for index, (_, frame) in enumerate(decoded) if frame is not None]
    valid_frames = [decoded[index][1] for index in valid_indices]
    results_by_index: dict[int, Any] = {}
    if valid_frames:
        results = model.predict(
            valid_frames,
            verbose=False,
            conf=args.det_conf,
            imgsz=args.imgsz,
            device=args.worker_device,
            half=args.half and args.worker_device != "cpu",
            batch=args.batch_size,
        )
        results_by_index = dict(zip(valid_indices, results))

    candidates_by_frame: list[list[Candidate]] = []
    rejected_out_of_range_candidates = 0
    problems: list[dict[str, str]] = []
    for index, (frame_number, frame) in enumerate(decoded):
        if frame is None:
            candidates_by_frame.append([])
            problems.append(
                {
                    "problem": f"Could not decode frame {frame_number} from {repo_relative(task.source_video)}.",
                    "fix": "Verify the raw MP4 or regenerate the source rally.",
                }
            )
            continue
        people = yolo_people(results_by_index[index], args.keypoint_conf)
        candidates, rejected = court_candidates_with_stats(
            people,
            homography,
            args.court_margin,
            args.min_valid_keypoints,
        )
        candidates_by_frame.append(candidates)
        rejected_out_of_range_candidates += rejected

    joints, positions, pose_observed, position_observed = track_candidates(
        candidates_by_frame,
        args.side_split,
        args.max_track_distance,
        args.max_track_gap,
    )
    joints, pose_valid = recover_short_gaps(joints, pose_observed)
    positions, position_valid = recover_short_gaps(positions, position_observed)
    shuttle_by_frame = load_shuttle(task.denoised_csv, task.width, task.height)
    shuttle = np.stack([shuttle_by_frame.get(frame, np.zeros((2,), dtype=np.float32)) for frame in frame_numbers])
    return (
        FeatureBundle(
            joints,
            positions,
            shuttle,
            pose_observed,
            position_observed,
            pose_valid,
            position_valid,
            rejected_out_of_range_candidates,
        ),
        problems[:3],
    )


def feature_paths(output_root: Path, video_id: str, clip_id: str) -> tuple[Path, Path, Path, Path]:
    base = output_root / video_id / clip_id
    return (
        Path(f"{base}_joints.npy"),
        Path(f"{base}_pos.npy"),
        Path(f"{base}_shuttle.npy"),
        Path(f"{base}_validity.npz"),
    )


def save_bundle(output_root: Path, video_id: str, clip_id: str, bundle: FeatureBundle) -> tuple[Path, Path, Path, Path]:
    joints_path, pos_path, shuttle_path, validity_path = feature_paths(output_root, video_id, clip_id)
    joints_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(joints_path, bundle.joints.astype(np.float32))
    np.save(pos_path, bundle.positions.astype(np.float32))
    np.save(shuttle_path, bundle.shuttle.astype(np.float32))
    np.savez_compressed(
        validity_path,
        p1_pose_valid=bundle.pose_valid[:, 0],
        p2_pose_valid=bundle.pose_valid[:, 1],
        p1_position_valid=bundle.position_valid[:, 0],
        p2_position_valid=bundle.position_valid[:, 1],
        p1_pose_observed=bundle.pose_observed[:, 0],
        p2_pose_observed=bundle.pose_observed[:, 1],
        p1_position_observed=bundle.position_observed[:, 0],
        p2_position_observed=bundle.position_observed[:, 1],
    )
    return joints_path, pos_path, shuttle_path, validity_path


def rate(mask: np.ndarray) -> str:
    return f"{float(mask.mean()):.6f}" if mask.size else ""


def validate_bundle(
    window: StrokeWindow,
    clip_id: str,
    bundle: FeatureBundle,
    problems: list[dict[str, str]],
    court_margin: float,
    side_split: float,
) -> dict[str, Any]:
    frame_count = bundle.joints.shape[0]
    valid_shape = (
        bundle.joints.shape == (frame_count, 2, 17, 2)
        and bundle.positions.shape == (frame_count, 2, 2)
        and bundle.shuttle.shape == (frame_count, 2)
        and bundle.pose_valid.shape == bundle.position_valid.shape == (frame_count, 2)
    )
    both_pose_missing = np.all(~bundle.pose_valid, axis=1)
    both_position_missing = np.all(~bundle.position_valid, axis=1)
    recovered_pose = bundle.pose_valid & ~bundle.pose_observed
    recovered_position = bundle.position_valid & ~bundle.position_observed
    valid_positions = bundle.positions[bundle.position_valid]
    position_out = int(
        np.sum((valid_positions < -court_margin) | (valid_positions > 1.0 + court_margin))
    )
    side_inconsistent = (
        (bundle.position_valid[:, 0] & (bundle.positions[:, 0, 1] > side_split))
        | (bundle.position_valid[:, 1] & (bundle.positions[:, 1, 1] <= side_split))
    )
    problem = ""
    fix = ""
    if not valid_shape:
        problem = "Feature bundle does not match the Phase 09 tracked-output contract."
        fix = "Regenerate the clip and verify joints, position, shuttle, and validity shapes."
    elif problems:
        problem = problems[0]["problem"]
        fix = problems[0]["fix"]
    return {
        "video_id": window.video_id,
        "rally_id": window.rally_id,
        "event_rank": window.event_rank,
        "clip_id": clip_id,
        "frame_count": frame_count,
        "valid_shape": valid_shape,
        "p1_pose_observed_rate": rate(bundle.pose_observed[:, 0]),
        "p2_pose_observed_rate": rate(bundle.pose_observed[:, 1]),
        "p1_pose_valid_rate": rate(bundle.pose_valid[:, 0]),
        "p2_pose_valid_rate": rate(bundle.pose_valid[:, 1]),
        "p1_position_observed_rate": rate(bundle.position_observed[:, 0]),
        "p2_position_observed_rate": rate(bundle.position_observed[:, 1]),
        "p1_position_valid_rate": rate(bundle.position_valid[:, 0]),
        "p2_position_valid_rate": rate(bundle.position_valid[:, 1]),
        "both_pose_missing_frames": int(both_pose_missing.sum()),
        "both_pose_missing_rate": rate(both_pose_missing),
        "both_position_missing_frames": int(both_position_missing.sum()),
        "both_position_missing_rate": rate(both_position_missing),
        "pose_recovered_values": int(recovered_pose.sum()),
        "position_recovered_values": int(recovered_position.sum()),
        "shuttle_visible_rate": rate(np.any(bundle.shuttle != 0.0, axis=1)),
        "pose_tensor_zero_rate": rate(bundle.joints == 0.0),
        "position_tensor_zero_rate": rate(bundle.positions == 0.0),
        "player_pos_out_of_filter_range": position_out,
        "rejected_out_of_range_candidates": bundle.rejected_out_of_range_candidates,
        "player_side_inconsistent_frames": int(side_inconsistent.sum()),
        "player_side_inconsistent_rate": rate(side_inconsistent),
        "problem": problem,
        "fix": fix,
    }


def worker_metadata_root(args: argparse.Namespace) -> Path:
    if args.worker_count == 1:
        return args.output_root
    return args.output_root / ".phase09_workers" / f"worker_{args.worker_index}"


def write_window_analysis(args: argparse.Namespace, windows: list[StrokeWindow]) -> None:
    matched_ids = load_match_clip_ids(args.phase07_matches)
    reference_lengths: dict[str, int] = {}
    for split in ("train", "val", "test"):
        metadata_path = args.reference_collated_root / f"{split}_metadata.csv"
        if metadata_path.exists():
            reference_lengths.update({row["clip_id"]: to_int(row["original_len"]) for row in read_csv(metadata_path)})
    rows = [
        {
            "video_id": window.video_id,
            "rally_id": window.rally_id,
            "event_rank": window.event_rank,
            "event_frame_original": window.event_frame_original,
            "prev_event_frame_original": window.prev_event_frame_original if window.prev_event_frame_original is not None else "",
            "next_event_frame_original": window.next_event_frame_original if window.next_event_frame_original is not None else "",
            "window_start_original": window.window_start_original,
            "window_end_original": window.window_end_original,
            "event_offset_frames": window.event_frame_original - window.window_start_original,
            "frame_count": window.window_end_original - window.window_start_original + 1,
            "window_strategy": args.window_strategy,
            "clip_id": clip_id_for_window(window, matched_ids),
            "reference_frame_count": reference_lengths.get(clip_id_for_window(window, matched_ids), ""),
            "frame_count_abs_error": (
                abs(
                    window.window_end_original
                    - window.window_start_original
                    + 1
                    - reference_lengths[clip_id_for_window(window, matched_ids)]
                )
                if clip_id_for_window(window, matched_ids) in reference_lengths
                else ""
            ),
        }
        for window in windows
    ]
    columns = list(rows[0]) if rows else []
    output_csv = args.output_root / f"phase09_{args.window_strategy}_window_analysis.csv"
    write_csv(output_csv, rows, columns)
    lengths = np.asarray([to_int(row["frame_count"]) for row in rows], dtype=np.int64)
    matched_rows = [row for row in rows if row["reference_frame_count"] != ""]
    reference = np.asarray([to_int(row["reference_frame_count"]) for row in matched_rows], dtype=np.int64)
    errors = np.asarray([to_int(row["frame_count_abs_error"]) for row in matched_rows], dtype=np.int64)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only",
        "window_strategy": args.window_strategy,
        "window_count": len(rows),
        "frame_count": {
            "mean": float(lengths.mean()) if lengths.size else None,
            "median": float(np.median(lengths)) if lengths.size else None,
            "min": int(lengths.min()) if lengths.size else None,
            "max": int(lengths.max()) if lengths.size else None,
        },
        "matched_reference": {
            "clip_count": len(matched_rows),
            "reference_mean": float(reference.mean()) if reference.size else None,
            "length_mae": float(errors.mean()) if errors.size else None,
            "exact_length_rate": float((errors == 0).mean()) if errors.size else None,
        },
        "output_csv": repo_relative(output_csv),
    }
    summary_path = args.output_root / f"phase09_{args.window_strategy}_window_analysis.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_worker(args: argparse.Namespace) -> None:
    windows = load_windows(args)
    metadata_root = worker_metadata_root(args)
    if not windows:
        write_csv(metadata_root / "phase09_feature_manifest.csv", [], MANIFEST_COLUMNS)
        write_csv(metadata_root / "phase09_feature_validation.csv", [], VALIDATION_COLUMNS)
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "worker_completed_empty_shard",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "worker_device": args.worker_device,
            "selected_windows": 0,
            "exported_feature_bundles": 0,
            "problems": [],
        }
        (metadata_root / "phase09_worker_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return
    tasks = load_phase06_tasks(args.phase06_manifest)
    windows = [
        rebuild_temporal_window(window, tasks[(window.video_id, window.rally_id)], args.window_strategy)
        if (window.video_id, window.rally_id) in tasks
        else window
        for window in windows
    ]
    if args.analyze_windows_only:
        write_window_analysis(args, windows)
        return
    homographies = load_homographies(args.homography_csv)
    matched_ids = load_match_clip_ids(args.phase07_matches)
    model = build_yolo_model(args.pose_weights)
    manifest_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []

    for window in windows:
        task = tasks.get((window.video_id, window.rally_id))
        homography = homographies.get(window.video_id)
        if task is None or homography is None:
            problems.append(
                {
                    "problem": f"Missing Phase 06 task or homography for video={window.video_id}, rally={window.rally_id}.",
                    "fix": "Restore the Phase 06 manifest and Phase 08/known ShuttleSet homography before Phase 09.",
                }
            )
            continue
        clip_id = clip_id_for_window(window, matched_ids)
        bundle, clip_problems = infer_clip(window, task, homography, model, args)
        validation = validate_bundle(window, clip_id, bundle, clip_problems, args.court_margin, args.side_split)
        validation_rows.append(validation)
        if not validation["valid_shape"]:
            problems.append({"problem": validation["problem"], "fix": validation["fix"]})
            continue
        paths = save_bundle(args.output_root, window.video_id, clip_id, bundle)
        manifest_rows.append(
            {
                "video_id": window.video_id,
                "rally_id": window.rally_id,
                "event_rank": window.event_rank,
                "clip_id": clip_id,
                "feature_source": "yolo26x_tracked",
                "window_strategy": args.window_strategy,
                "player_order": "Top,Bottom",
                "event_frame_original": window.event_frame_original,
                "prev_event_frame_original": window.prev_event_frame_original if window.prev_event_frame_original is not None else "",
                "next_event_frame_original": window.next_event_frame_original if window.next_event_frame_original is not None else "",
                "event_offset_frames": window.event_frame_original - window.window_start_original,
                "window_start_original": window.window_start_original,
                "window_end_original": window.window_end_original,
                "frame_count": bundle.joints.shape[0],
                "joints_shape": "x".join(map(str, bundle.joints.shape)),
                "pos_shape": "x".join(map(str, bundle.positions.shape)),
                "shuttle_shape": "x".join(map(str, bundle.shuttle.shape)),
                "joints_npy": repo_relative(paths[0]),
                "pos_npy": repo_relative(paths[1]),
                "shuttle_npy": repo_relative(paths[2]),
                "validity_npz": repo_relative(paths[3]),
                "source_video": repo_relative(task.source_video),
                "worker_device": args.worker_device,
            }
        )

    write_csv(metadata_root / "phase09_feature_manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    write_csv(metadata_root / "phase09_feature_validation.csv", validation_rows, VALIDATION_COLUMNS)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "worker_completed" if manifest_rows else "worker_blocked",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "worker_device": args.worker_device,
        "selected_windows": len(windows),
        "exported_feature_bundles": len(manifest_rows),
        "problems": problems,
    }
    (metadata_root / "phase09_worker_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def aggregate_quality(validation_rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    total_frames = sum(to_int(row["frame_count"]) for row in validation_rows)
    both_pose_missing = sum(to_int(row["both_pose_missing_frames"]) for row in validation_rows)
    both_position_missing = sum(to_int(row["both_position_missing_frames"]) for row in validation_rows)
    pose_rate = both_pose_missing / total_frames if total_frames else 1.0
    position_rate = both_position_missing / total_frames if total_frames else 1.0
    player_pose_missing_rates = [
        (
            sum((1.0 - float(row[f"p{player}_pose_valid_rate"])) * to_int(row["frame_count"]) for row in validation_rows)
            / total_frames
            if total_frames
            else 1.0
        )
        for player in (1, 2)
    ]
    player_position_missing_rates = [
        (
            sum(
                (1.0 - float(row[f"p{player}_position_valid_rate"])) * to_int(row["frame_count"])
                for row in validation_rows
            )
            / total_frames
            if total_frames
            else 1.0
        )
        for player in (1, 2)
    ]
    weighted_rate = lambda key: (
        sum(to_float(row[key]) * to_int(row["frame_count"]) for row in validation_rows) / total_frames
        if total_frames
        else 1.0
    )
    side_inconsistent_frames = sum(to_int(row["player_side_inconsistent_frames"]) for row in validation_rows)
    player_pos_out_of_filter_range = sum(
        to_int(row["player_pos_out_of_filter_range"]) for row in validation_rows
    )
    side_inconsistent_rate = side_inconsistent_frames / total_frames if total_frames else 1.0
    status = (
        "quality_ready"
        if (
            max(player_pose_missing_rates) <= args.max_player_pose_missing_rate
            and max(player_position_missing_rates) <= args.max_player_position_missing_rate
            and pose_rate <= args.max_both_pose_missing_rate
            and position_rate <= args.max_both_position_missing_rate
            and player_pos_out_of_filter_range == 0
            and side_inconsistent_rate <= args.max_player_side_inconsistent_rate
        )
        else "completed_needs_quality_review"
    )
    return {
        "status": status,
        "total_frames": total_frames,
        "both_pose_missing_frames": both_pose_missing,
        "both_pose_missing_rate": pose_rate,
        "both_position_missing_frames": both_position_missing,
        "both_position_missing_rate": position_rate,
        "p1_pose_missing_rate": player_pose_missing_rates[0],
        "p2_pose_missing_rate": player_pose_missing_rates[1],
        "p1_position_missing_rate": player_position_missing_rates[0],
        "p2_position_missing_rate": player_position_missing_rates[1],
        "pose_recovered_values": sum(to_int(row["pose_recovered_values"]) for row in validation_rows),
        "position_recovered_values": sum(to_int(row["position_recovered_values"]) for row in validation_rows),
        "pose_tensor_zero_rate": weighted_rate("pose_tensor_zero_rate"),
        "position_tensor_zero_rate": weighted_rate("position_tensor_zero_rate"),
        "player_pos_out_of_filter_range": player_pos_out_of_filter_range,
        "rejected_out_of_range_candidates": sum(
            to_int(row["rejected_out_of_range_candidates"]) for row in validation_rows
        ),
        "player_side_inconsistent_frames": side_inconsistent_frames,
        "player_side_inconsistent_rate": side_inconsistent_rate,
        "quality_thresholds": {
            "max_player_pose_missing_rate": args.max_player_pose_missing_rate,
            "max_player_position_missing_rate": args.max_player_position_missing_rate,
            "max_both_pose_missing_rate": args.max_both_pose_missing_rate,
            "max_both_position_missing_rate": args.max_both_position_missing_rate,
            "max_player_side_inconsistent_rate": args.max_player_side_inconsistent_rate,
        },
    }


def merge_worker_outputs(args: argparse.Namespace, devices: list[str]) -> None:
    manifests: list[dict[str, str]] = []
    validations: list[dict[str, str]] = []
    worker_summaries: list[dict[str, Any]] = []
    for index in range(len(devices)):
        metadata_root = args.output_root / ".phase09_workers" / f"worker_{index}"
        manifests.extend(read_csv(metadata_root / "phase09_feature_manifest.csv"))
        validations.extend(read_csv(metadata_root / "phase09_feature_validation.csv"))
        worker_summaries.append(json.loads((metadata_root / "phase09_worker_summary.json").read_text(encoding="utf-8")))
    manifests.sort(key=lambda row: (to_int(row["video_id"]), to_int(row["rally_id"]), to_int(row["event_rank"])))
    validations.sort(key=lambda row: (to_int(row["video_id"]), to_int(row["rally_id"]), to_int(row["event_rank"])))
    write_csv(args.output_root / "phase09_feature_manifest.csv", manifests, MANIFEST_COLUMNS)
    write_csv(args.output_root / "phase09_feature_validation.csv", validations, VALIDATION_COLUMNS)
    quality = aggregate_quality(validations, args)
    selected_windows = sum(to_int(summary["selected_windows"]) for summary in worker_summaries)
    worker_problems = [problem for summary in worker_summaries for problem in summary["problems"]]
    completion_status = (
        "blocked"
        if not manifests
        else ("partial_with_problems" if len(manifests) != selected_windows or worker_problems else quality["status"])
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "pose_features/extract.py",
        "status": completion_status,
        "feature_contract": {
            "legacy_bst_triple": {
                "joints": "(T,2,17,2) float32",
                "positions": "(T,2,2) float32",
                "shuttle": "(T,2) float32",
            },
            "validity_sidecar": {
                "p1_pose_valid": "(T,) bool",
                "p2_pose_valid": "(T,) bool",
                "p1_position_valid": "(T,) bool",
                "p2_position_valid": "(T,) bool",
                "observed_masks": "same four masks with _observed suffix",
            },
        },
        "runtime_config": summary_config(args, devices),
        "selected_windows": selected_windows,
        "exported_feature_bundles": len(manifests),
        "quality": quality,
        "problems": worker_problems,
        "workers": worker_summaries,
        "outputs": {
            "manifest": repo_relative(args.output_root / "phase09_feature_manifest.csv"),
            "validation": repo_relative(args.output_root / "phase09_feature_validation.csv"),
            "summary": repo_relative(args.output_root / "phase09_feature_summary.json"),
        },
    }
    (args.output_root / "phase09_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summary_config(args: argparse.Namespace, devices: list[str]) -> dict[str, Any]:
    ignored = {"worker_device", "worker_index", "worker_count"}
    output: dict[str, Any] = {"devices": devices}
    for key, value in vars(args).items():
        if key in ignored:
            continue
        output[key] = repo_relative(value) if isinstance(value, Path) else value
    return output


def launch_workers(args: argparse.Namespace, devices: list[str]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    for index, device in enumerate(devices):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-device",
            device,
            "--worker-index",
            str(index),
            "--worker-count",
            str(len(devices)),
        ]
        processes.append(subprocess.Popen(command))
    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"One or more Phase 09 workers failed: return_codes={return_codes}")
    merge_worker_outputs(args, devices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 09: tracked YOLO26x pose, player position, shuttle, and validity export.")
    parser.add_argument(
        "--stroke-windows",
        type=Path,
        default=DEFAULT_WINDOWS,
        help="Phase 07-compatible windows; defaults to the exact Phase 02-label-derived bundle.",
    )
    parser.add_argument(
        "--phase07-matches",
        type=Path,
        default=DEFAULT_MATCHES,
        help="Exact Phase 02 label-to-window mapping produced by Phase 07.",
    )
    parser.add_argument("--phase06-manifest", type=Path, default=DEFAULT_PHASE06_MANIFEST)
    parser.add_argument("--homography-csv", type=Path, default=DEFAULT_HOMOGRAPHY_CSV)
    parser.add_argument("--pose-weights", type=Path, default=DEFAULT_POSE_WEIGHTS)
    parser.add_argument("--reference-collated-root", type=Path, default=DEFAULT_REFERENCE_COLLATED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", default=[], help="Video id, comma list, or range, for example 41-44.")
    parser.add_argument("--rally-id", action="append", default=[])
    parser.add_argument("--max-clips", type=int, default=0, help="Limit clips before worker sharding; 0 means all.")
    parser.add_argument(
        "--window-strategy",
        choices=("bst-compatible", "midpoint"),
        default="bst-compatible",
        help="BST-compatible reproduces previous/current/next hit windows; midpoint preserves the old ablation.",
    )
    parser.add_argument(
        "--analyze-windows-only",
        action="store_true",
        help="Write temporal-window lengths and exit without loading YOLO or running inference.",
    )
    parser.add_argument("--devices", default="0,1", help="Comma-separated inference devices. Default uses two GPUs: 0,1.")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU YOLO inference batch size.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--keypoint-conf", type=float, default=0.20)
    parser.add_argument("--court-margin", type=float, default=0.20)
    parser.add_argument("--side-split", type=float, default=0.50)
    parser.add_argument("--min-valid-keypoints", type=int, default=5)
    parser.add_argument("--max-track-distance", type=float, default=0.45)
    parser.add_argument("--max-track-gap", type=int, default=10, help="Forget stale track memory after this many missing frames.")
    parser.add_argument("--max-player-pose-missing-rate", type=float, default=0.30)
    parser.add_argument("--max-player-position-missing-rate", type=float, default=0.20)
    parser.add_argument("--max-both-pose-missing-rate", type=float, default=0.20)
    parser.add_argument("--max-both-position-missing-rate", type=float, default=0.10)
    parser.add_argument("--max-player-side-inconsistent-rate", type=float, default=0.01)
    parser.add_argument("--worker-device", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0.0 <= args.side_split <= 1.0:
        parser.error("--side-split must be in [0,1]")
    return args


def main() -> None:
    args = parse_args()
    if args.analyze_windows_only:
        windows = load_windows(args)
        tasks = load_phase06_tasks(args.phase06_manifest)
        windows = [
            rebuild_temporal_window(window, tasks[(window.video_id, window.rally_id)], args.window_strategy)
            if (window.video_id, window.rally_id) in tasks
            else window
            for window in windows
        ]
        write_window_analysis(args, windows)
        return
    if args.worker_device:
        run_worker(args)
        return
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device.")
    if len(devices) == 1:
        args.worker_device = devices[0]
        run_worker(args)
        merge_worker_outputs_single(args, devices)
    else:
        launch_workers(args, devices)


def merge_worker_outputs_single(args: argparse.Namespace, devices: list[str]) -> None:
    manifest = read_csv(args.output_root / "phase09_feature_manifest.csv")
    validation = read_csv(args.output_root / "phase09_feature_validation.csv")
    worker_summary = json.loads((args.output_root / "phase09_worker_summary.json").read_text(encoding="utf-8"))
    quality = aggregate_quality(validation, args)
    selected_windows = to_int(worker_summary["selected_windows"])
    completion_status = (
        "blocked"
        if not manifest
        else (
            "partial_with_problems"
            if len(manifest) != selected_windows or worker_summary["problems"]
            else quality["status"]
        )
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "pose_features/extract.py",
        "status": completion_status,
        "feature_contract": {
            "legacy_bst_triple": {
                "joints": "(T,2,17,2) float32",
                "positions": "(T,2,2) float32",
                "shuttle": "(T,2) float32",
            },
            "validity_sidecar": "<clip_id>_validity.npz with separate pose/position observed and valid masks",
        },
        "runtime_config": summary_config(args, devices),
        "selected_windows": selected_windows,
        "exported_feature_bundles": len(manifest),
        "quality": quality,
        "problems": worker_summary["problems"],
        "workers": [worker_summary],
        "outputs": {
            "manifest": repo_relative(args.output_root / "phase09_feature_manifest.csv"),
            "validation": repo_relative(args.output_root / "phase09_feature_validation.csv"),
            "summary": repo_relative(args.output_root / "phase09_feature_summary.json"),
        },
    }
    (args.output_root / "phase09_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
