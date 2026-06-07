from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = ROOT / "project" / "outputs" / "features_yolo26x_tracked_phase02_40_44"
DEFAULT_HOMOGRAPHY_CSV = ROOT / "project" / "dataset" / "ShuttleSet" / "set" / "homography.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "visualizations" / "phase09_yolo26x_tracked_phase02_40_44"

PLAYER_COLORS = [(255, 140, 40), (40, 210, 255)]
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray
    corners: np.ndarray
    border_left: float
    border_right: float
    border_up: float
    border_down: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def project_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_xy, np.ones((1, points_xy.shape[1]))], axis=0)
    projected = matrix @ homogeneous
    return projected[:2] / projected[2:3]


def load_homographies(path: Path) -> dict[str, Homography]:
    output: dict[str, Homography] = {}
    for row in read_csv(path):
        matrix = np.asarray(ast.literal_eval(row["homography_matrix"]), dtype=np.float64)
        corners = np.asarray(
            [
                [float(row["upleft_x"]), float(row["upright_x"]), float(row["downright_x"]), float(row["downleft_x"])],
                [float(row["upleft_y"]), float(row["upright_y"]), float(row["downright_y"]), float(row["downleft_y"])],
            ],
            dtype=np.float64,
        )
        projected = project_points(matrix, corners)
        output[row["id"]] = Homography(
            matrix=matrix,
            corners=corners,
            border_left=float(projected[0, 0]),
            border_right=float(projected[0, 1]),
            border_up=float(projected[1, 0]),
            border_down=float(projected[1, 3]),
        )
    return output


def position_to_camera(position: np.ndarray, homography: Homography) -> np.ndarray:
    court = np.asarray(
        [
            homography.border_left + position[0] * (homography.border_right - homography.border_left),
            homography.border_up + position[1] * (homography.border_down - homography.border_up),
        ],
        dtype=np.float64,
    ).reshape(2, 1)
    return project_points(np.linalg.inv(homography.matrix), court)[:, 0]


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
        return frame
    finally:
        capture.release()


def put_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int] = (18, 28)) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(image, line, (x, y + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)


def draw_source_panel(
    frame: np.ndarray,
    homography: Homography,
    position: np.ndarray,
    position_valid: np.ndarray,
    shuttle: np.ndarray,
    title: list[str],
) -> np.ndarray:
    output = frame.copy()
    court = np.round(homography.corners.T).astype(np.int32)
    cv2.polylines(output, [court], True, (80, 255, 80), 3, cv2.LINE_AA)
    for player_index in range(2):
        if position_valid[player_index]:
            point = np.round(position_to_camera(position[player_index], homography)).astype(int)
            cv2.circle(output, tuple(point), 13, PLAYER_COLORS[player_index], -1, cv2.LINE_AA)
            cv2.putText(output, f"P{player_index + 1}", tuple(point + [14, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLAYER_COLORS[player_index], 2)
    if np.any(shuttle != 0):
        point = np.round(shuttle * np.asarray([output.shape[1], output.shape[0]])).astype(int)
        cv2.drawMarker(output, tuple(point), (0, 0, 255), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
        cv2.putText(output, "shuttle", tuple(point + [14, -12]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    shade = output.copy()
    cv2.rectangle(shade, (0, 0), (output.shape[1], 105), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.6, output, 0.4, 0, output)
    put_lines(output, title)
    return output


def draw_pose_panel(joints: np.ndarray, pose_valid: np.ndarray, pose_observed: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "Stored BST-normalized player poses", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    for player_index in range(2):
        pose = joints[player_index]
        center = np.asarray([width * (0.27 + player_index * 0.48), height * 0.55])
        points = pose * min(width, height) * 0.72 + center
        valid = np.any(pose != 0, axis=1)
        for start, end in COCO_EDGES:
            if valid[start] and valid[end]:
                cv2.line(panel, tuple(points[start].astype(int)), tuple(points[end].astype(int)), PLAYER_COLORS[player_index], 3, cv2.LINE_AA)
        for point in points[valid]:
            cv2.circle(panel, tuple(point.astype(int)), 4, PLAYER_COLORS[player_index], -1, cv2.LINE_AA)
        state = "observed" if pose_observed[player_index] else ("recovered" if pose_valid[player_index] else "missing")
        label = f"P{player_index + 1}: {state}"
        cv2.putText(panel, label, (int(center[0] - 65), height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_COLORS[player_index], 2)
    return panel


def draw_court_panel(
    position: np.ndarray,
    position_valid: np.ndarray,
    position_observed: np.ndarray,
    shuttle: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "Normalized court positions", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    left, top, right, bottom = 90, 65, width - 90, height - 55
    cv2.rectangle(panel, (left, top), (right, bottom), (80, 255, 80), 3)
    cv2.line(panel, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (200, 200, 200), 2)
    for player_index in range(2):
        if position_valid[player_index]:
            point = np.asarray([left + position[player_index, 0] * (right - left), top + position[player_index, 1] * (bottom - top)])
            point = np.round(point).astype(int)
            cv2.circle(panel, tuple(point), 12, PLAYER_COLORS[player_index], -1, cv2.LINE_AA)
            state = "obs" if position_observed[player_index] else "rec"
            cv2.putText(panel, f"P{player_index + 1} {state}", tuple(point + [14, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_COLORS[player_index], 2)
    cv2.putText(panel, f"shuttle image xy: ({shuttle[0]:.3f}, {shuttle[1]:.3f})", (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return panel


def select_samples(validation: list[dict[str, str]], count_per_category: int) -> list[tuple[str, dict[str, str]]]:
    categories = {
        "good": lambda row: float(row["both_pose_missing_rate"]) == 0 and float(row["both_position_missing_rate"]) == 0,
        "partial_dropout": lambda row: 0.4 <= float(row["both_pose_missing_rate"]) <= 0.6,
        "all_pose_missing": lambda row: float(row["both_pose_missing_rate"]) == 1,
    }
    selected: list[tuple[str, dict[str, str]]] = []
    for category, predicate in categories.items():
        candidates = [row for row in validation if predicate(row)]
        candidates.sort(key=lambda row: (row["video_id"], int(row["rally_id"]), int(row["event_rank"])))
        if candidates:
            indices = np.linspace(0, len(candidates) - 1, min(count_per_category, len(candidates)), dtype=int)
            selected.extend((category, candidates[index]) for index in indices)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Phase 09 player pose, player position, and shuttle visual QA.")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--homography-csv", type=Path, default=DEFAULT_HOMOGRAPHY_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--count-per-category", type=int, default=2)
    args = parser.parse_args()

    manifest = {row["clip_id"]: row for row in read_csv(args.feature_root / "phase09_feature_manifest.csv")}
    validation = read_csv(args.feature_root / "phase09_feature_validation.csv")
    homographies = load_homographies(args.homography_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contact_rows: list[np.ndarray] = []

    for category, quality in select_samples(validation, args.count_per_category):
        row = manifest[quality["clip_id"]]
        joints = np.load(resolve_path(row["joints_npy"]))
        position = np.load(resolve_path(row["pos_npy"]))
        shuttle = np.load(resolve_path(row["shuttle_npy"]))
        validity = np.load(resolve_path(row["validity_npz"]))
        pose_valid = np.stack([validity["p1_pose_valid"], validity["p2_pose_valid"]], axis=1)
        pose_observed = np.stack([validity["p1_pose_observed"], validity["p2_pose_observed"]], axis=1)
        position_valid = np.stack([validity["p1_position_valid"], validity["p2_position_valid"]], axis=1)
        position_observed = np.stack([validity["p1_position_observed"], validity["p2_position_observed"]], axis=1)
        valid = np.any(pose_valid, axis=1)
        center = len(joints) // 2
        frame_offset = min(np.flatnonzero(valid), key=lambda index: abs(int(index) - center)) if np.any(valid) else center
        original_frame = int(row["window_start_original"]) + int(frame_offset)
        frame = read_frame(resolve_path(row["source_video"]), original_frame)
        source = draw_source_panel(
            frame,
            homographies[row["video_id"]],
            position[frame_offset],
            position_valid[frame_offset],
            shuttle[frame_offset],
            [
                f"{category} | clip={row['clip_id']} | source frame={original_frame}",
                f"both-pose missing={float(quality['both_pose_missing_rate']):.1%} | shuttle visibility={float(quality['shuttle_visible_rate']):.1%}",
                "green=court | blue/orange=projected player feet | red=shuttle",
            ],
        )
        panel_width = max(520, source.shape[1] // 2)
        pose_panel = draw_pose_panel(
            joints[frame_offset],
            pose_valid[frame_offset],
            pose_observed[frame_offset],
            (panel_width, source.shape[0] // 2),
        )
        court_panel = draw_court_panel(
            position[frame_offset],
            position_valid[frame_offset],
            position_observed[frame_offset],
            shuttle[frame_offset],
            (panel_width, source.shape[0] - pose_panel.shape[0]),
        )
        combined = np.hstack([source, np.vstack([pose_panel, court_panel])])
        path = args.output_root / f"{category}_{row['clip_id']}.jpg"
        cv2.imwrite(str(path), combined)
        contact_rows.append(cv2.resize(combined, (1200, int(combined.shape[0] * 1200 / combined.shape[1]))))
        print(path.relative_to(ROOT))

    if contact_rows:
        width = max(image.shape[1] for image in contact_rows)
        contact_rows = [cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for image in contact_rows]
        contact = np.vstack(contact_rows)
        path = args.output_root / "phase09_visual_qa_contact_sheet.jpg"
        cv2.imwrite(str(path), contact)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
