#!/usr/bin/env python3
"""Generate selectable visual examples from cached video-44 inference outputs."""
from __future__ import annotations

import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/video44_showcase_matplotlib")
import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "project" / "video_44"
VIDEO = next((ROOT / "project" / "dataset" / "ShuttleSet_raw_videos").glob("44 - *.mp4"))
RALLIES = ROOT / "project" / "outputs" / "rallies" / "44" / "rally_intervals.csv"
SHUTTLE_MANIFEST = ROOT / "project" / "outputs" / "shuttle" / "phase06_shuttle_tracking_manifest.csv"
FEATURE_ROOT = ROOT / "project" / "outputs" / "features_yolo26x_bst_tracked_phase02_41_44"
FEATURE_MANIFEST = FEATURE_ROOT / "phase09_feature_manifest.csv"
FEATURE_VALIDATION = FEATURE_ROOT / "phase09_feature_validation.csv"
STRUCTURED = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44" / "phase10_structured_strokes.csv"
HOMOGRAPHY = ROOT / "project" / "dataset" / "ShuttleSet" / "set" / "homography.csv"

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
    (14, 16),
]
PLAYER_COLORS = [(255, 145, 35), (35, 210, 255)]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index}")
    return frame


def add_title(image: np.ndarray, lines: list[str], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    overlay = image.copy()
    height = 18 + 27 * len(lines)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (15, 26 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def sheet(images: list[np.ndarray], columns: int = 3, tile: tuple[int, int] = (480, 270)) -> np.ndarray:
    prepared = [cv2.resize(image, tile) for image in images]
    blank = np.full((tile[1], tile[0], 3), 245, dtype=np.uint8)
    while len(prepared) % columns:
        prepared.append(blank.copy())
    rows = [np.hstack(prepared[index:index + columns]) for index in range(0, len(prepared), columns)]
    return np.vstack(rows)


def sampled_indices(start: int, end: int, count: int = 6) -> list[int]:
    return np.linspace(start, end, count, dtype=int).tolist()


def ensure_dirs() -> None:
    for name in [
        "rally_filtering", "court_detection", "shuttle_tracking", "pose_estimation",
        "stroke_classification", "integrated", "data_mining",
    ]:
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)


def visualize_rally_filtering(capture: cv2.VideoCapture, intervals: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in intervals.itertuples(index=False):
        status = "accepted" if bool(row.accepted) else "rejected"
        path = OUTPUT / "rally_filtering" / f"rally_{int(row.rally_interval_id):03d}_{status}_sequence.jpg"
        if path.exists():
            records.append({"folder": "rally_filtering", "path": str(path.relative_to(ROOT)), "description": f"{status} rally sequence"})
            continue
        images = []
        for frame_index in sampled_indices(int(row.start_frame), int(row.end_frame)):
            frame = read_frame(capture, frame_index)
            add_title(frame, [f"Rally filter | interval {int(row.rally_interval_id):03d}", f"frame {frame_index} | accepted={row.accepted}"])
            images.append(frame)
        cv2.imwrite(str(path), sheet(images, columns=1))
        records.append({"folder": "rally_filtering", "path": str(path.relative_to(ROOT)), "description": f"{status} rally sequence"})
    return records


def load_court_corners() -> np.ndarray:
    row = pd.read_csv(HOMOGRAPHY).query("id == 44").iloc[0]
    return np.asarray(
        [
            [row.upleft_x, row.upleft_y], [row.upright_x, row.upright_y],
            [row.downright_x, row.downright_y], [row.downleft_x, row.downleft_y],
        ],
        dtype=np.int32,
    )


def visualize_court(capture: cv2.VideoCapture, intervals: pd.DataFrame) -> list[dict[str, str]]:
    corners = load_court_corners()
    selected = intervals[intervals["accepted"]].iloc[np.linspace(0, len(intervals[intervals["accepted"]]) - 1, 16, dtype=int)]
    records = []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        frame_index = (int(row.start_frame) + int(row.end_frame)) // 2
        path = OUTPUT / "court_detection" / f"court_candidate_{index:02d}_frame_{frame_index}.jpg"
        if path.exists():
            records.append({"folder": "court_detection", "path": str(path.relative_to(ROOT)), "description": "court polygon overlay"})
            continue
        frame = read_frame(capture, frame_index)
        cv2.polylines(frame, [corners], True, (50, 255, 70), 5, cv2.LINE_AA)
        for label, point in zip(["UL", "UR", "DR", "DL"], corners):
            cv2.circle(frame, tuple(point), 10, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, label, tuple(point + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        add_title(frame, ["Court detection / supplied homography", f"candidate {index:02d} | rally {int(row.rally_interval_id)} | frame {frame_index}"])
        cv2.imwrite(str(path), frame)
        records.append({"folder": "court_detection", "path": str(path.relative_to(ROOT)), "description": "court polygon overlay"})
    return records


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    frame = frame.rename(columns={"frame": "frame_index"})
    frame["frame_index"] = pd.to_numeric(frame["frame_index"], errors="coerce").fillna(-1).astype(int)
    return frame.set_index("frame_index")


def visualize_shuttle(capture: cv2.VideoCapture, manifest: pd.DataFrame) -> list[dict[str, str]]:
    selected = manifest.iloc[np.linspace(0, len(manifest) - 1, 16, dtype=int)]
    records = []
    for candidate, row in enumerate(selected.itertuples(index=False), 1):
        trajectory = load_trajectory(resolve(row.denoised_csv))
        images = []
        for frame_index in sampled_indices(int(row.start_frame), int(row.end_frame), 6):
            frame = read_frame(capture, frame_index)
            relative = frame_index - int(row.start_frame)
            history = trajectory.loc[max(0, relative - 25):relative]
            points = []
            for item in history.itertuples():
                if int(item.visibility) == 1:
                    points.append((int(float(item.x)), int(float(item.y))))
            for start, end in zip(points, points[1:]):
                cv2.line(frame, start, end, (0, 255, 255), 3, cv2.LINE_AA)
            if points:
                cv2.circle(frame, points[-1], 8, (0, 0, 255), -1, cv2.LINE_AA)
            add_title(frame, [f"Shuttle tracking | rally {row.rally_id}", f"frame {frame_index} | yellow=trail | red=current"])
            images.append(frame)
        path = OUTPUT / "shuttle_tracking" / f"shuttle_candidate_{candidate:02d}_rally_{row.rally_id}.jpg"
        cv2.imwrite(str(path), sheet(images))
        records.append({"folder": "shuttle_tracking", "path": str(path.relative_to(ROOT)), "description": "six-frame shuttle trail sequence"})
    return records


def draw_pose_panel(joints: np.ndarray, validity: np.ndarray, title: str) -> np.ndarray:
    panel = np.full((720, 640, 3), 25, dtype=np.uint8)
    cv2.putText(panel, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    for player in range(2):
        center = np.asarray([190 + player * 290, 380])
        points = joints[player] * 410 + center
        valid = np.any(joints[player] != 0, axis=1)
        for start, end in COCO_EDGES:
            if valid[start] and valid[end]:
                cv2.line(panel, tuple(points[start].astype(int)), tuple(points[end].astype(int)), PLAYER_COLORS[player], 3, cv2.LINE_AA)
        for point in points[valid]:
            cv2.circle(panel, tuple(point.astype(int)), 5, PLAYER_COLORS[player], -1, cv2.LINE_AA)
        state = "valid" if bool(validity[player]) else "missing"
        cv2.putText(panel, f"P{player + 1}: {state}", (105 + player * 290, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.65, PLAYER_COLORS[player], 2)
    return panel


def visualize_pose(capture: cv2.VideoCapture, manifest: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, str]]:
    merged = manifest.merge(validation[["clip_id", "both_pose_missing_rate", "p1_pose_valid_rate", "p2_pose_valid_rate"]], on="clip_id")
    merged = merged.sort_values("both_pose_missing_rate")
    indices = np.unique(np.linspace(0, len(merged) - 1, 18, dtype=int))
    records = []
    for candidate, (_, row) in enumerate(merged.iloc[indices].iterrows(), 1):
        joints = np.load(resolve(row.joints_npy))
        validity = np.load(resolve(row.validity_npz))
        pose_valid = np.stack([validity["p1_pose_valid"], validity["p2_pose_valid"]], axis=1)
        center = len(joints) // 2
        source = read_frame(capture, int(row.window_start_original) + center)
        add_title(source, [f"Pose estimation | {row.clip_id}", f"both-pose missing={row.both_pose_missing_rate:.1%}"])
        pose = draw_pose_panel(joints[center], pose_valid[center], "BST-normalized pose inference")
        combined = np.hstack([source, pose])
        path = OUTPUT / "pose_estimation" / f"pose_candidate_{candidate:02d}_{row.clip_id}.jpg"
        cv2.imwrite(str(path), combined)
        records.append({"folder": "pose_estimation", "path": str(path.relative_to(ROOT)), "description": f"pose example missing={row.both_pose_missing_rate:.1%}"})
    return records


def select_classification_examples(strokes: pd.DataFrame) -> pd.DataFrame:
    groups = [
        strokes[strokes["correct"]].nlargest(8, "confidence"),
        strokes[~strokes["correct"]].nlargest(8, "confidence"),
        strokes.nsmallest(8, "confidence"),
    ]
    return pd.concat(groups).drop_duplicates("clip_id").head(24)


def visualize_classification(capture: cv2.VideoCapture, strokes: pd.DataFrame) -> list[dict[str, str]]:
    records = []
    for candidate, row in enumerate(select_classification_examples(strokes).itertuples(index=False), 1):
        frame = read_frame(capture, int(row.event_frame_original))
        color = (60, 220, 80) if bool(row.correct) else (70, 70, 245)
        add_title(
            frame,
            [
                f"Stroke classification | candidate {candidate:02d} | {'correct' if row.correct else 'incorrect'}",
                f"GT: {row.true_label_name}",
                f"Pred: {row.predicted_label_name} | confidence={float(row.confidence):.3f}",
                f"Top2: {row.top2_label_name} ({float(row.top2_score):.3f}) | Top3: {row.top3_label_name} ({float(row.top3_score):.3f})",
            ],
            color,
        )
        path = OUTPUT / "stroke_classification" / f"stroke_candidate_{candidate:02d}_{row.clip_id}.jpg"
        cv2.imwrite(str(path), frame)
        records.append({"folder": "stroke_classification", "path": str(path.relative_to(ROOT)), "description": "classified event frame"})
    return records


def visualize_integrated(capture: cv2.VideoCapture, strokes: pd.DataFrame) -> list[dict[str, str]]:
    rally_sizes = strokes.groupby("rally_id").size().sort_values(ascending=False)
    selected = rally_sizes.head(14).index
    records = []
    for candidate, rally_id in enumerate(selected, 1):
        rally = strokes[strokes["rally_id"] == rally_id].sort_values("event_rank")
        rally = rally.iloc[np.unique(np.linspace(0, len(rally) - 1, min(8, len(rally)), dtype=int))]
        images = []
        for row in rally.itertuples(index=False):
            frame = read_frame(capture, int(row.event_frame_original))
            add_title(frame, [f"Integrated rally {rally_id} | event {row.event_rank}", f"{row.predicted_label_name} | conf={float(row.confidence):.2f}"])
            images.append(frame)
        path = OUTPUT / "integrated" / f"integrated_candidate_{candidate:02d}_rally_{rally_id}.jpg"
        cv2.imwrite(str(path), sheet(images, columns=4))
        records.append({"folder": "integrated", "path": str(path.relative_to(ROOT)), "description": "integrated predicted rally sequence"})
    return records


def save_plot(path: Path) -> dict[str, str]:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return {"folder": "data_mining", "path": str(path.relative_to(ROOT)), "description": path.stem.replace("_", " ")}


def visualize_mining(strokes: pd.DataFrame) -> list[dict[str, str]]:
    records = []
    sns.set_theme(style="whitegrid")

    distribution = strokes["predicted_stroke_type"].value_counts()
    plt.figure(figsize=(11, 6))
    sns.barplot(x=distribution.values, y=distribution.index, color="#3b82b8")
    plt.title("Video 44 predicted stroke distribution")
    plt.xlabel("Predicted events")
    records.append(save_plot(OUTPUT / "data_mining" / "01_stroke_distribution.png"))

    states = sorted(strokes["predicted_label_name"].unique())
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    pairs: Counter[tuple[str, str]] = Counter()
    motifs: Counter[tuple[str, str, str]] = Counter()
    for _, rally in strokes.sort_values(["rally_id", "event_rank"]).groupby("rally_id"):
        labels = rally["predicted_label_name"].tolist()
        pairs.update(zip(labels, labels[1:]))
        motifs.update(zip(labels, labels[1:], labels[2:]))
    for (source, target), count in pairs.items():
        matrix.loc[source, target] += count
    matrix = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    plt.figure(figsize=(13, 11))
    sns.heatmap(matrix, cmap="viridis", vmin=0)
    plt.title("Video 44 Markov transition probabilities")
    records.append(save_plot(OUTPUT / "data_mining" / "02_markov_transition_heatmap.png"))

    top_pairs = pd.Series({" -> ".join(pair): count for pair, count in pairs.most_common(15)})
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_pairs.values, y=top_pairs.index, color="#2a9d8f")
    plt.title("Video 44 strongest predicted transitions")
    records.append(save_plot(OUTPUT / "data_mining" / "03_strongest_transitions.png"))

    top_motifs = pd.Series({" -> ".join(motif): count for motif, count in motifs.most_common(15)})
    plt.figure(figsize=(13, 8))
    sns.barplot(x=top_motifs.values, y=top_motifs.index, color="#e09f3e")
    plt.title("Video 44 frequent three-stroke motifs")
    records.append(save_plot(OUTPUT / "data_mining" / "04_frequent_motifs.png"))

    rally_lengths = strokes.groupby("rally_id").size()
    plt.figure(figsize=(9, 5))
    sns.histplot(rally_lengths, bins=20, color="#6a4c93")
    plt.title("Video 44 predicted rally-length distribution")
    plt.xlabel("Predicted strokes per rally")
    records.append(save_plot(OUTPUT / "data_mining" / "05_rally_lengths.png"))

    plt.figure(figsize=(9, 5))
    sns.histplot(data=strokes, x="confidence", hue="correct", bins=20, multiple="stack")
    plt.title("Video 44 classification confidence and correctness")
    records.append(save_plot(OUTPUT / "data_mining" / "06_confidence_correctness.png"))
    return records


def main() -> None:
    ensure_dirs()
    intervals = pd.read_csv(RALLIES)
    intervals["accepted"] = intervals["accepted"].astype(str).str.lower().eq("true")
    shuttle = pd.read_csv(SHUTTLE_MANIFEST).query("video_id == 44")
    feature_manifest = pd.read_csv(FEATURE_MANIFEST).query("video_id == 44")
    feature_validation = pd.read_csv(FEATURE_VALIDATION).query("video_id == 44")
    strokes = pd.read_csv(STRUCTURED).query("video_id == 44")

    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    records: list[dict[str, str]] = []
    try:
        records += visualize_rally_filtering(capture, intervals)
        records += visualize_court(capture, intervals)
        records += visualize_shuttle(capture, shuttle)
        records += visualize_pose(capture, feature_manifest, feature_validation)
        records += visualize_classification(capture, strokes)
        records += visualize_integrated(capture, strokes)
    finally:
        capture.release()
    records += visualize_mining(strokes)

    with (OUTPUT / "visualization_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["folder", "path", "description"])
        writer.writeheader()
        writer.writerows(records)
    summary = Counter(record["folder"] for record in records)
    (OUTPUT / "visualization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
