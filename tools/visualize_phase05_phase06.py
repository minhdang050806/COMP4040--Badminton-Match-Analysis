from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE05_ROOT = ROOT / "project" / "outputs" / "rallies"
DEFAULT_PHASE06_ROOT = ROOT / "project" / "outputs" / "shuttle"
DEFAULT_RAW_VIDEO_ROOT = ROOT / "project" / "dataset" / "ShuttleSet_raw_videos"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "visualizations"


@dataclass(frozen=True)
class RallySelection:
    video_id: str
    rally_id: str
    source_video: Path
    start_frame: int
    end_frame: int
    fps: float
    width: int
    height: int
    denoised_csv: Path | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def video_metadata(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
        width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Incomplete video metadata for {video_path}")
        return fps, width, height
    finally:
        cap.release()


def resolve_video_by_id(raw_video_root: Path, video_id: str) -> Path:
    numeric = safe_int(video_id, -1)
    prefixes = [f"{numeric:02d} - ", f"{numeric} - "] if numeric >= 0 else [f"{video_id} - "]
    matches = [
        path
        for path in sorted(raw_video_root.glob("*.mp4"))
        if any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one MP4 for video_id={video_id}, found {len(matches)} under {raw_video_root}")
    return matches[0]


def text_box(frame: np.ndarray, lines: list[str], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = 18, 24
    line_height = 25
    width = max(320, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0] for line in lines) + 24)
    height = line_height * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 22), (x - 8 + width, y - 22 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def load_phase05_interval(phase05_root: Path, video_id: str, rally_id: str) -> dict[str, str]:
    interval_path = phase05_root / video_id / "rally_intervals.csv"
    if not interval_path.exists():
        raise FileNotFoundError(f"Missing Phase 05 interval CSV: {interval_path}")
    for row in read_csv(interval_path):
        if row["rally_interval_id"] == rally_id:
            return row
    raise FileNotFoundError(f"Rally {rally_id} not found in {interval_path}")


def select_phase05_rally(phase05_root: Path, raw_video_root: Path, video_id: str, rally_id: str) -> RallySelection:
    row = load_phase05_interval(phase05_root, video_id, rally_id)
    source_video = resolve_video_by_id(raw_video_root, video_id)
    fps, width, height = video_metadata(source_video)
    return RallySelection(
        video_id=video_id,
        rally_id=rally_id,
        source_video=source_video,
        start_frame=safe_int(row["start_frame"]),
        end_frame=safe_int(row["end_frame"]),
        fps=fps,
        width=width,
        height=height,
    )


def select_phase06_rally(phase06_root: Path, video_id: str, rally_id: str | None) -> RallySelection:
    manifest_path = phase06_root / "phase06_shuttle_tracking_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase 06 manifest: {manifest_path}")
    matches = [row for row in read_csv(manifest_path) if row["video_id"] == video_id]
    if rally_id is not None:
        matches = [row for row in matches if row["rally_id"] == rally_id]
    if not matches:
        raise FileNotFoundError(f"No Phase 06 rally found for video_id={video_id}, rally_id={rally_id or 'first available'}")
    row = matches[0]
    denoised_csv = Path(row["denoised_csv"])
    if not denoised_csv.is_absolute():
        denoised_csv = ROOT / denoised_csv
    return RallySelection(
        video_id=row["video_id"],
        rally_id=row["rally_id"],
        source_video=Path(row["source_video"]),
        start_frame=safe_int(row["start_frame"]),
        end_frame=safe_int(row["end_frame"]),
        fps=safe_float(row["fps"], 30.0),
        width=safe_int(row["width"]),
        height=safe_int(row["height"]),
        denoised_csv=denoised_csv,
    )


def draw_phase05_timeline(phase05_root: Path, video_id: str, output_root: Path) -> Path:
    states_path = phase05_root / video_id / "sampled_shot_angle_states.csv"
    intervals_path = phase05_root / video_id / "rally_intervals.csv"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing sampled state CSV: {states_path}")
    if not intervals_path.exists():
        raise FileNotFoundError(f"Missing interval CSV: {intervals_path}")

    states = read_csv(states_path)
    intervals = read_csv(intervals_path)
    frames = [safe_int(row["frame_index"]) for row in states]
    if not frames:
        raise RuntimeError(f"No sampled frames in {states_path}")

    min_frame, max_frame = min(frames), max(frames)
    width, height = 1800, 520
    left, right = 90, 40
    top, bottom = 76, 60
    plot_w = width - left - right
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    def x_for(frame: int) -> int:
        if max_frame == min_frame:
            return left
        return int(left + (frame - min_frame) / (max_frame - min_frame) * plot_w)

    cv2.putText(img, f"Phase 05 rally filtering timeline: video {video_id}", (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, "raw_sa", (18, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, "smooth", (18, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, "intervals", (18, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.line(img, (left, 160), (width - right, 160), (205, 205, 205), 1)
    cv2.line(img, (left, 255), (width - right, 255), (205, 205, 205), 1)
    cv2.line(img, (left, 365), (width - right, 365), (205, 205, 205), 1)

    for row in states:
        x = x_for(safe_int(row["frame_index"]))
        raw_active = safe_int(row["raw_sa"]) == 1
        smooth_active = row["smoothed_sa"] != "" and safe_int(row["smoothed_sa"]) == 1
        cv2.line(img, (x, 160), (x, 110 if raw_active else 150), (30, 115, 210) if raw_active else (185, 185, 185), 1)
        cv2.line(img, (x, 255), (x, 205 if smooth_active else 245), (0, 150, 110) if smooth_active else (185, 185, 185), 1)

    for row in intervals:
        start = x_for(safe_int(row["start_frame"]))
        end = max(start + 2, x_for(safe_int(row["end_frame"])))
        accepted = bool_from_csv(row["accepted"])
        color = (45, 150, 70) if accepted else (60, 80, 220)
        y1, y2 = (330, 382) if accepted else (390, 430)
        cv2.rectangle(img, (start, y1), (end, y2), color, -1)
        cv2.rectangle(img, (start, y1), (end, y2), (255, 255, 255), 1)

    cv2.putText(img, f"frames {min_frame}-{max_frame} | intervals {len(intervals)} | accepted {sum(bool_from_csv(row['accepted']) for row in intervals)}", (left, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 2, cv2.LINE_AA)
    out = output_path(output_root / f"phase05_video_{video_id}_timeline.png")
    cv2.imwrite(str(out), img)
    return out


def draw_phase05_contact_sheet(phase05_root: Path, selection: RallySelection, output_root: Path) -> Path:
    row = load_phase05_interval(phase05_root, selection.video_id, selection.rally_id)
    start = safe_int(row["start_frame"])
    end = safe_int(row["end_frame"])
    middle = (start + end) // 2
    status = "accepted" if bool_from_csv(row["accepted"]) else f"rejected:{row.get('reject_reason', '')}"
    frames = []
    for label, frame_index in [("start", start), ("middle", middle), ("end", end)]:
        frame = read_frame(selection.source_video, frame_index)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (45, 150, 70) if status == "accepted" else (60, 80, 220), 8)
        text_box(
            frame,
            [
                f"Phase 05 {label}",
                f"video {selection.video_id} rally {selection.rally_id} | {status}",
                f"frame {frame_index} | {safe_float(row['duration_sec']):.2f}s",
            ],
        )
        frames.append(cv2.resize(frame, (480, 270)))
    sheet = np.hstack(frames)
    out = output_path(output_root / f"phase05_video_{selection.video_id}_rally_{selection.rally_id}_contact.jpg")
    cv2.imwrite(str(out), sheet)
    return out


def load_trajectory(path: Path) -> dict[int, tuple[int, float, float]]:
    rows = read_csv(path)
    trajectory: dict[int, tuple[int, float, float]] = {}
    for row in rows:
        original_frame = safe_int(row["OriginalFrame"])
        trajectory[original_frame] = (safe_int(row["Visibility"]), safe_float(row["X"]), safe_float(row["Y"]))
    return trajectory


def draw_shuttle_overlay(
    frame: np.ndarray,
    frame_index: int,
    trajectory: dict[int, tuple[int, float, float]],
    trail: int,
    selection: RallySelection,
) -> None:
    points: list[tuple[int, int]] = []
    for idx in range(max(selection.start_frame, frame_index - trail), frame_index + 1):
        visible, x, y = trajectory.get(idx, (0, 0.0, 0.0))
        if visible and x > 0 and y > 0:
            points.append((int(round(x)), int(round(y))))
    for prev, cur in zip(points, points[1:]):
        cv2.line(frame, prev, cur, (0, 210, 255), 3, cv2.LINE_AA)
    if points:
        cv2.circle(frame, points[-1], 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, points[-1], 13, (255, 255, 255), 2, cv2.LINE_AA)
    text_box(
        frame,
        [
            "Denoised shuttle trajectory",
            f"video {selection.video_id} rally {selection.rally_id}",
            f"frame {frame_index} / {selection.start_frame}-{selection.end_frame}",
        ],
    )


def draw_phase06_contact_sheet(selection: RallySelection, output_root: Path, samples: int, trail: int) -> Path:
    if selection.denoised_csv is None:
        raise RuntimeError("Phase 06 contact sheet requires a denoised trajectory CSV.")
    trajectory = load_trajectory(selection.denoised_csv)
    frame_indices = np.linspace(selection.start_frame, selection.end_frame, num=max(2, samples), dtype=int).tolist()
    thumbs = []
    for frame_index in frame_indices:
        frame = read_frame(selection.source_video, frame_index)
        draw_shuttle_overlay(frame, frame_index, trajectory, trail, selection)
        thumbs.append(cv2.resize(frame, (426, 240)))
    rows = []
    for idx in range(0, len(thumbs), 3):
        chunk = thumbs[idx : idx + 3]
        if len(chunk) < 3:
            chunk.extend([np.full_like(thumbs[0], 245) for _ in range(3 - len(chunk))])
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    out = output_path(output_root / f"phase06_video_{selection.video_id}_rally_{selection.rally_id}_contact.jpg")
    cv2.imwrite(str(out), sheet)
    return out


def phase06_validation_lookup(phase06_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    validation_path = phase06_root / "phase06_shuttle_tracking_validation.csv"
    if not validation_path.exists():
        return {}
    return {(row["video_id"], row["rally_id"]): row for row in read_csv(validation_path)}


def draw_phase06_multi_rally_sheet(
    selections: list[RallySelection],
    phase06_root: Path,
    output_root: Path,
    samples: int,
    trail: int,
) -> Path:
    if not selections:
        raise ValueError("At least one Phase 06 rally is required.")

    validation = phase06_validation_lookup(phase06_root)
    thumb_width, thumb_height = 360, 203
    label_width = 310
    rows: list[np.ndarray] = []
    for selection in selections:
        if selection.denoised_csv is None:
            raise RuntimeError("Phase 06 comparison requires denoised trajectory CSVs.")
        trajectory = load_trajectory(selection.denoised_csv)
        frame_indices = np.linspace(selection.start_frame, selection.end_frame, num=max(2, samples), dtype=int)
        thumbs: list[np.ndarray] = []
        for sample_index, frame_index in enumerate(frame_indices):
            frame = read_frame(selection.source_video, int(frame_index))
            draw_shuttle_overlay(frame, int(frame_index), trajectory, trail, selection)
            cv2.putText(
                frame,
                f"{sample_index + 1}/{len(frame_indices)}",
                (frame.shape[1] - 90, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            thumbs.append(cv2.resize(frame, (thumb_width, thumb_height)))

        metrics = validation.get((selection.video_id, selection.rally_id), {})
        label = np.full((thumb_height, label_width, 3), 245, dtype=np.uint8)
        lines = [
            f"Video {selection.video_id}, rally {selection.rally_id}",
            f"Frames: {selection.start_frame}-{selection.end_frame}",
            f"Duration: {(selection.end_frame - selection.start_frame + 1) / selection.fps:.1f}s",
            f"Denoised coverage: {safe_float(metrics.get('denoised_coverage')):.1%}",
            f"Removed jumps: {safe_int(metrics.get('removed_jump_count'))}",
            f"Interpolated: {safe_int(metrics.get('interpolated_count'))}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                label,
                line,
                (14, 30 + index * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
        rows.append(np.hstack([label, *thumbs]))

    title_height = 80
    sheet_width = rows[0].shape[1]
    title = np.full((title_height, sheet_width, 3), 250, dtype=np.uint8)
    cv2.putText(
        title,
        "Tracked rallies: chronological frames shown left to right",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        title,
        "Yellow trail = recent denoised shuttle path; red point = current shuttle location",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    sheet = np.vstack([title, *rows])
    out = output_path(output_root / "phase06_rallies_side_by_side.jpg")
    cv2.imwrite(str(out), sheet)
    return out


def write_phase06_overlay_clip(selection: RallySelection, output_root: Path, stride: int, max_frames: int, trail: int) -> Path:
    if selection.denoised_csv is None:
        raise RuntimeError("Phase 06 overlay clip requires a denoised trajectory CSV.")
    trajectory = load_trajectory(selection.denoised_csv)
    cap = cv2.VideoCapture(str(selection.source_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {selection.source_video}")

    out = output_path(output_root / f"phase06_video_{selection.video_id}_rally_{selection.rally_id}_overlay.mp4")
    clip_fps = max(1.0, selection.fps / max(1, stride))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), clip_fps, (selection.width, selection.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not create video writer: {out}")

    written = 0
    frame_index = selection.start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    try:
        while frame_index <= selection.end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_index - selection.start_frame) % max(1, stride) == 0:
                draw_shuttle_overlay(frame, frame_index, trajectory, trail, selection)
                writer.write(frame)
                written += 1
                if max_frames > 0 and written >= max_frames:
                    break
            frame_index += 1
    finally:
        writer.release()
        cap.release()
    if written == 0:
        raise RuntimeError(f"No frames were written for {selection}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Phase 05 rally intervals and Phase 06 shuttle trajectories.")
    parser.add_argument("--video-id", default="40", help="ShuttleSet video id, e.g. 40.")
    parser.add_argument("--rally-id", default="1", help="Rally interval id. Use Phase 06 manifest ids for Phase 06 overlays.")
    parser.add_argument("--phase", choices=["5", "6", "both"], default="both")
    parser.add_argument("--phase05-root", type=Path, default=DEFAULT_PHASE05_ROOT)
    parser.add_argument("--phase06-root", type=Path, default=DEFAULT_PHASE06_ROOT)
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contact-samples", type=int, default=6)
    parser.add_argument("--clip-stride", type=int, default=3, help="Write every Nth source frame to Phase 06 overlay MP4.")
    parser.add_argument("--max-clip-frames", type=int, default=240, help="0 writes the full selected rally.")
    parser.add_argument("--trail", type=int, default=20, help="Number of previous frames shown as shuttle trail.")
    parser.add_argument(
        "--compare-rally",
        action="append",
        default=[],
        metavar="VIDEO_ID:RALLY_ID",
        help="Repeat to create one Phase 06 sheet with each rally as a chronological row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    written: list[Path] = []

    if args.phase in {"5", "both"}:
        phase05_selection = select_phase05_rally(args.phase05_root, args.raw_video_root, args.video_id, args.rally_id)
        written.append(draw_phase05_timeline(args.phase05_root, args.video_id, output_root))
        written.append(draw_phase05_contact_sheet(args.phase05_root, phase05_selection, output_root))
    if args.phase in {"6", "both"}:
        phase06_selection = select_phase06_rally(args.phase06_root, args.video_id, args.rally_id)
        written.append(draw_phase06_contact_sheet(phase06_selection, output_root, args.contact_samples, args.trail))
        written.append(write_phase06_overlay_clip(phase06_selection, output_root, args.clip_stride, args.max_clip_frames, args.trail))
        if args.compare_rally:
            selections = []
            for value in args.compare_rally:
                if ":" not in value:
                    raise ValueError(f"Expected --compare-rally VIDEO_ID:RALLY_ID, got {value!r}")
                video_id, rally_id = value.split(":", 1)
                selections.append(select_phase06_rally(args.phase06_root, video_id, rally_id))
            written.append(draw_phase06_multi_rally_sheet(selections, args.phase06_root, output_root, args.contact_samples, args.trail))

    print("Wrote visualization artifacts:")
    for path in written:
        print(f"- {repo_relative(path)}")


if __name__ == "__main__":
    main()
