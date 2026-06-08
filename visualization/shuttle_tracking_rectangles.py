from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE06_ROOT = ROOT / "project" / "outputs" / "shuttle"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "visualizations" / "best_shuttle_rectangles"


@dataclass(frozen=True)
class Rally:
    video_id: str
    rally_id: str
    source_video: Path
    start_frame: int
    end_frame: int
    fps: float
    denoised_csv: Path
    denoised_coverage: float
    denoised_visible_rows: int
    processed_frames: int


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


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def load_rallies(phase06_root: Path, min_frames: int) -> list[Rally]:
    manifest_path = phase06_root / "phase06_shuttle_tracking_manifest.csv"
    validation_path = phase06_root / "phase06_shuttle_tracking_validation.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation CSV: {validation_path}")

    validation = {
        (row["video_id"], row["rally_id"]): row
        for row in read_csv(validation_path)
    }
    rallies: list[Rally] = []
    for row in read_csv(manifest_path):
        key = (row["video_id"], row["rally_id"])
        metrics = validation.get(key)
        if metrics is None:
            continue
        processed_frames = safe_int(row["processed_frames"])
        visible_rows = safe_int(metrics.get("denoised_visible_rows"))
        coverage = safe_float(metrics.get("denoised_coverage"))
        if processed_frames < min_frames or visible_rows < min_frames:
            continue
        denoised_csv = resolve_repo_path(row["denoised_csv"])
        source_video = resolve_repo_path(row["source_video"])
        if not denoised_csv.exists() or not source_video.exists():
            continue
        rallies.append(
            Rally(
                video_id=row["video_id"],
                rally_id=row["rally_id"],
                source_video=source_video,
                start_frame=safe_int(row["start_frame"]),
                end_frame=safe_int(row["end_frame"]),
                fps=safe_float(row["fps"], 30.0),
                denoised_csv=denoised_csv,
                denoised_coverage=coverage,
                denoised_visible_rows=visible_rows,
                processed_frames=processed_frames,
            )
        )
    return rallies


def load_visible_points(path: Path) -> list[dict[str, float]]:
    visible: list[dict[str, float]] = []
    for row in read_csv(path):
        visibility = safe_int(row.get("Visibility"))
        x = safe_float(row.get("X"))
        y = safe_float(row.get("Y"))
        original_frame = safe_int(row.get("OriginalFrame"))
        if visibility == 1 and x > 0 and y > 0:
            visible.append({"frame": float(original_frame), "x": x, "y": y})
    return visible


def sample_points(points: list[dict[str, float]], samples: int, trim_fraction: float) -> list[dict[str, float]]:
    if not points:
        return []
    trim = int(round(len(points) * max(0.0, min(trim_fraction, 0.4))))
    trimmed = points[trim : len(points) - trim] if len(points) - (2 * trim) >= samples else points
    if len(trimmed) <= samples:
        return trimmed
    indexes = np.linspace(0, len(trimmed) - 1, num=samples, dtype=int)
    return [trimmed[int(index)] for index in indexes]


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index}")
    return frame


def draw_label(frame: np.ndarray, lines: list[str]) -> None:
    overlay = frame.copy()
    x, y = 10, 10
    line_height = 22
    width = max(230, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] for line in lines) + 18)
    height = line_height * len(lines) + 14
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 8, y + 22 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_shuttle_rectangle(frame: np.ndarray, x: float, y: float, box_size: int) -> None:
    cx = int(round(x))
    cy = int(round(y))
    half = box_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame.shape[1] - 1, cx + half)
    y2 = min(frame.shape[0] - 1, cy + half)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def make_rally_strip(
    rally: Rally,
    output_root: Path,
    samples: int,
    thumb_width: int,
    box_size: int,
    trim_fraction: float,
) -> Path:
    points = sample_points(load_visible_points(rally.denoised_csv), samples, trim_fraction)
    if not points:
        raise RuntimeError(f"No visible denoised shuttle points for video {rally.video_id} rally {rally.rally_id}")

    cap = cv2.VideoCapture(str(rally.source_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {rally.source_video}")

    thumbs: list[np.ndarray] = []
    try:
        for idx, point in enumerate(points, start=1):
            frame_index = int(round(point["frame"]))
            frame = read_frame(cap, frame_index)
            draw_shuttle_rectangle(frame, point["x"], point["y"], box_size)
            draw_label(
                frame,
                [
                    f"V{rally.video_id} R{rally.rally_id}",
                    f"frame {frame_index}",
                    f"{idx}/{len(points)}",
                ],
            )
            thumb_height = int(round(thumb_width * frame.shape[0] / frame.shape[1]))
            thumbs.append(cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()

    header_height = 72
    sheet_width = thumb_width * len(thumbs)
    header = np.full((header_height, sheet_width, 3), 245, dtype=np.uint8)
    duration = (rally.end_frame - rally.start_frame + 1) / max(rally.fps, 1.0)
    title = (
        f"Video {rally.video_id}, rally {rally.rally_id} | "
        f"denoised coverage {rally.denoised_coverage:.1%} | "
        f"{rally.denoised_visible_rows}/{rally.processed_frames} visible frames | "
        f"{duration:.1f}s"
    )
    cv2.putText(header, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        header,
        "Frames are ordered horizontally from left to right; red/white rectangle marks the denoised shuttle.",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    strip = np.vstack([header, np.hstack(thumbs)])
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"video_{rally.video_id}_rally_{rally.rally_id}_shuttle_rectangles.jpg"
    cv2.imwrite(str(out), strip)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create horizontal shuttle-tracking frame strips with rectangles around the denoised shuttle."
    )
    parser.add_argument("--phase06-root", type=Path, default=DEFAULT_PHASE06_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--box-size", type=int, default=42)
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.12,
        help="Skip this fraction of visible points at each rally end before sampling frames.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Optional video id filter. Repeat for multiple videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_filter = set(args.video_id)
    rallies = load_rallies(args.phase06_root, args.min_frames)
    if video_filter:
        rallies = [rally for rally in rallies if rally.video_id in video_filter]
    if not rallies:
        raise RuntimeError("No eligible rallies found.")

    # Prefer long, high-coverage rallies: these make clearer contact sheets.
    ranked = sorted(
        rallies,
        key=lambda rally: (rally.denoised_coverage, rally.denoised_visible_rows, rally.processed_frames),
        reverse=True,
    )
    written: list[Path] = []
    for rally in ranked[: max(1, args.top_k)]:
        written.append(
            make_rally_strip(
                rally=rally,
                output_root=args.output_root,
                samples=max(2, args.samples),
                thumb_width=args.thumb_width,
                box_size=args.box_size,
                trim_fraction=args.trim_fraction,
            )
        )

    print("Wrote shuttle rectangle visualizations:")
    for path in written:
        print(f"- {repo_relative(path)}")


if __name__ == "__main__":
    main()
