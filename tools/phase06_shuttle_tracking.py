from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
TRACKNET_ROOT = ROOT / "external_repos" / "TrackNetV3"
DEFAULT_MODEL = TRACKNET_ROOT / "exp" / "model_best.pt"
DEFAULT_PHASE05_ROOT = ROOT / "project" / "outputs" / "rallies"
DEFAULT_RAW_VIDEO_ROOT = ROOT / "project" / "dataset" / "ShuttleSet_raw_videos"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "shuttle"

TRACKNET_HEIGHT = 288
TRACKNET_WIDTH = 512
RAW_COLUMNS = ["Frame", "Visibility", "X", "Y", "OriginalFrame", "TimeSec"]
METADATA_COLUMNS = [
    "video_id",
    "rally_id",
    "source_video",
    "start_frame",
    "end_frame",
    "fps",
    "width",
    "height",
    "frame_count",
    "decoder",
    "processed_frames",
    "device",
    "model_file",
    "raw_csv",
    "denoised_csv",
]


@dataclass(frozen=True)
class VideoTask:
    video_id: str
    rally_id: str
    video_path: Path
    start_frame: int
    end_frame: int | None


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def select_device(requested: str, gpu_id: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto" and not torch.cuda.is_available():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for Phase 06, but torch.cuda.is_available() is False.")
    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise ValueError(f"--gpu-id={gpu_id} is out of range; {device_count} CUDA device(s) available.")
    return torch.device(f"cuda:{gpu_id}")


def load_tracknet(model_file: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not model_file.exists():
        raise FileNotFoundError(f"TrackNetV3 checkpoint not found: {model_file}")

    sys.path.insert(0, str(TRACKNET_ROOT))
    from model import TrackNetV2

    checkpoint = torch.load(model_file, map_location=device)
    params = dict(checkpoint.get("param_dict", {}))
    num_frame = int(params.get("num_frame", 3))
    input_type = str(params.get("input_type", "2d"))
    model_name = str(params.get("model_name", "TrackNetV2"))
    if model_name != "TrackNetV2":
        raise ValueError(f"Unsupported TrackNet checkpoint model_name={model_name!r}")
    if input_type != "2d":
        raise ValueError(f"Unsupported TrackNet checkpoint input_type={input_type!r}")

    model = TrackNetV2(in_dim=num_frame * 3, out_dim=num_frame).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    params["num_frame"] = num_frame
    params["input_type"] = input_type
    params["model_name"] = model_name
    return model, params


def video_metadata(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    metadata = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    if metadata["fps"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
        raise RuntimeError(f"Incomplete video metadata for {video_path}: {metadata}")
    return metadata


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    den = float(denominator)
    return float(numerator) / den if den else 0.0


def ffprobe_stream(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    return dict(streams[0])


def ffprobe_metadata(video_path: Path) -> dict[str, Any]:
    stream = ffprobe_stream(video_path)
    fps = parse_rate(str(stream.get("avg_frame_rate", "")))
    frame_count = int(float(stream.get("nb_frames") or 0))
    if frame_count <= 0 and fps > 0:
        frame_count = int(round(float(stream.get("duration") or 0.0) * fps))
    metadata = {
        "fps": fps,
        "width": int(float(stream.get("width") or 0)),
        "height": int(float(stream.get("height") or 0)),
        "frame_count": frame_count,
    }
    if metadata["fps"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
        raise RuntimeError(f"Incomplete ffprobe metadata for {video_path}: {metadata}")
    return metadata


def ffprobe_codec_name(video_path: Path) -> str:
    return str(ffprobe_stream(video_path).get("codec_name", "")).lower()


def select_decoder(video_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    codec_name = ffprobe_codec_name(video_path)
    if codec_name == "av1":
        return "ffmpeg"
    if codec_name in {"h264", "avc1"}:
        return "opencv"
    return "opencv"


def resolve_video_by_id(raw_video_root: Path, video_id: str) -> Path:
    try:
        numeric = int(video_id)
    except ValueError:
        numeric = -1
    prefixes = [f"{numeric:02d} - ", f"{numeric} - "] if numeric >= 0 else [f"{video_id} - "]
    matches = [
        path
        for path in sorted(raw_video_root.glob("*.mp4"))
        if any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one MP4 for video_id={video_id}, found {len(matches)} under {raw_video_root}")
    return matches[0]


def normalize_phase05_video_id(phase05_root: Path, video_id: str) -> str:
    """Resolve CLI video ids like "1" to Phase 05's zero-padded directory "01"."""
    raw = str(video_id)
    direct_path = phase05_root / raw / "rally_intervals.csv"
    if direct_path.exists():
        return raw

    try:
        numeric = int(raw)
    except ValueError:
        return raw

    padded = f"{numeric:02d}"
    padded_path = phase05_root / padded / "rally_intervals.csv"
    if padded_path.exists():
        return padded
    return raw


def tasks_from_phase05(
    phase05_root: Path,
    raw_video_root: Path,
    video_ids: list[str],
    rally_ids: list[str],
    accepted_only: bool,
    max_rallies: int,
) -> list[VideoTask]:
    selected_video_ids = video_ids
    if not selected_video_ids:
        selected_video_ids = sorted(
            path.name
            for path in phase05_root.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "rally_intervals.csv").exists()
        )

    allowed_rallies = set(rally_ids)
    tasks: list[VideoTask] = []
    for requested_video_id in selected_video_ids:
        video_id = normalize_phase05_video_id(phase05_root, str(requested_video_id))
        intervals_path = phase05_root / video_id / "rally_intervals.csv"
        if not intervals_path.exists():
            available = sorted(
                path.name
                for path in phase05_root.iterdir()
                if path.is_dir() and path.name.isdigit() and (path / "rally_intervals.csv").exists()
            )
            raise FileNotFoundError(
                f"Missing Phase 05 intervals for explicitly selected video_id={requested_video_id}: {intervals_path}. "
                f"Available video ids with intervals: {', '.join(available) if available else 'none'}"
            )
        video_path = resolve_video_by_id(raw_video_root, video_id)
        for row in read_csv(intervals_path):
            rally_id = row["rally_interval_id"]
            if allowed_rallies and rally_id not in allowed_rallies:
                continue
            if accepted_only and not bool_from_csv(row.get("accepted", "")):
                continue
            tasks.append(
                VideoTask(
                    video_id=str(video_id),
                    rally_id=rally_id,
                    video_path=video_path,
                    start_frame=int(float(row["start_frame"])),
                    end_frame=int(float(row["end_frame"])),
                )
            )
            if max_rallies > 0 and len(tasks) >= max_rallies:
                return tasks
    return tasks


def task_from_input_video(input_video: Path, video_id: str, rally_id: str, start_frame: int, end_frame: int | None) -> VideoTask:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    return VideoTask(
        video_id=video_id,
        rally_id=rally_id,
        video_path=input_video,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def iter_video_frames_opencv(task: VideoTask, max_frames: int) -> Iterable[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(task.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {task.video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, task.start_frame)
    original_frame = task.start_frame
    emitted = 0
    try:
        while True:
            if task.end_frame is not None and original_frame > task.end_frame:
                break
            if max_frames > 0 and emitted >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield original_frame, frame
            emitted += 1
            original_frame += 1
    finally:
        cap.release()


def iter_video_frames_ffmpeg(
    task: VideoTask,
    metadata: dict[str, Any],
    max_frames: int,
) -> Iterable[tuple[int, np.ndarray]]:
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    end_frame = task.end_frame
    if end_frame is None:
        end_frame = max(int(metadata["frame_count"]) - 1, task.start_frame)
    requested_frames = max(0, end_frame - task.start_frame + 1)
    if max_frames > 0:
        requested_frames = min(requested_frames, max_frames)
    if requested_frames <= 0:
        return

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{task.start_frame / fps:.6f}",
        "-i",
        str(task.video_path),
        "-frames:v",
        str(requested_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_size = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created.")

    emitted = 0
    try:
        while emitted < requested_frames:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError(f"ffmpeg returned an incomplete frame for {task.video_path}")
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            yield task.start_frame + emitted, frame
            emitted += 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        if emitted == 0 and process.returncode not in (0, -15, None):
            raise RuntimeError(f"ffmpeg failed for {task.video_path}: {stderr}")


def iter_video_frames(
    task: VideoTask,
    metadata: dict[str, Any],
    max_frames: int,
    decoder: str,
) -> Iterable[tuple[int, np.ndarray]]:
    if decoder == "opencv":
        yield from iter_video_frames_opencv(task, max_frames=max_frames)
    elif decoder == "ffmpeg":
        yield from iter_video_frames_ffmpeg(task, metadata=metadata, max_frames=max_frames)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")


def make_batch(sequences: list[list[np.ndarray]]) -> torch.Tensor:
    batch = []
    for sequence in sequences:
        channels = np.array([], dtype=np.float32).reshape(0, TRACKNET_HEIGHT, TRACKNET_WIDTH)
        for frame in sequence:
            resized = cv2.resize(frame, (TRACKNET_WIDTH, TRACKNET_HEIGHT))
            resized = np.moveaxis(resized, -1, 0).astype(np.float32)
            channels = np.concatenate((channels, resized), axis=0)
        batch.append(channels / 255.0)
    return torch.from_numpy(np.stack(batch, axis=0)).float()


def object_center(heatmap: np.ndarray) -> tuple[int, int]:
    if np.amax(heatmap) == 0:
        return 0, 0
    contours, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0
    rects = [cv2.boundingRect(contour) for contour in contours]
    target = max(rects, key=lambda rect: rect[2] * rect[3])
    return int(target[0] + target[2] / 2), int(target[1] + target[3] / 2)


def infer_task(
    model: torch.nn.Module,
    task: VideoTask,
    metadata: dict[str, Any],
    decoder: str,
    num_frame: int,
    batch_size: int,
    threshold: float,
    max_frames: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_sequences: list[list[np.ndarray]] = []
    pending_original_frames: list[list[int]] = []
    frame_buffer: list[np.ndarray] = []
    original_buffer: list[int] = []
    x_scale = metadata["width"] / TRACKNET_WIDTH
    y_scale = metadata["height"] / TRACKNET_HEIGHT
    fps = metadata["fps"]

    def flush() -> None:
        nonlocal pending_sequences, pending_original_frames
        if not pending_sequences:
            return
        x = make_batch(pending_sequences).to(device)
        with torch.no_grad():
            y_pred = model(x)
        heatmaps = (y_pred.detach().cpu().numpy() > threshold).astype("uint8") * 255
        for sequence_index, frame_ids in enumerate(pending_original_frames):
            for offset, original_frame in enumerate(frame_ids):
                cx, cy = object_center(heatmaps[sequence_index, offset])
                x_pred = int(round(cx * x_scale)) if cx > 0 else 0
                y_pred_px = int(round(cy * y_scale)) if cy > 0 else 0
                visible = 1 if x_pred > 0 and y_pred_px > 0 else 0
                rows.append(
                    {
                        "Frame": original_frame - task.start_frame,
                        "Visibility": visible,
                        "X": x_pred,
                        "Y": y_pred_px,
                        "OriginalFrame": original_frame,
                        "TimeSec": round(original_frame / fps, 6) if fps else "",
                    }
                )
        pending_sequences = []
        pending_original_frames = []

    for original_frame, frame in iter_video_frames(task, metadata=metadata, max_frames=max_frames, decoder=decoder):
        frame_buffer.append(frame)
        original_buffer.append(original_frame)
        if len(frame_buffer) == num_frame:
            pending_sequences.append(list(frame_buffer))
            pending_original_frames.append(list(original_buffer))
            frame_buffer = []
            original_buffer = []
            if len(pending_sequences) == batch_size:
                flush()

    if frame_buffer:
        actual_original = list(original_buffer)
        while len(frame_buffer) < num_frame:
            frame_buffer.append(frame_buffer[-1])
            original_buffer.append(original_buffer[-1])
        pending_sequences.append(list(frame_buffer))
        pending_original_frames.append(actual_original)

    flush()
    rows.sort(key=lambda row: int(row["Frame"]))
    return rows


def denoise_rows(
    raw_rows: list[dict[str, Any]],
    width: int,
    height: int,
    max_jump_px: float,
    interpolate_gap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in raw_rows]
    removed_jump_count = 0
    out_of_range_count = 0

    last_visible: dict[str, Any] | None = None
    for row in rows:
        visible = int(row["Visibility"]) == 1
        x = float(row["X"])
        y = float(row["Y"])
        if visible and not (0 <= x <= width and 0 <= y <= height):
            row["Visibility"] = 0
            row["X"] = 0
            row["Y"] = 0
            out_of_range_count += 1
            visible = False
        if visible and last_visible is not None:
            dist = math.hypot(x - float(last_visible["X"]), y - float(last_visible["Y"]))
            frame_gap = max(int(row["Frame"]) - int(last_visible["Frame"]), 1)
            if dist / frame_gap > max_jump_px:
                row["Visibility"] = 0
                row["X"] = 0
                row["Y"] = 0
                removed_jump_count += 1
                visible = False
        if visible:
            last_visible = row

    interpolated_count = 0
    idx = 0
    while idx < len(rows):
        if int(rows[idx]["Visibility"]) == 1:
            idx += 1
            continue
        gap_start = idx
        while idx < len(rows) and int(rows[idx]["Visibility"]) == 0:
            idx += 1
        gap_end = idx - 1
        gap_len = gap_end - gap_start + 1
        prev_idx = gap_start - 1
        next_idx = idx
        if (
            0 <= prev_idx < len(rows)
            and next_idx < len(rows)
            and int(rows[prev_idx]["Visibility"]) == 1
            and int(rows[next_idx]["Visibility"]) == 1
            and gap_len <= interpolate_gap
        ):
            prev_row = rows[prev_idx]
            next_row = rows[next_idx]
            denom = int(next_row["Frame"]) - int(prev_row["Frame"])
            if denom > 0:
                for fill_idx in range(gap_start, gap_end + 1):
                    alpha = (int(rows[fill_idx]["Frame"]) - int(prev_row["Frame"])) / denom
                    rows[fill_idx]["X"] = round(float(prev_row["X"]) * (1 - alpha) + float(next_row["X"]) * alpha, 3)
                    rows[fill_idx]["Y"] = round(float(prev_row["Y"]) * (1 - alpha) + float(next_row["Y"]) * alpha, 3)
                    rows[fill_idx]["Visibility"] = 1
                    interpolated_count += 1

    stats = {
        "removed_jump_count": removed_jump_count,
        "out_of_range_count": out_of_range_count,
        "interpolated_count": interpolated_count,
    }
    return rows, stats


def validate_rows(rows: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "visible_rows": 0,
            "coverage": 0.0,
            "out_of_range_visible_rows": 0,
            "first_frame": "",
            "last_frame": "",
        }
    visible_rows = [row for row in rows if int(row["Visibility"]) == 1]
    out_of_range = [
        row
        for row in visible_rows
        if not (0 <= float(row["X"]) <= width and 0 <= float(row["Y"]) <= height)
    ]
    return {
        "rows": len(rows),
        "visible_rows": len(visible_rows),
        "coverage": len(visible_rows) / len(rows) if rows else 0.0,
        "out_of_range_visible_rows": len(out_of_range),
        "first_frame": int(rows[0]["Frame"]),
        "last_frame": int(rows[-1]["Frame"]),
    }


def process_task(
    model: torch.nn.Module,
    params: dict[str, Any],
    task: VideoTask,
    output_root: Path,
    model_file: Path,
    batch_size: int,
    threshold: float,
    max_frames: int,
    max_jump_px: float,
    interpolate_gap: int,
    device: torch.device,
    decoder_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    decoder_used = select_decoder(task.video_path, decoder_mode)
    metadata = ffprobe_metadata(task.video_path) if decoder_used == "ffmpeg" else video_metadata(task.video_path)
    end_frame = task.end_frame
    if end_frame is None:
        end_frame = max(metadata["frame_count"] - 1, task.start_frame)
    task = VideoTask(task.video_id, task.rally_id, task.video_path, task.start_frame, end_frame)

    video_dir = output_root / task.video_id
    raw_csv = video_dir / f"{task.rally_id}_ball_raw.csv"
    denoised_csv = video_dir / f"{task.rally_id}_ball_denoised.csv"
    metadata_json = video_dir / f"{task.rally_id}_metadata.json"

    raw_rows = infer_task(
        model=model,
        task=task,
        metadata=metadata,
        decoder=decoder_used,
        num_frame=int(params["num_frame"]),
        batch_size=batch_size,
        threshold=threshold,
        max_frames=max_frames,
        device=device,
    )
    denoised_rows, denoise_stats = denoise_rows(
        raw_rows,
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        max_jump_px=max_jump_px,
        interpolate_gap=interpolate_gap,
    )

    write_csv(raw_csv, raw_rows, RAW_COLUMNS)
    write_csv(denoised_csv, denoised_rows, RAW_COLUMNS)

    raw_validation = validate_rows(raw_rows, int(metadata["width"]), int(metadata["height"]))
    denoised_validation = validate_rows(denoised_rows, int(metadata["width"]), int(metadata["height"]))
    runtime_sec = time.time() - started
    task_metadata = {
        "video_id": task.video_id,
        "rally_id": task.rally_id,
        "source_video": str(task.video_path),
        "start_frame": task.start_frame,
        "end_frame": task.end_frame,
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "decoder": decoder_used,
        "processed_frames": len(raw_rows),
        "device": str(device),
        "model_file": str(model_file),
        "raw_csv": str(raw_csv),
        "denoised_csv": str(denoised_csv),
        "raw_validation": raw_validation,
        "denoised_validation": denoised_validation,
        "denoise_stats": denoise_stats,
        "runtime_sec": runtime_sec,
    }
    metadata_json.write_text(json.dumps(task_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_row = {
        "video_id": task.video_id,
        "rally_id": task.rally_id,
        "source_video": str(task.video_path),
        "start_frame": task.start_frame,
        "end_frame": task.end_frame,
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "decoder": decoder_used,
        "processed_frames": len(raw_rows),
        "device": str(device),
        "model_file": str(model_file),
        "raw_csv": str(raw_csv),
        "denoised_csv": str(denoised_csv),
    }
    validation_row = {
        "video_id": task.video_id,
        "rally_id": task.rally_id,
        "raw_rows": raw_validation["rows"],
        "raw_visible_rows": raw_validation["visible_rows"],
        "raw_coverage": raw_validation["coverage"],
        "raw_out_of_range_visible_rows": raw_validation["out_of_range_visible_rows"],
        "denoised_rows": denoised_validation["rows"],
        "denoised_visible_rows": denoised_validation["visible_rows"],
        "denoised_coverage": denoised_validation["coverage"],
        "denoised_out_of_range_visible_rows": denoised_validation["out_of_range_visible_rows"],
        "removed_jump_count": denoise_stats["removed_jump_count"],
        "out_of_range_count": denoise_stats["out_of_range_count"],
        "interpolated_count": denoise_stats["interpolated_count"],
        "runtime_sec": runtime_sec,
    }
    return summary_row, validation_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 06 shuttle tracking and denoising with TrackNetV3.")
    parser.add_argument("--phase05-root", type=Path, default=DEFAULT_PHASE05_ROOT)
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEO_ROOT)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", default=[], help="Phase 05/ShuttleSet video id. Repeatable.")
    parser.add_argument("--rally-id", action="append", default=[], help="Phase 05 rally_interval_id. Repeatable.")
    parser.add_argument("--max-rallies", type=int, default=1, help="Maximum Phase 05 accepted intervals to process; 0 means all selected.")
    parser.add_argument("--include-rejected", action="store_true", help="Process rejected Phase 05 intervals too.")
    parser.add_argument("--input-video", type=Path, default=None, help="Direct video/rally clip input, bypassing Phase 05 intervals.")
    parser.add_argument("--direct-video-id", default="direct")
    parser.add_argument("--direct-rally-id", default="rally_0001")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index to use when device is auto/cuda.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug frame cap per rally; 0 means full interval.")
    parser.add_argument("--max-jump-px", type=float, default=100.0)
    parser.add_argument("--interpolate-gap", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    device = select_device(args.device, args.gpu_id)
    model, params = load_tracknet(args.model_file, device=device)

    if args.input_video is not None:
        tasks = [
            task_from_input_video(
                input_video=args.input_video,
                video_id=args.direct_video_id,
                rally_id=args.direct_rally_id,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
        ]
    else:
        tasks = tasks_from_phase05(
            phase05_root=args.phase05_root,
            raw_video_root=args.raw_video_root,
            video_ids=[str(v) for v in args.video_id],
            rally_ids=[str(r) for r in args.rally_id],
            accepted_only=not args.include_rejected,
            max_rallies=args.max_rallies,
        )

    if not tasks:
        raise RuntimeError("No Phase 06 tasks selected.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        print(
            f"[phase06] {task_index}/{len(tasks)} "
            f"video={task.video_id} rally={task.rally_id} "
            f"frames={task.start_frame}-{task.end_frame}",
            flush=True,
        )
        summary_row, validation_row = process_task(
            model=model,
            params=params,
            task=task,
            output_root=args.output_root,
            model_file=args.model_file,
            batch_size=args.batch_size,
            threshold=args.threshold,
            max_frames=args.max_frames,
            max_jump_px=args.max_jump_px,
            interpolate_gap=args.interpolate_gap,
            device=device,
            decoder_mode=args.decoder,
        )
        summary_rows.append(summary_row)
        validation_rows.append(validation_row)
        print(
            f"[phase06] done {task_index}/{len(tasks)} "
            f"video={task.video_id} rally={task.rally_id} "
            f"decoder={summary_row['decoder']} "
            f"processed_frames={summary_row['processed_frames']} "
            f"denoised_coverage={float(validation_row['denoised_coverage']):.4f} "
            f"runtime_sec={float(validation_row['runtime_sec']):.2f}",
            flush=True,
        )

    write_csv(args.output_root / "phase06_shuttle_tracking_manifest.csv", summary_rows, METADATA_COLUMNS)
    write_csv(
        args.output_root / "phase06_shuttle_tracking_validation.csv",
        validation_rows,
        [
            "video_id",
            "rally_id",
            "raw_rows",
            "raw_visible_rows",
            "raw_coverage",
            "raw_out_of_range_visible_rows",
            "denoised_rows",
            "denoised_visible_rows",
            "denoised_coverage",
            "denoised_out_of_range_visible_rows",
            "removed_jump_count",
            "out_of_range_count",
            "interpolated_count",
            "runtime_sec",
        ],
    )

    processed_frames = int(sum(int(row["processed_frames"]) for row in summary_rows))
    total_raw_visible_rows = int(sum(int(row["raw_visible_rows"]) for row in validation_rows))
    total_denoised_visible_rows = int(sum(int(row["denoised_visible_rows"]) for row in validation_rows))
    aggregate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "tracknet_root": str(TRACKNET_ROOT),
        "model_file": str(args.model_file),
        "device": str(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "runtime_config": {
            "phase05_root": str(args.phase05_root),
            "raw_video_root": str(args.raw_video_root),
            "output_root": str(args.output_root),
            "video_ids": [str(v) for v in args.video_id],
            "rally_ids": [str(r) for r in args.rally_id],
            "max_rallies": args.max_rallies,
            "include_rejected": args.include_rejected,
            "input_video": str(args.input_video) if args.input_video is not None else "",
            "direct_video_id": args.direct_video_id,
            "direct_rally_id": args.direct_rally_id,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "requested_device": args.device,
            "gpu_id": args.gpu_id,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "decoder": args.decoder,
            "max_frames": args.max_frames,
            "max_jump_px": args.max_jump_px,
            "interpolate_gap": args.interpolate_gap,
        },
        "checkpoint_params": params,
        "task_count": len(tasks),
        "processed_frames": processed_frames,
        "total_raw_visible_rows": total_raw_visible_rows,
        "raw_coverage": total_raw_visible_rows / processed_frames if processed_frames else 0.0,
        "total_denoised_visible_rows": total_denoised_visible_rows,
        "denoised_coverage": total_denoised_visible_rows / processed_frames if processed_frames else 0.0,
        "removed_jump_count": int(sum(int(row["removed_jump_count"]) for row in validation_rows)),
        "out_of_range_count": int(sum(int(row["out_of_range_count"]) for row in validation_rows)),
        "interpolated_count": int(sum(int(row["interpolated_count"]) for row in validation_rows)),
        "runtime_sec_total": float(sum(float(row["runtime_sec"]) for row in validation_rows)),
        "outputs": {
            "manifest": str(args.output_root / "phase06_shuttle_tracking_manifest.csv"),
            "validation": str(args.output_root / "phase06_shuttle_tracking_validation.csv"),
        },
    }
    (args.output_root / "phase06_shuttle_tracking_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
