from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
AUTOMATED_ROOT = ROOT / "external_repos" / "Automated-Hit-frame-Detection-for-Badminton-Match-Analysis"
AUTOMATED_SRC = AUTOMATED_ROOT / "src"
DEFAULT_RAW_VIDEOS = ROOT / "project" / "dataset" / "ShuttleSet_raw_videos"
DEFAULT_MATCH_CSV = ROOT / "project" / "dataset" / "ShuttleSet" / "set" / "match.csv"
DEFAULT_GT_STROKES = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_RALLY_SUMMARY = ROOT / "project" / "outputs" / "tables" / "phase02_rally_summary.csv"
DEFAULT_UPSTREAM_SACNN = AUTOMATED_SRC / "models" / "weights" / "sacnn.pt"
DEFAULT_FINETUNED_SACNN = ROOT / "project" / "weights" / "sacnn_shuttleset_finetuned_protocol.pt"
DEFAULT_SACNN = DEFAULT_FINETUNED_SACNN
DEFAULT_TRAINING_SUMMARY = ROOT / "project" / "outputs" / "sacnn_training_protocol" / "training_summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "rallies"

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


SAMPLE_COLUMNS = [
    "video_id",
    "match_id",
    "sample_index",
    "frame_index",
    "timestamp_sec",
    "raw_sa",
    "smoothed_sa",
    "transition_code",
    "transition_name",
]

INTERVAL_COLUMNS = [
    "video_id",
    "match_id",
    "rally_interval_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "sample_count",
    "accepted",
    "reject_reason",
]

VALIDATION_COLUMNS = [
    "video_id",
    "match_id",
    "video_file",
    "fps",
    "width",
    "height",
    "frame_count",
    "decoder",
    "sample_stride_frames",
    "sample_period_sec",
    "sampled_frames",
    "sampled_start_frame",
    "sampled_end_frame",
    "raw_active_samples",
    "smoothed_active_samples",
    "predicted_intervals",
    "accepted_intervals",
    "rejected_intervals",
    "total_match_annotated_hit_rows",
    "annotated_hit_rows",
    "annotated_hits_inside_accepted_intervals",
    "annotated_hit_coverage",
    "total_match_annotated_rallies",
    "annotated_rallies",
    "annotated_rallies_overlapping_accepted_intervals",
    "annotated_rally_overlap_rate",
    "runtime_sec",
]


@dataclass(frozen=True)
class VideoJob:
    match_id: int
    video_id: str
    video_name: str
    video_path: Path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


class ShotAngleQueue:
    """Project-local copy of the upstream queue logic without RallyProcessor imports."""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.queue: list[list[Any]] = []
        self.last_sa = 0

    def push(self, frame_info: list[Any]) -> tuple[list[Any] | None, int | None]:
        if len(self.queue) < self.max_len:
            self.queue.append(frame_info)
            return None, None

        first_info = self.queue.pop(0)
        sa, sa_condition = self._check_sa_condition(int(first_info[0]))
        self.last_sa = sa
        first_info[0] = sa
        self.queue.append(frame_info)
        return first_info, sa_condition

    def _check_sa_condition(self, sa: int) -> tuple[int, int]:
        total = sa
        if self.last_sa == 1 and sa == 0:
            total += sum(int(info[0]) for info in self.queue)
            if total <= (self.max_len / 2):
                return 0, 3
            return 1, 2
        if self.last_sa == 0 and sa == 1:
            total += sum(int(info[0]) for info in self.queue)
            if total >= (self.max_len / 2):
                return 1, 1
            return 0, 0
        if self.last_sa == 1 and sa == 1:
            return 1, 2
        return 0, 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(AUTOMATED_SRC))
    from models.sacnn import SACNN

    model = SACNN().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_device(cpu: bool, gpu_id: int) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise RuntimeError(f"Requested --gpu-id {gpu_id}, but only {device_count} CUDA device(s) are visible.")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def load_training_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def frame_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((216, 384)),
            transforms.CenterCrop((216, 216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def discover_jobs(raw_video_root: Path, match_csv: Path, requested_ids: set[int] | None, max_videos: int | None) -> list[VideoJob]:
    match_rows = read_csv(match_csv)
    video_by_prefix = {}
    for path in sorted(raw_video_root.glob("*.mp4")):
        prefix = path.name.split(" - ", 1)[0]
        if prefix.isdigit():
            video_by_prefix[int(prefix)] = path

    jobs: list[VideoJob] = []
    for row in match_rows:
        match_id = safe_int(row["id"])
        if requested_ids and match_id not in requested_ids:
            continue
        video_path = video_by_prefix.get(match_id)
        if video_path is None:
            continue
        jobs.append(
            VideoJob(
                match_id=match_id,
                video_id=f"{match_id:02d}",
                video_name=row["video"],
                video_path=video_path,
            )
        )
        if max_videos is not None and len(jobs) >= max_videos:
            break
    return jobs


def video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    return VideoMetadata(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" not in value:
        return safe_float(value)
    numerator, denominator = value.split("/", 1)
    den = safe_float(denominator)
    return safe_float(numerator) / den if den else 0.0


def ffprobe_metadata(video_path: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    stream = streams[0]
    fps = parse_rate(str(stream.get("avg_frame_rate", "")))
    frame_count = safe_int(stream.get("nb_frames"))
    if frame_count <= 0 and fps > 0:
        frame_count = int(round(safe_float(stream.get("duration")) * fps))
    metadata = VideoMetadata(
        fps=fps,
        width=safe_int(stream.get("width")),
        height=safe_int(stream.get("height")),
        frame_count=frame_count,
    )
    if metadata.fps <= 0 or metadata.width <= 0 or metadata.height <= 0:
        raise RuntimeError(f"ffprobe metadata is invalid for {video_path}: {metadata}")
    return metadata


def ffprobe_codec_name(video_path: Path) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().lower()


def opencv_can_decode_first_frame(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    ok, _frame = cap.read()
    cap.release()
    return bool(ok)


def iter_sampled_frames(
    cap: cv2.VideoCapture,
    frame_count: int,
    sample_stride: int,
    max_samples: int | None,
) -> Iterable[tuple[int, np.ndarray]]:
    sample_index = 0
    frame_index = 0
    while frame_index < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_stride == 0:
            yield frame_index, frame
            sample_index += 1
            if max_samples is not None and sample_index >= max_samples:
                break
        frame_index += 1


def iter_sampled_frames_ffmpeg(
    video_path: Path,
    metadata: VideoMetadata,
    sample_stride: int,
    max_samples: int | None,
) -> Iterable[tuple[int, np.ndarray]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_size = metadata.width * metadata.height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created.")

    sample_index = 0
    frame_index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError(f"ffmpeg returned an incomplete frame for {video_path}")
            if frame_index % sample_stride == 0:
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((metadata.height, metadata.width, 3))
                yield frame_index, frame
                sample_index += 1
                if max_samples is not None and sample_index >= max_samples:
                    break
            frame_index += 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            if process.returncode not in (0, -15, None) and stderr:
                raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")


def predict_video_states(
    model: torch.nn.Module,
    device: torch.device,
    job: VideoJob,
    sample_period_sec: float,
    batch_size: int,
    queue_length: int,
    max_samples: int | None,
    decoder: str,
) -> tuple[VideoMetadata, list[dict[str, Any]], float, str]:
    start = time.time()
    decoder_used = decoder
    if decoder == "auto":
        codec_name = ffprobe_codec_name(job.video_path)
        if codec_name == "av1":
            decoder_used = "ffmpeg"
        else:
            decoder_used = "opencv" if opencv_can_decode_first_frame(job.video_path) else "ffmpeg"

    if decoder_used == "opencv":
        cap = cv2.VideoCapture(str(job.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {job.video_path}")
        metadata = video_metadata(cap)
        if metadata.fps <= 0 or metadata.frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video metadata is invalid for {job.video_path}")
    elif decoder_used == "ffmpeg":
        cap = None
        metadata = ffprobe_metadata(job.video_path)
    else:
        raise RuntimeError(f"Unknown decoder: {decoder}")

    sample_stride = max(1, int(round(metadata.fps * sample_period_sec)))
    preprocess = frame_preprocess()
    queue = ShotAngleQueue(queue_length)
    rows: list[dict[str, Any]] = []
    batch_frames: list[torch.Tensor] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch_frames:
            return
        batch = torch.stack(batch_frames, dim=0).to(device)
        with torch.no_grad():
            raw_predictions = torch.argmax(model(batch), dim=1).detach().cpu().numpy().astype(int)
        for frame_index, raw_sa in zip(batch_indices, raw_predictions):
            sample_index = len(rows)
            frame_info, condition = queue.push([int(raw_sa), frame_index, sample_index])
            if frame_info is None:
                smoothed_sa = None
                output_frame = frame_index
                output_sample_index = sample_index
            else:
                smoothed_sa = int(frame_info[0])
                output_frame = int(frame_info[1])
                output_sample_index = int(frame_info[2])
            rows.append(
                {
                    "video_id": job.video_id,
                    "match_id": job.match_id,
                    "sample_index": output_sample_index,
                    "frame_index": output_frame,
                    "timestamp_sec": output_frame / metadata.fps,
                    "raw_sa": int(raw_sa),
                    "smoothed_sa": "" if smoothed_sa is None else smoothed_sa,
                    "transition_code": "" if condition is None else condition,
                    "transition_name": transition_name(condition),
                }
            )
        batch_frames.clear()
        batch_indices.clear()

    if decoder_used == "opencv":
        assert cap is not None
        frame_iter = iter_sampled_frames(cap, metadata.frame_count, sample_stride, max_samples)
    else:
        frame_iter = iter_sampled_frames_ffmpeg(job.video_path, metadata, sample_stride, max_samples)

    for frame_index, frame in frame_iter:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(preprocess(Image.fromarray(rgb)))
        batch_indices.append(frame_index)
        if len(batch_frames) >= batch_size:
            flush_batch()

    flush_batch()
    if cap is not None:
        cap.release()
    return metadata, rows, time.time() - start, decoder_used


def transition_name(code: int | None) -> str:
    return {
        None: "queued",
        0: "inactive_to_inactive",
        1: "inactive_to_active",
        2: "active_to_active",
        3: "active_to_inactive",
    }[code]


def build_intervals(
    job: VideoJob,
    sample_rows: list[dict[str, Any]],
    fps: float,
    min_rally_sec: float,
    max_rally_sec: float,
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    active_start: int | None = None
    active_samples = 0

    processed_rows = [row for row in sample_rows if row["smoothed_sa"] != ""]
    for row in processed_rows:
        code = row["transition_code"]
        frame_index = int(row["frame_index"])
        if code == 1 and active_start is None:
            active_start = frame_index
            active_samples = 1
        elif code == 2 and active_start is not None:
            active_samples += 1
        elif code == 3 and active_start is not None:
            intervals.append(make_interval_row(job, len(intervals) + 1, active_start, frame_index, fps, active_samples))
            active_start = None
            active_samples = 0

    if active_start is not None and processed_rows:
        end_frame = int(processed_rows[-1]["frame_index"])
        intervals.append(make_interval_row(job, len(intervals) + 1, active_start, end_frame, fps, active_samples))

    for row in intervals:
        duration = float(row["duration_sec"])
        accepted = min_rally_sec <= duration <= max_rally_sec
        row["accepted"] = accepted
        if accepted:
            row["reject_reason"] = ""
        elif duration < min_rally_sec:
            row["reject_reason"] = "too_short"
        else:
            row["reject_reason"] = "too_long"
    return intervals


def make_interval_row(
    job: VideoJob,
    interval_id: int,
    start_frame: int,
    end_frame: int,
    fps: float,
    sample_count: int,
) -> dict[str, Any]:
    end_frame = max(end_frame, start_frame)
    return {
        "video_id": job.video_id,
        "match_id": job.match_id,
        "rally_interval_id": interval_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_frame / fps,
        "end_time_sec": end_frame / fps,
        "duration_sec": (end_frame - start_frame) / fps,
        "sample_count": sample_count,
        "accepted": True,
        "reject_reason": "",
    }


def intervals_contain(frame: int, intervals: list[dict[str, Any]]) -> bool:
    return any(bool(row["accepted"]) and int(row["start_frame"]) <= frame <= int(row["end_frame"]) for row in intervals)


def intervals_overlap(start: int, end: int, intervals: list[dict[str, Any]]) -> bool:
    for row in intervals:
        if not bool(row["accepted"]):
            continue
        pred_start = int(row["start_frame"])
        pred_end = int(row["end_frame"])
        if max(start, pred_start) <= min(end, pred_end):
            return True
    return False


def validation_row(
    job: VideoJob,
    metadata: VideoMetadata,
    sample_stride: int,
    sample_period_sec: float,
    sample_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    gt_rows: list[dict[str, str]],
    rally_rows: list[dict[str, str]],
    runtime_sec: float,
    decoder_used: str,
) -> dict[str, Any]:
    sampled_frames = [int(row["frame_index"]) for row in sample_rows]
    sampled_start = min(sampled_frames) if sampled_frames else 0
    sampled_end = max(sampled_frames) if sampled_frames else 0
    gt_for_match_all = [row for row in gt_rows if safe_int(row.get("match_id")) == job.match_id]
    rally_for_match_all = [row for row in rally_rows if safe_int(row.get("match_id")) == job.match_id]
    gt_for_match = [
        row
        for row in gt_for_match_all
        if sampled_start <= safe_int(row.get("frame_num_int")) <= sampled_end
    ]
    rally_for_match = [
        row
        for row in rally_for_match_all
        if max(sampled_start, safe_int(row.get("first_frame_num"))) <= min(sampled_end, safe_int(row.get("last_frame_num")))
    ]
    hit_frames = [safe_int(row.get("frame_num_int")) for row in gt_for_match]
    hit_inside = sum(1 for frame in hit_frames if intervals_contain(frame, intervals))
    rally_overlap = 0
    for row in rally_for_match:
        start = safe_int(row.get("first_frame_num"))
        end = safe_int(row.get("last_frame_num"))
        if intervals_overlap(start, end, intervals):
            rally_overlap += 1

    accepted = sum(1 for row in intervals if bool(row["accepted"]))
    rejected = len(intervals) - accepted
    raw_active = sum(1 for row in sample_rows if int(row["raw_sa"]) == 1)
    smoothed_active = sum(1 for row in sample_rows if row["smoothed_sa"] != "" and int(row["smoothed_sa"]) == 1)
    return {
        "video_id": job.video_id,
        "match_id": job.match_id,
        "video_file": relative(job.video_path),
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": metadata.frame_count,
        "decoder": decoder_used,
        "sample_stride_frames": sample_stride,
        "sample_period_sec": sample_period_sec,
        "sampled_frames": len(sample_rows),
        "sampled_start_frame": sampled_start,
        "sampled_end_frame": sampled_end,
        "raw_active_samples": raw_active,
        "smoothed_active_samples": smoothed_active,
        "predicted_intervals": len(intervals),
        "accepted_intervals": accepted,
        "rejected_intervals": rejected,
        "total_match_annotated_hit_rows": len(gt_for_match_all),
        "annotated_hit_rows": len(hit_frames),
        "annotated_hits_inside_accepted_intervals": hit_inside,
        "annotated_hit_coverage": hit_inside / len(hit_frames) if hit_frames else None,
        "total_match_annotated_rallies": len(rally_for_match_all),
        "annotated_rallies": len(rally_for_match),
        "annotated_rallies_overlapping_accepted_intervals": rally_overlap,
        "annotated_rally_overlap_rate": rally_overlap / len(rally_for_match) if rally_for_match else None,
        "runtime_sec": runtime_sec,
    }


def export_visual_review(
    output_root: Path,
    job: VideoJob,
    intervals: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    accepted = [row for row in intervals if bool(row["accepted"])]
    rejected = [row for row in intervals if not bool(row["accepted"])]
    selected = [(row, "accepted") for row in accepted[:max_items]] + [(row, "rejected") for row in rejected[:max_items]]
    if not selected:
        return []

    review_root = output_root / "visual_review" / job.video_id
    review_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(job.video_path))
    rows: list[dict[str, Any]] = []
    for interval, status in selected:
        interval_id = int(interval["rally_interval_id"])
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        mid = (start + end) // 2
        for frame_role, frame_index in [("start", start), ("middle", mid), ("end", end)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            label = f"video {job.video_id} interval {interval_id} {status} {frame_role} frame {frame_index}"
            cv2.putText(frame, label, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            out_path = review_root / f"interval_{interval_id:04d}_{status}_{frame_role}.jpg"
            cv2.imwrite(str(out_path), frame)
            rows.append(
                {
                    "video_id": job.video_id,
                    "match_id": job.match_id,
                    "rally_interval_id": interval_id,
                    "interval_status": status,
                    "frame_role": frame_role,
                    "frame_index": frame_index,
                    "image_path": relative(out_path),
                    "human_review_status": "pending",
                }
            )
    cap.release()
    return rows


def export_rally_clips(
    output_root: Path,
    job: VideoJob,
    intervals: list[dict[str, Any]],
    metadata: VideoMetadata,
) -> list[dict[str, Any]]:
    accepted = [row for row in intervals if bool(row["accepted"])]
    if not accepted:
        return []

    clips_root = output_root / job.video_id / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for interval in accepted:
        interval_id = int(interval["rally_interval_id"])
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        clip_path = clips_root / f"rally_{interval_id:04d}.mp4"
        cap = cv2.VideoCapture(str(job.video_path))
        if not cap.isOpened():
            rows.append(
                {
                    "video_id": job.video_id,
                    "match_id": job.match_id,
                    "rally_interval_id": interval_id,
                    "clip_path": relative(clip_path),
                    "start_frame": start,
                    "end_frame": end,
                    "status": "open_failed",
                }
            )
            continue
        writer = cv2.VideoWriter(str(clip_path), fourcc, metadata.fps, (metadata.width, metadata.height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        written = 0
        for _frame_index in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        cap.release()
        writer.release()
        rows.append(
            {
                "video_id": job.video_id,
                "match_id": job.match_id,
                "rally_interval_id": interval_id,
                "clip_path": relative(clip_path),
                "start_frame": start,
                "end_frame": end,
                "status": "ok" if written > 0 else "write_failed",
            }
        )
    return rows


def summarize(
    validation_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    training_summary: dict[str, Any] | None,
    clip_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_hits = sum(safe_int(row["annotated_hit_rows"]) for row in validation_rows)
    covered_hits = sum(safe_int(row["annotated_hits_inside_accepted_intervals"]) for row in validation_rows)
    total_rallies = sum(safe_int(row["annotated_rallies"]) for row in validation_rows)
    overlapped_rallies = sum(safe_int(row["annotated_rallies_overlapping_accepted_intervals"]) for row in validation_rows)
    hit_coverage = covered_hits / total_hits if total_hits else None
    rally_overlap_rate = overlapped_rallies / total_rallies if total_rallies else None
    quality_status = "not_evaluated"
    if total_hits:
        if covered_hits == 0:
            quality_status = "failed_zero_annotation_coverage"
        elif hit_coverage is not None and hit_coverage < 0.5:
            quality_status = "needs_calibration_low_annotation_coverage"
        else:
            quality_status = "passed_annotation_coverage_smoke"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "rally_filtering/filter_rallies.py",
        "sacnn_path": relative(Path(args.sacnn_path).resolve()),
        "upstream_sacnn_path": relative(DEFAULT_UPSTREAM_SACNN.resolve()),
        "training_summary_path": "" if args.training_summary is None else relative(Path(args.training_summary).resolve()),
        "training_best_val_f1": None if training_summary is None else training_summary.get("best_val_f1"),
        "training_epochs": None if training_summary is None else training_summary.get("epochs"),
        "raw_video_root": relative(Path(args.raw_video_root).resolve()),
        "output_root": relative(Path(args.output_root).resolve()),
        "device": str(device),
        "gpu_id": args.gpu_id,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "decoder_requested": args.decoder,
        "decoder_counts": {
            "opencv": sum(1 for row in validation_rows if row.get("decoder") == "opencv"),
            "ffmpeg": sum(1 for row in validation_rows if row.get("decoder") == "ffmpeg"),
        },
        "queue_length": args.queue_length,
        "sample_period_sec": args.sample_period_sec,
        "batch_size": args.batch_size,
        "min_rally_sec": args.min_rally_sec,
        "max_rally_sec": args.max_rally_sec,
        "video_count": len(validation_rows),
        "total_sampled_frames": sum(safe_int(row["sampled_frames"]) for row in validation_rows),
        "total_predicted_intervals": sum(safe_int(row["predicted_intervals"]) for row in validation_rows),
        "total_accepted_intervals": sum(safe_int(row["accepted_intervals"]) for row in validation_rows),
        "total_rejected_intervals": sum(safe_int(row["rejected_intervals"]) for row in validation_rows),
        "exported_clip_count": sum(1 for row in clip_rows if row.get("status") == "ok"),
        "total_annotated_hit_rows": total_hits,
        "total_annotated_hits_inside_accepted_intervals": covered_hits,
        "annotated_hit_coverage": hit_coverage,
        "total_annotated_rallies": total_rallies,
        "total_annotated_rallies_overlapping_accepted_intervals": overlapped_rallies,
        "annotated_rally_overlap_rate": rally_overlap_rate,
        "machine_validation_status": "passed" if validation_rows else "failed",
        "rally_filtering_quality_status": quality_status,
        "human_visual_review_status": "pending",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 05 SA-CNN rally interval extraction for ShuttleSet raw videos.")
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEOS)
    parser.add_argument("--match-csv", type=Path, default=DEFAULT_MATCH_CSV)
    parser.add_argument("--ground-truth-table", type=Path, default=DEFAULT_GT_STROKES)
    parser.add_argument("--rally-summary", type=Path, default=DEFAULT_RALLY_SUMMARY)
    parser.add_argument("--sacnn-path", type=Path, default=DEFAULT_SACNN)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", type=int, help="1-based ShuttleSet match id. Can be passed more than once.")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Default is one video unless --video-id is passed; set 0 for all selected videos.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index to use when CUDA is available.")
    parser.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto")
    parser.add_argument("--sample-period-sec", type=float, default=1.0)
    parser.add_argument("--queue-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-rally-sec", type=float, default=2.0)
    parser.add_argument("--max-rally-sec", type=float, default=180.0)
    parser.add_argument("--visual-samples", type=int, default=3)
    parser.add_argument("--export-clips", action="store_true", help="Export accepted rally intervals as MP4 clips.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device(cpu=args.cpu, gpu_id=args.gpu_id)
    sacnn_path = Path(args.sacnn_path).resolve()
    if not sacnn_path.exists():
        raise FileNotFoundError(f"SA-CNN checkpoint not found: {sacnn_path}")
    model = load_model(sacnn_path, device)
    training_summary = load_training_summary(None if args.training_summary is None else Path(args.training_summary).resolve())
    requested_ids = set(args.video_id) if args.video_id else None
    if args.max_videos == 0:
        max_videos = None
    elif args.max_videos is None:
        max_videos = None if requested_ids else 1
    else:
        max_videos = args.max_videos
    jobs = discover_jobs(Path(args.raw_video_root).resolve(), Path(args.match_csv).resolve(), requested_ids, max_videos)
    if not jobs:
        raise RuntimeError("No Phase 05 videos selected.")

    gt_rows = read_csv(Path(args.ground_truth_table).resolve())
    rally_rows = read_csv(Path(args.rally_summary).resolve())
    validation_rows: list[dict[str, Any]] = []
    all_visual_rows: list[dict[str, Any]] = []
    all_clip_rows: list[dict[str, Any]] = []

    for job in jobs:
        metadata, sample_rows, runtime_sec, decoder_used = predict_video_states(
            model=model,
            device=device,
            job=job,
            sample_period_sec=args.sample_period_sec,
            batch_size=args.batch_size,
            queue_length=args.queue_length,
            max_samples=args.max_samples,
            decoder=args.decoder,
        )
        if not sample_rows:
            raise RuntimeError(
                f"Decoded zero sampled frames from {job.video_path}. "
                "This usually means OpenCV cannot decode the local video codec."
            )
        sample_stride = max(1, int(round(metadata.fps * args.sample_period_sec)))
        intervals = build_intervals(job, sample_rows, metadata.fps, args.min_rally_sec, args.max_rally_sec)
        video_root = output_root / job.video_id
        write_csv(video_root / "sampled_shot_angle_states.csv", sample_rows, SAMPLE_COLUMNS)
        write_csv(video_root / "rally_intervals.csv", intervals, INTERVAL_COLUMNS)
        validation_rows.append(
            validation_row(
                job=job,
                metadata=metadata,
                sample_stride=sample_stride,
                sample_period_sec=args.sample_period_sec,
                sample_rows=sample_rows,
                intervals=intervals,
                gt_rows=gt_rows,
                rally_rows=rally_rows,
                runtime_sec=runtime_sec,
                decoder_used=decoder_used,
            )
        )
        all_visual_rows.extend(export_visual_review(output_root, job, intervals, args.visual_samples))
        if args.export_clips:
            all_clip_rows.extend(export_rally_clips(output_root, job, intervals, metadata))

    write_csv(output_root / "phase05_video_validation.csv", validation_rows, VALIDATION_COLUMNS)
    write_csv(
        output_root / "visual_review_manifest.csv",
        all_visual_rows,
        [
            "video_id",
            "match_id",
            "rally_interval_id",
            "interval_status",
            "frame_role",
            "frame_index",
            "image_path",
            "human_review_status",
        ],
    )
    write_csv(
        output_root / "clip_manifest.csv",
        all_clip_rows,
        ["video_id", "match_id", "rally_interval_id", "clip_path", "start_frame", "end_frame", "status"],
    )
    summary = summarize(validation_rows, args, device, training_summary, all_clip_rows)
    (output_root / "phase05_rally_filtering_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
