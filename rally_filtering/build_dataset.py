from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_VIDEOS = ROOT / "project" / "dataset" / "ShuttleSet_raw_videos"
DEFAULT_SET_ROOT = ROOT / "project" / "dataset" / "ShuttleSet" / "set"
DEFAULT_MATCH_CSV = DEFAULT_SET_ROOT / "match.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "dataset" / "sa_cnn_data"


@dataclass(frozen=True)
class MatchJob:
    match_id: int
    video_name: str
    video_path: Path
    annotation_dir: Path
    split: str


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def split_for_match(match_id: int, val_ids: set[int], test_ids: set[int]) -> str:
    if match_id in test_ids:
        return "test"
    if match_id in val_ids:
        return "val"
    return "train"


def parse_id_list(value: str) -> set[int]:
    ids: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(chunk))
    return ids


def discover_jobs(
    raw_video_root: Path,
    match_csv: Path,
    set_root: Path,
    requested_ids: set[int] | None,
    max_videos: int | None,
    val_ids: set[int],
    test_ids: set[int],
) -> list[MatchJob]:
    video_by_id: dict[int, Path] = {}
    for path in sorted(raw_video_root.glob("*.mp4")):
        prefix = path.name.split(" - ", 1)[0]
        if prefix.isdigit():
            video_by_id[int(prefix)] = path

    jobs: list[MatchJob] = []
    for row in read_csv(match_csv):
        match_id = safe_int(row.get("id"))
        if requested_ids is not None and match_id not in requested_ids:
            continue
        video_path = video_by_id.get(match_id)
        annotation_dir = set_root / row["video"]
        if video_path is None or not annotation_dir.exists():
            continue
        jobs.append(
            MatchJob(
                match_id=match_id,
                video_name=row["video"],
                video_path=video_path,
                annotation_dir=annotation_dir,
                split=split_for_match(match_id, val_ids, test_ids),
            )
        )
        if max_videos is not None and len(jobs) >= max_videos:
            break
    return jobs


def get_video_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    return VideoMetadata(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
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
        frame_count=frame_count,
        width=safe_int(stream.get("width")),
        height=safe_int(stream.get("height")),
    )
    if metadata.fps <= 0 or metadata.frame_count <= 0 or metadata.width <= 0 or metadata.height <= 0:
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


def select_decoder(video_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    codec_name = ffprobe_codec_name(video_path)
    if codec_name == "av1":
        return "ffmpeg"
    return "opencv" if opencv_can_decode_first_frame(video_path) else "ffmpeg"


def metadata_for_video(video_path: Path, decoder: str) -> VideoMetadata:
    if decoder == "ffmpeg":
        return ffprobe_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    metadata = get_video_metadata(cap)
    cap.release()
    if metadata.fps <= 0 or metadata.frame_count <= 0:
        raise RuntimeError(f"Video metadata is invalid for {video_path}: {metadata}")
    return metadata


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def annotation_intervals(job: MatchJob, fps: float, frame_count: int, positive_pad_sec: float) -> list[tuple[int, int]]:
    pad_frames = int(round(fps * positive_pad_sec))
    intervals: list[tuple[int, int]] = []
    for set_csv in sorted(job.annotation_dir.glob("set*.csv")):
        by_rally: dict[int, list[int]] = {}
        for row in read_csv(set_csv):
            rally_id = safe_int(row.get("rally"))
            frame = safe_int(row.get("frame_num"), default=-1)
            if rally_id <= 0 or frame < 0:
                continue
            by_rally.setdefault(rally_id, []).append(frame)
        for frames in by_rally.values():
            start = max(0, min(frames) - pad_frames)
            end = min(frame_count - 1, max(frames) + pad_frames)
            if end >= start:
                intervals.append((start, end))
    return merge_intervals(intervals)


def interval_length(interval: tuple[int, int]) -> int:
    return interval[1] - interval[0] + 1


def sample_positive_frames(
    intervals: list[tuple[int, int]],
    count: int,
    rng: random.Random,
    min_stride_frames: int,
) -> list[int]:
    candidates: list[int] = []
    for start, end in intervals:
        step = max(1, min_stride_frames)
        candidates.extend(range(start, end + 1, step))
    if len(candidates) <= count:
        return sorted(set(candidates))
    return sorted(rng.sample(candidates, count))


def complement_intervals(intervals: list[tuple[int, int]], frame_count: int, guard_frames: int) -> list[tuple[int, int]]:
    guarded = merge_intervals(
        [(max(0, start - guard_frames), min(frame_count - 1, end + guard_frames)) for start, end in intervals]
    )
    complement: list[tuple[int, int]] = []
    cursor = 0
    for start, end in guarded:
        if cursor < start:
            complement.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor < frame_count:
        complement.append((cursor, frame_count - 1))
    return complement


def sample_frames_from_intervals(
    intervals: list[tuple[int, int]],
    count: int,
    rng: random.Random,
    min_stride_frames: int,
) -> list[int]:
    candidates: list[int] = []
    for start, end in intervals:
        if interval_length((start, end)) < min_stride_frames:
            continue
        candidates.extend(range(start, end + 1, max(1, min_stride_frames)))
    if len(candidates) <= count:
        return sorted(set(candidates))
    return sorted(rng.sample(candidates, count))


def read_frame(cap: cv2.VideoCapture, frame_index: int, seek_back_frames: int = 30) -> Any | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok:
        return frame

    retry_start = max(0, frame_index - seek_back_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, retry_start)
    current = retry_start
    while current <= frame_index:
        ok, frame = cap.read()
        if not ok:
            return None
        if current == frame_index:
            return frame
        current += 1
    return None


def save_samples(
    job: MatchJob,
    output_root: Path,
    label: int,
    frame_indices: list[int],
    resize_width: int,
    resize_height: int,
    decoder: str,
    metadata: VideoMetadata,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_dir = output_root / job.split / str(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    if decoder == "opencv":
        cap = cv2.VideoCapture(str(job.video_path))
        if not cap.isOpened():
            return rows
        for frame_index in frame_indices:
            frame = read_frame(cap, frame_index)
            if frame is not None:
                row = save_sample_frame(job, output_root, label, frame_index, frame, resize_width, resize_height)
                if row is not None:
                    rows.append(row)
        cap.release()
        return rows

    if decoder != "ffmpeg":
        raise RuntimeError(f"Unknown decoder: {decoder}")

    rows.extend(
        save_samples_ffmpeg(
            job=job,
            output_root=output_root,
            label=label,
            frame_indices=frame_indices,
            resize_width=resize_width,
            resize_height=resize_height,
            metadata=metadata,
        )
    )
    return rows


def save_sample_frame(
    job: MatchJob,
    output_root: Path,
    label: int,
    frame_index: int,
    frame: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> dict[str, Any] | None:
    if resize_width > 0 and resize_height > 0:
        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    out_dir = output_root / job.split / str(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"m{job.match_id:02d}_f{frame_index:07d}_y{label}.jpg"
    ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return {
        "image_path": relative(out_path),
        "split": job.split,
        "label": label,
        "match_id": job.match_id,
        "video_name": job.video_name,
        "video_file": relative(job.video_path),
        "frame_index": frame_index,
    }


def save_samples_ffmpeg(
    job: MatchJob,
    output_root: Path,
    label: int,
    frame_indices: list[int],
    resize_width: int,
    resize_height: int,
    metadata: VideoMetadata,
) -> list[dict[str, Any]]:
    targets = sorted(set(frame_indices))
    if not targets:
        return []
    target_set = set(targets)
    max_target = max(targets)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(job.video_path),
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

    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while frame_index <= max_target:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError(f"ffmpeg returned an incomplete frame for {job.video_path}")
            if frame_index in target_set:
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((metadata.height, metadata.width, 3))
                row = save_sample_frame(job, output_root, label, frame_index, frame, resize_width, resize_height)
                if row is not None:
                    rows.append(row)
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
                raise RuntimeError(f"ffmpeg failed for {job.video_path}: {stderr}")
    return rows


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Output root already contains files: {output_root}. "
                "Use --overwrite for a clean rebuild or --summarize-existing to reconstruct manifests."
            )
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    val_ids = parse_id_list(args.val_ids)
    test_ids = parse_id_list(args.test_ids)
    requested_ids = parse_id_list(args.video_ids) if args.video_ids else None
    max_videos = None if args.max_videos == 0 else args.max_videos
    jobs = discover_jobs(
        raw_video_root=Path(args.raw_video_root).resolve(),
        match_csv=Path(args.match_csv).resolve(),
        set_root=Path(args.set_root).resolve(),
        requested_ids=requested_ids,
        max_videos=max_videos,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    if not jobs:
        raise RuntimeError("No SA-CNN dataset jobs selected.")

    manifest_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for job in jobs:
        try:
            decoder_used = select_decoder(job.video_path, args.decoder)
            metadata = metadata_for_video(job.video_path, decoder_used)
        except Exception as exc:
            video_rows.append(
                {
                    "match_id": job.match_id,
                    "split": job.split,
                    "status": "metadata_or_decoder_failed",
                    "decoder": args.decoder,
                    "video_file": relative(job.video_path),
                    "error": str(exc),
                }
            )
            continue

        positives = annotation_intervals(job, metadata.fps, metadata.frame_count, args.positive_pad_sec)
        negatives = complement_intervals(positives, metadata.frame_count, int(round(metadata.fps * args.negative_guard_sec)))
        min_stride = max(1, int(round(metadata.fps * args.min_sample_gap_sec)))
        pos_frames = sample_positive_frames(positives, args.samples_per_class_per_video, rng, min_stride)
        neg_frames = sample_frames_from_intervals(negatives, args.samples_per_class_per_video, rng, min_stride)

        pos_rows = save_samples(
            job,
            output_root,
            1,
            pos_frames,
            args.resize_width,
            args.resize_height,
            decoder_used,
            metadata,
        )
        neg_rows = save_samples(
            job,
            output_root,
            0,
            neg_frames,
            args.resize_width,
            args.resize_height,
            decoder_used,
            metadata,
        )

        manifest_rows.extend(pos_rows)
        manifest_rows.extend(neg_rows)
        video_rows.append(
            {
                "match_id": job.match_id,
                "split": job.split,
                "status": video_status(pos_rows=len(pos_rows), neg_rows=len(neg_rows)),
                "decoder": decoder_used,
                "video_file": relative(job.video_path),
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "positive_intervals": len(positives),
                "negative_intervals": len(negatives),
                "positive_images": len(pos_rows),
                "negative_images": len(neg_rows),
            }
        )
        print(
            f"match {job.match_id:02d} {job.split}: "
            f"{len(pos_rows)} positive, {len(neg_rows)} negative images"
        )

    manifest_fields = ["image_path", "split", "label", "match_id", "video_name", "video_file", "frame_index"]
    video_fields = [
        "match_id",
        "split",
        "status",
        "decoder",
        "video_file",
        "error",
        "fps",
        "frame_count",
        "width",
        "height",
        "positive_intervals",
        "negative_intervals",
        "positive_images",
        "negative_images",
    ]
    write_csv(output_root / "manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_root / "video_summary.csv", video_rows, video_fields)

    split_counts: dict[str, dict[str, int]] = {
        split: {"0": 0, "1": 0} for split in ("train", "val", "test")
    }
    for row in manifest_rows:
        split_counts[str(row["split"])][str(row["label"])] += 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "rally_filtering/build_dataset.py",
        "output_root": relative(output_root),
        "raw_video_root": relative(Path(args.raw_video_root).resolve()),
        "set_root": relative(Path(args.set_root).resolve()),
        "match_csv": relative(Path(args.match_csv).resolve()),
        "decoder_requested": args.decoder,
        "decoder_counts": {
            "opencv": sum(1 for row in video_rows if row.get("decoder") == "opencv"),
            "ffmpeg": sum(1 for row in video_rows if row.get("decoder") == "ffmpeg"),
        },
        "seed": args.seed,
        "samples_per_class_per_video": args.samples_per_class_per_video,
        "positive_pad_sec": args.positive_pad_sec,
        "negative_guard_sec": args.negative_guard_sec,
        "min_sample_gap_sec": args.min_sample_gap_sec,
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
        "video_jobs": len(jobs),
        "videos_ok": sum(1 for row in video_rows if row.get("status") == "ok"),
        "videos_incomplete": sum(1 for row in video_rows if row.get("status") != "ok"),
        "split_counts": split_counts,
        "total_images": len(manifest_rows),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def video_status(pos_rows: int, neg_rows: int) -> str:
    if pos_rows > 0 and neg_rows > 0:
        return "ok"
    if pos_rows > 0 and neg_rows == 0:
        return "missing_negative_images"
    if pos_rows == 0 and neg_rows > 0:
        return "missing_positive_images"
    return "decode_or_write_failed"


def summarize_existing(
    output_root: Path,
    raw_video_root: Path,
    match_csv: Path,
    val_ids: set[int],
    test_ids: set[int],
    samples_per_class_per_video: int,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    raw_video_root = raw_video_root.resolve()
    match_rows = {safe_int(row.get("id")): row for row in read_csv(match_csv) if safe_int(row.get("id")) > 0}
    video_by_id: dict[int, Path] = {}
    for path in sorted(raw_video_root.glob("*.mp4")):
        prefix = path.name.split(" - ", 1)[0]
        if prefix.isdigit():
            video_by_id[int(prefix)] = path

    manifest_rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {
        split: {"0": 0, "1": 0} for split in ("train", "val", "test")
    }
    counts_by_match: dict[int, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        for label in ("0", "1"):
            for image_path in sorted((output_root / split / label).glob("*.jpg")):
                stem = image_path.stem
                match_id = 0
                frame_index = 0
                parts = stem.split("_")
                if len(parts) >= 2 and parts[0].startswith("m") and parts[1].startswith("f"):
                    match_id = safe_int(parts[0][1:])
                    frame_index = safe_int(parts[1][1:])
                match_row = match_rows.get(match_id, {})
                video_path = video_by_id.get(match_id)
                manifest_rows.append(
                    {
                        "image_path": relative(image_path),
                        "split": split,
                        "label": int(label),
                        "match_id": match_id,
                        "video_name": match_row.get("video", ""),
                        "video_file": "" if video_path is None else relative(video_path),
                        "frame_index": frame_index,
                    }
                )
                split_counts[split][label] += 1
                counts = counts_by_match.setdefault(match_id, {"0": 0, "1": 0})
                counts[label] += 1

    manifest_fields = ["image_path", "split", "label", "match_id", "video_name", "video_file", "frame_index"]
    write_csv(output_root / "manifest.csv", manifest_rows, manifest_fields)
    video_rows: list[dict[str, Any]] = []
    for match_id in sorted(match_rows):
        split = split_for_match(match_id, val_ids, test_ids)
        counts = counts_by_match.get(match_id, {"0": 0, "1": 0})
        video_path = video_by_id.get(match_id)
        if counts["0"] >= samples_per_class_per_video and counts["1"] >= samples_per_class_per_video:
            status = "complete"
        elif counts["0"] or counts["1"]:
            status = "partial"
        elif video_path is None:
            status = "missing_video"
        else:
            status = "no_images"
        video_rows.append(
            {
                "match_id": match_id,
                "split": split,
                "status": status,
                "decoder": "",
                "video_file": "" if video_path is None else relative(video_path),
                "error": "",
                "fps": "",
                "frame_count": "",
                "width": "",
                "height": "",
                "positive_intervals": "",
                "negative_intervals": "",
                "positive_images": counts["1"],
                "negative_images": counts["0"],
            }
        )
    video_fields = [
        "match_id",
        "split",
        "status",
        "decoder",
        "video_file",
        "error",
        "fps",
        "frame_count",
        "width",
        "height",
        "positive_intervals",
        "negative_intervals",
        "positive_images",
        "negative_images",
    ]
    write_csv(output_root / "video_summary.csv", video_rows, video_fields)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "rally_filtering/build_dataset.py --summarize-existing",
        "output_root": relative(output_root),
        "split_counts": split_counts,
        "total_images": len(manifest_rows),
        "complete_videos": sum(1 for row in video_rows if row["status"] == "complete"),
        "partial_videos": sum(1 for row in video_rows if row["status"] == "partial"),
        "no_image_videos": sum(1 for row in video_rows if row["status"] == "no_images"),
        "note": "Summary reconstructed from existing ImageFolder JPEG files.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ShuttleSet-derived ImageFolder dataset for SA-CNN training.")
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEOS)
    parser.add_argument("--set-root", type=Path, default=DEFAULT_SET_ROOT)
    parser.add_argument("--match-csv", type=Path, default=DEFAULT_MATCH_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-ids", default="", help="Comma/range list, e.g. '2,9,17-20'. Default: all.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all selected videos.")
    parser.add_argument("--val-ids", default="35-39")
    parser.add_argument("--test-ids", default="40-44")
    parser.add_argument("--samples-per-class-per-video", type=int, default=200)
    parser.add_argument("--positive-pad-sec", type=float, default=2.0)
    parser.add_argument("--negative-guard-sec", type=float, default=4.0)
    parser.add_argument("--min-sample-gap-sec", type=float, default=1.0)
    parser.add_argument("--resize-width", type=int, default=384)
    parser.add_argument("--resize-height", type=int, default=216)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--decoder", choices=["auto", "opencv", "ffmpeg"], default="auto")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Delete output-root before building a fresh ImageFolder dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = (
        summarize_existing(
            Path(args.output_root),
            Path(args.raw_video_root),
            Path(args.match_csv),
            parse_id_list(args.val_ids),
            parse_id_list(args.test_ids),
            args.samples_per_class_per_video,
        )
        if args.summarize_existing
        else build_dataset(args)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
