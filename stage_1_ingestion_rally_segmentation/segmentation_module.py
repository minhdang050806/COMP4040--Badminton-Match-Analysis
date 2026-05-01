"""Stage 1 — Ingestion & Rally Segmentation.

Reuses SACNN + ShotAngleQueue from the original Automated-Hit-frame-Detection
project. The original VideoResolver inlined I/O, court detection and the
rally processor; here we expose a thin SegmentationModule that emits
RallySegment records (Contract A) instead of writing private JSONs.
"""
from __future__ import annotations

import os
import sys
from typing import List

import cv2
from PIL import Image

from common.contracts import RallySegment
from common.io import ensure_dir, save_json

# Local re-export of SACNN — original code path
from .models.sacnn import SACNNContainer  # noqa: E402


class ShotAngleQueue:
    """Lifted verbatim (refactored to a flat module) from VideoResolver."""

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.queue: list = []
        self.last_sa = 0

    def push(self, frame_info):
        sa_condition = None
        if len(self.queue) < self.max_len:
            self.queue.append(frame_info)
            return None, None
        first_info = self.queue.pop(0)
        sa, sa_condition = self._check_sa_condition(first_info[0])
        self.last_sa = sa
        first_info[0] = sa
        self.queue.append(frame_info)
        return first_info, sa_condition

    def get(self, idx):
        return self.queue[idx]

    def _check_sa_condition(self, sa):
        s = 0 + sa
        if self.last_sa == 1 and sa == 0:
            for info in self.queue:
                s += info[0]
            return (0, 3) if s <= self.max_len / 2 else (1, 2)
        if self.last_sa == 0 and sa == 1:
            for info in self.queue:
                s += info[0]
            return (1, 1) if s >= self.max_len / 2 else (0, 0)
        if self.last_sa == 1 and sa == 1:
            return 1, 2
        if self.last_sa == 0 and sa == 0:
            return 0, 0
        return self.last_sa, 0


class SegmentationModule:
    """Detect rally [start, end] frames, export per-rally MP4 clips, emit
    Contract A records to ``rallies_dir``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sacnn = SACNNContainer({"sacnn_path": cfg.models.sacnn})
        self.queue_len = cfg.stage_1.sa_queue_length

    # ------------------------------------------------------------------
    def process(self, video_path: str) -> List[RallySegment]:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        clip_dir = ensure_dir(os.path.join(
            self.cfg.storage.rallies_dir, video_name, "clips"))
        rallies_meta_path = os.path.join(
            self.cfg.storage.rallies_dir, video_name, "rallies.json")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(3))
        h = int(cap.get(4))
        total = int(cap.get(7))
        time_rate = 0.1
        frame_rate = max(1, round(int(fps) * time_rate))

        sa_queue = ShotAngleQueue(self.queue_len)
        rally_id = 0
        rally_start = None
        rally_buffer: list = []
        results: List[RallySegment] = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                sa = self.sacnn.predict(pil)
                info, cond = sa_queue.push([sa, frame])
                if info is not None:
                    cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if cond == 1:                                # 0->1
                        rally_start = int(cur_frame)
                        rally_buffer = [frame]
                    elif cond == 2 and rally_start is not None:  # 1->1
                        rally_buffer.append(frame)
                    elif cond == 3 and rally_start is not None:  # 1->0
                        rally_id += 1
                        end_frame = int(cur_frame)
                        clip_path = os.path.join(
                            clip_dir, f"rally_{rally_id}.mp4")
                        out = cv2.VideoWriter(
                            clip_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(1, int(fps / frame_rate)),
                            (w, h),
                        )
                        for f_ in rally_buffer:
                            out.write(f_)
                        out.release()

                        results.append(RallySegment(
                            video_path=video_path,
                            rally_id=rally_id,
                            start_frame=rally_start,
                            end_frame=end_frame,
                            clip_path=clip_path,
                            fps=fps,
                        ))
                        rally_start = None
                        rally_buffer = []
            frame_count += 1
        cap.release()

        save_json(rallies_meta_path, [r.to_dict() for r in results])
        return results

    # ------------------------------------------------------------------
    def process_dir(self, source_dir: str) -> List[RallySegment]:
        all_segments: List[RallySegment] = []
        for fn in sorted(os.listdir(source_dir)):
            if not fn.lower().endswith((".mp4", ".mov", ".mkv")):
                continue
            all_segments.extend(self.process(os.path.join(source_dir, fn)))
        return all_segments


# CLI entry point ------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))
    from common.config import load_config
    cfg = load_config()
    mod = SegmentationModule(cfg)
    seg = mod.process_dir(cfg.video.source_dir)
    print(f"Stage 1: produced {len(seg)} rally segments")


if __name__ == "__main__":
    main()
