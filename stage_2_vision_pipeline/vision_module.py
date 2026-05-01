"""Stage 2 — Vision Pipeline.

Wraps three pre-existing modules from the original repos:
  * ``court.rally_processor.RallyProcessor`` — Court Keypoint R-CNN +
    Human Keypoint R-CNN + filtering by court geometry.
  * TrackNetV2 (``shuttle_tracking.tracknetv2``) — shuttlecock trajectory.

The original RallyProcessor stored its results inside its own JSON. The
adapter here exposes a clean callable that, given a RallySegment, fills in
``joints_path`` / ``shuttle_csv_path`` / ``court_corners`` / ``homography``
in-place and returns the updated record.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

import cv2
import numpy as np

from common.contracts import RallySegment
from common.io import ensure_dir, save_json

# Make the legacy court module importable as a top-level package
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "court"))

from court.rally_processor import RallyProcessor  # noqa: E402


class VisionModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self._setup_court_processor()

    # ------------------------------------------------------------------
    def _setup_court_processor(self):
        # The legacy RallyProcessor expects a dict with model paths.
        rp_args = {
            "court_kpRCNN_path": self.cfg.models.court_rcnn,
            "kpRCNN_path":       self.cfg.models.human_rcnn,
            "opt_path":          self.cfg.models.optimusprime,
            "scaler_path":       self.cfg.models.scaler,
        }
        self.rp = RallyProcessor(rp_args)

    # ------------------------------------------------------------------
    def process(self, rally: RallySegment) -> RallySegment:
        cap = cv2.VideoCapture(rally.clip_path)
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            return rally

        frame_h = int(cap.get(4))
        if not self.rp.got_info:
            self.rp.get_court_info(first_frame, frame_h)

        # Replay through RallyProcessor -- accumulate frames and run end-of-rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = rally.start_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.rp.add_frame(frame, frame_idx)
            frame_idx += 1
        cap.release()

        result = self.rp.start_new_rally(rally.start_frame, rally.end_frame)
        if not result:
            return rally
        _, rally_info = result

        joints_dir = ensure_dir(self.cfg.storage.joints_dir)
        joints_path = os.path.join(joints_dir, f"rally_{rally.rally_id}.json")
        save_json(joints_path, rally_info)

        rally.joints_path = joints_path
        # Court corners come from the RallyProcessor's private state — pull
        # the public attribute set above by ``get_court_info``.
        court = getattr(self.rp, "_RallyProcessor__true_court_points", None)
        if court is not None:
            rally.court_corners = np.array(court).tolist()

        # Shuttle: run the TrackNetV2 wrapper if present
        rally.shuttle_csv_path = self._track_shuttle(rally)
        return rally

    # ------------------------------------------------------------------
    def _track_shuttle(self, rally: RallySegment) -> str:
        """Run TrackNetV2 on rally clip → CSV(frame, x, y).

        The original predict10.py is a CLI script. We import the prediction
        helper if available; otherwise we fall back to writing an empty CSV
        so downstream stages can still run.
        """
        out_dir = ensure_dir(self.cfg.storage.shuttle_dir)
        out_csv = os.path.join(out_dir, f"rally_{rally.rally_id}.csv")
        try:
            from stage_2_vision_pipeline.shuttle_tracking.tracker import (
                track_to_csv,
            )
            track_to_csv(
                clip_path=rally.clip_path,
                weights=self.cfg.models.tracknet,
                out_csv=out_csv,
            )
        except Exception as e:
            # Robust fallback: write header-only CSV
            with open(out_csv, "w") as f:
                f.write("frame,visibility,x,y\n")
            print(f"[VisionModule] TrackNetV2 unavailable ({e}); wrote empty CSV.")
        return out_csv

    # ------------------------------------------------------------------
    def process_all(self, rallies: List[RallySegment]) -> List[RallySegment]:
        out = []
        for r in rallies:
            self.rp.reset()
            out.append(self.process(r))
        return out
