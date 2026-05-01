"""Stage 3 — Hit-Frame Detection.

Strategy A (primary):
    Run OptimusPrime on the joint sequence emitted by Stage 2; detect
    direction-token transitions and turn them into HitEvent records.

Strategy B (cross-check):
    Trajectory-based event detection on the shuttle CSV (legacy
    event_detection_custom from A-New-Perspective). Used as a fallback /
    cross-check when joints are unavailable.

The orchestrator picks A when ``rally.joints_path`` is populated, else B.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

from common.contracts import HitEvent, RallySegment
from common.io import load_json, save_json, ensure_dir

from stage_3_hit_frame_detection.optimusprime.transformer import OptimusPrimeContainer


class HitDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opt = OptimusPrimeContainer({
            "opt_path":    cfg.models.optimusprime,
            "scaler_path": cfg.models.scaler,
        })

    # ------------------------------------------------------------------
    def detect(self, rally: RallySegment) -> List[HitEvent]:
        if rally.joints_path and os.path.exists(rally.joints_path):
            return self._detect_from_joints(rally)
        if rally.shuttle_csv_path and os.path.exists(rally.shuttle_csv_path):
            return self._detect_from_trajectory(rally)
        return []

    # ------------------------------------------------------------------
    def _detect_from_joints(self, rally: RallySegment) -> List[HitEvent]:
        rally_info = load_json(rally.joints_path)
        joints = np.array(rally_info["player_joints"]) \
            if isinstance(rally_info, dict) and "player_joints" in rally_info \
            else np.array(rally_info)
        directions = self.opt.predict(joints)              # (T,)
        events: List[HitEvent] = []
        prev = directions[0] if len(directions) else 0
        seq = 0
        for i in range(1, len(directions)):
            cur = directions[i]
            transition = (prev, cur)
            if transition in {(0, 1), (0, 2), (1, 2), (2, 1)}:
                seq += 1
                events.append(HitEvent(
                    rally_id=rally.rally_id,
                    shot_seq=seq,
                    hit_frame=int(rally.start_frame + i),
                    direction_before=int(prev),
                    direction_after=int(cur),
                    joints_at_hit=np.array(joints[i]).tolist(),
                ))
            prev = cur
        return events

    # ------------------------------------------------------------------
    def _detect_from_trajectory(self, rally: RallySegment) -> List[HitEvent]:
        """Use the legacy peak-detection routine on shuttle CSV."""
        try:
            from stage_3_hit_frame_detection.event_detection.event_detection_custom import (
                detect_events,
            )
        except Exception:
            return []
        rows = []
        with open(rally.shuttle_csv_path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    rows.append((int(parts[0]),
                                 float(parts[2]) if parts[2] else 0.0,
                                 float(parts[3]) if parts[3] else 0.0))
        if not rows:
            return []
        ys = np.array([r[2] for r in rows])
        # Simple peak-detection: angle change > 5° on Y. The reference impl
        # in event_detection_custom is more sophisticated; keep a fallback.
        events: List[HitEvent] = []
        for i in range(2, len(ys) - 2):
            if (ys[i] - ys[i - 2]) * (ys[i + 2] - ys[i]) < 0:
                events.append(HitEvent(
                    rally_id=rally.rally_id,
                    shot_seq=len(events) + 1,
                    hit_frame=int(rows[i][0]),
                    shuttle_at_hit=[rows[i][1], rows[i][2]],
                ))
        return events

    # ------------------------------------------------------------------
    def detect_all(self, rallies: List[RallySegment]) -> List[HitEvent]:
        out: List[HitEvent] = []
        for r in rallies:
            out.extend(self.detect(r))
        out_dir = ensure_dir(self.cfg.storage.hit_events_dir)
        save_json(os.path.join(out_dir, "hit_events.json"),
                  [e.to_dict() for e in out])
        return out
