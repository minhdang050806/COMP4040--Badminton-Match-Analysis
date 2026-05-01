"""Stage 4 — orchestration of feature extraction → BST → ViT ensemble."""
from __future__ import annotations

import os
from typing import List

import numpy as np

from common.contracts import HitEvent, RallySegment, StrokeRecord, STROKE_CSV_COLUMNS
from common.io import save_csv, load_json, ensure_dir, save_json

from stage_4_stroke_classification.feature_extraction import FeatureExtractor
from stage_4_stroke_classification.bst import StrokeClassifier
from stage_4_stroke_classification.vit_ensemble import ViTPipeline


class StrokePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fx = FeatureExtractor(seq_len=cfg.bst.seq_len)
        self.classifier = StrokeClassifier(
            weights_path=cfg.models.bst,
            n_classes=cfg.bst.n_classes,
            seq_len=cfg.bst.seq_len,
            d_model=cfg.bst.d_model,
        )
        self.vit = ViTPipeline(cfg)

    # ------------------------------------------------------------------
    def _load_features_for_rally(self, rally: RallySegment):
        if not rally.joints_path or not os.path.exists(rally.joints_path):
            return None
        rally_info = load_json(rally.joints_path)
        joints = (rally_info.get("player_joints")
                  if isinstance(rally_info, dict) else rally_info)
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim != 4:
            return None

        T = joints.shape[0]
        shuttle = np.zeros((T, 2), dtype=np.float32)
        if rally.shuttle_csv_path and os.path.exists(rally.shuttle_csv_path):
            try:
                arr = np.genfromtxt(rally.shuttle_csv_path, delimiter=",",
                                    skip_header=1)
                for row in arr:
                    f = int(row[0]) - rally.start_frame
                    if 0 <= f < T and len(row) >= 4:
                        shuttle[f] = [row[2], row[3]]
            except Exception:
                pass

        court_xy = joints[:, :, 15:17, :].mean(axis=2)  # ankles → court pos
        # Fold any homography here — fall back to frame-normalised coords.
        court_xy = court_xy / np.array([1280.0, 720.0], dtype=np.float32)
        return self.fx.extract(joints, shuttle, court_xy)

    # ------------------------------------------------------------------
    def run(self, rallies: List[RallySegment],
            hit_events: List[HitEvent]) -> List[StrokeRecord]:
        # Index events by rally
        evt_by_rally: dict = {}
        for e in hit_events:
            evt_by_rally.setdefault(e.rally_id, []).append(e)

        records: List[StrokeRecord] = []
        for rally in rallies:
            features = self._load_features_for_rally(rally)
            for evt in evt_by_rally.get(rally.rally_id, []):
                if features is None:
                    cls, probs, name = -1, None, ""
                else:
                    cls, probs, name = self.classifier.predict(features)
                attrs = self.vit.predict(rally, evt)
                rec = StrokeRecord(
                    video_name=os.path.splitext(
                        os.path.basename(rally.video_path))[0],
                    rally_id=rally.rally_id,
                    shot_seq=evt.shot_seq,
                    hit_frame=evt.hit_frame,
                    stroke_class=cls,
                    stroke_class_name=name,
                    stroke_probs=(probs.tolist() if probs is not None else None),
                    **attrs,
                )
                records.append(rec)
        return records

    # ------------------------------------------------------------------
    def export_csv(self, records: List[StrokeRecord], out_csv: str) -> str:
        rows = []
        for r in records:
            d = r.to_dict()
            hx = (r.hitter_xy_court or [None, None])
            dx = (r.defender_xy_court or [None, None])
            lx = (r.landing_xy_court or [None, None])
            d.update({
                "hitter_x": hx[0], "hitter_y": hx[1],
                "defender_x": dx[0], "defender_y": dx[1],
                "landing_x": lx[0], "landing_y": lx[1],
            })
            rows.append(d)
        save_csv(out_csv, rows, STROKE_CSV_COLUMNS)
        return out_csv
