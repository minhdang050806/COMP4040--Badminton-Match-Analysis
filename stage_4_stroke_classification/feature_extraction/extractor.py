"""Feature extraction adapter — Section 3.1.2 Step 2.

Converts a (RallySegment, HitEvent) pair into the dict of tensors that BST
expects. Reuses the normalisation logic from
``preparing_data/prepare_train_on_shuttleset.py``.
"""
from __future__ import annotations

import numpy as np
from typing import Dict


class FeatureExtractor:
    """Stateless adapter — produces a dict of fixed-length tensors."""

    def __init__(self, seq_len: int = 30, n_people: int = 2,
                 n_joints: int = 17, frame_w: int = 1280, frame_h: int = 720):
        self.seq_len = seq_len
        self.n_people = n_people
        self.n_joints = n_joints
        self.frame_w = frame_w
        self.frame_h = frame_h

    # ------------------------------------------------------------------
    def _pad_or_crop(self, arr: np.ndarray) -> np.ndarray:
        T = arr.shape[0]
        if T >= self.seq_len:
            stride = max(1, T // self.seq_len)
            return arr[: stride * self.seq_len: stride][: self.seq_len]
        pad_shape = (self.seq_len - T,) + arr.shape[1:]
        return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], 0)

    # ------------------------------------------------------------------
    def _normalise_joints_in_bbox(self, joints: np.ndarray) -> np.ndarray:
        """joints: (T, P, J, 2). Returns (T, P, J*2) normalised by bbox."""
        T, P, J, _ = joints.shape
        out = np.zeros((T, P, J * 2), dtype=np.float32)
        for t in range(T):
            for p in range(P):
                pts = joints[t, p]
                if pts.size == 0 or np.allclose(pts, 0):
                    continue
                top_left = pts.min(0)
                bot_right = pts.max(0)
                diag = np.linalg.norm(bot_right - top_left) or 1.0
                norm = (pts - top_left) / diag
                out[t, p] = norm.flatten()
        return out

    # ------------------------------------------------------------------
    def extract(self, joints_seq: np.ndarray, shuttle_xy: np.ndarray,
                court_xy: np.ndarray) -> Dict[str, np.ndarray]:
        """
        joints_seq : (T, P, J, 2)  pixel coords
        shuttle_xy : (T, 2)        pixel coords
        court_xy   : (T, P, 2)     court coords [0,1] (after homography)
        """
        joints_seq = self._pad_or_crop(np.asarray(joints_seq, dtype=np.float32))
        shuttle_xy = self._pad_or_crop(np.asarray(shuttle_xy, dtype=np.float32))
        court_xy   = self._pad_or_crop(np.asarray(court_xy,   dtype=np.float32))

        human_pose = self._normalise_joints_in_bbox(joints_seq)
        shuttle_norm = shuttle_xy / np.array([self.frame_w, self.frame_h],
                                              dtype=np.float32)
        return {
            "human_pose": human_pose,
            "pos":        court_xy,
            "shuttle":    shuttle_norm,
            "video_len":  int(min(self.seq_len, joints_seq.shape[0])),
        }
