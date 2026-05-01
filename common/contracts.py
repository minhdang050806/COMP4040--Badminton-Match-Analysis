"""Stage data contracts shared across the pipeline.

Each dataclass corresponds to one of the contracts defined in
system_architecture.md (Section 3.1). They are JSON-serialisable through
`asdict` + the helpers in common.io.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np


def _to_list(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


@dataclass
class RallySegment:
    """Contract A — output of Stage 1."""
    video_path: str
    rally_id: int
    start_frame: int
    end_frame: int
    clip_path: str = ""
    joints_path: str = ""
    shuttle_csv_path: str = ""
    court_corners: Optional[list] = None     # (6, 2)
    homography: Optional[list] = None        # (3, 3)
    fps: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["court_corners"] = _to_list(self.court_corners)
        d["homography"] = _to_list(self.homography)
        return d


@dataclass
class HitEvent:
    """Contract B — output of Stage 3."""
    rally_id: int
    shot_seq: int
    hit_frame: int
    direction_before: int = -1
    direction_after: int = -1
    joints_at_hit: Optional[list] = None    # (2, 17, 2)
    shuttle_at_hit: Optional[list] = None   # (2,)
    court_position: Optional[list] = None   # (2, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("joints_at_hit", "shuttle_at_hit", "court_position"):
            d[k] = _to_list(getattr(self, k))
        return d


@dataclass
class StrokeRecord:
    """Contract C — output of Stage 4 (one row per stroke)."""
    video_name: str
    rally_id: int
    shot_seq: int
    hit_frame: int
    # BST output
    stroke_class: int = -1
    stroke_class_name: str = ""
    stroke_probs: Optional[list] = None
    # ViT outputs
    hitter: int = -1
    backhand: int = -1
    ball_height: int = -1
    ball_type: int = -1
    winner: int = -1
    # locations (court coordinates [0,1])
    hitter_xy_court: Optional[list] = None
    defender_xy_court: Optional[list] = None
    landing_xy_court: Optional[list] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("stroke_probs", "hitter_xy_court",
                  "defender_xy_court", "landing_xy_court"):
            d[k] = _to_list(getattr(self, k))
        return d


# CSV column ordering used by Stage 4 → Stage 5
STROKE_CSV_COLUMNS: List[str] = [
    "video_name", "rally_id", "shot_seq", "hit_frame",
    "stroke_class", "stroke_class_name",
    "hitter", "backhand", "ball_height", "ball_type", "winner",
    "hitter_x", "hitter_y", "defender_x", "defender_y",
    "landing_x", "landing_y",
]
