"""Offensive-defensive scoring — Section 4.4.1."""
from __future__ import annotations

import numpy as np
import pandas as pd


OFFENSIVE_WEIGHTS = {
    "smash":  +1.0, "rush_to_kill": +1.0, "rush": +0.8, "push": +0.7,
    "drop":   +0.5, "drive": +0.4, "net_shot": +0.3, "block": +0.0,
    "clear": -0.3,  "defensive_lift": -0.8, "lift": -0.8, "lob": -0.7,
}

# Court zones using the 6×3 grid: front rows = small y, rear = large y.
FRONT_ZONES = {0, 1, 2, 3, 4, 5}
REAR_ZONES  = {12, 13, 14, 15, 16, 17}


def offensive_score(stroke_class_name: str, hitter_zone: int) -> float:
    base = OFFENSIVE_WEIGHTS.get(stroke_class_name, 0.0)
    if hitter_zone in FRONT_ZONES:
        base += 0.3
    if hitter_zone in REAR_ZONES:
        base -= 0.2
    return base


def score_rally(stroke_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Adds ``offensive_score`` and rolling-mean momentum per rally."""
    df = stroke_df.copy().sort_values(["video_name", "rally_id", "shot_seq"])
    df["offensive_score"] = df.apply(
        lambda r: offensive_score(r.stroke_class_name, r.hitter_court_zone),
        axis=1,
    )
    df["offensive_momentum"] = (
        df.groupby(["video_name", "rally_id"])["offensive_score"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df
