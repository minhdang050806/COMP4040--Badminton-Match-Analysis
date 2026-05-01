"""Stage 5 — Feature engineering.

Implements Section 4.1 of the architecture document. Reads the strokes
CSV produced by Stage 4 and emits per-stroke / per-player feature frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


COURT_GRID = (6, 3)        # 6 columns × 3 rows for hitter / defender / landing


def _zone(x, y, grid=COURT_GRID):
    if pd.isna(x) or pd.isna(y):
        return -1
    cx = min(int(x * grid[0]), grid[0] - 1)
    cy = min(int(y * grid[1]), grid[1] - 1)
    return cy * grid[0] + cx


def build_stroke_features(strokes_csv: str) -> pd.DataFrame:
    df = pd.read_csv(strokes_csv)

    df["hitter_court_zone"]   = df.apply(lambda r: _zone(r.hitter_x,   r.hitter_y),   axis=1)
    df["defender_court_zone"] = df.apply(lambda r: _zone(r.defender_x, r.defender_y), axis=1)
    df["landing_zone"]        = df.apply(lambda r: _zone(r.landing_x,  r.landing_y),  axis=1)

    df["hitter_to_center_dist"] = np.sqrt(
        (df.hitter_x - 0.5) ** 2 + (df.hitter_y - 0.5) ** 2)
    df["defender_to_net_dist"]  = (df.defender_y - 0.5).abs()
    df["shot_depth"]            = df.landing_y - 0.5
    df["shot_cross_court"]      = (df.landing_x - df.hitter_x).abs() > 0.3

    return df


def build_player_features(stroke_df: pd.DataFrame, n_classes: int = 25
                          ) -> pd.DataFrame:
    """One row per (video_name, hitter)."""
    rows = []
    for (video, player), grp in stroke_df.groupby(["video_name", "hitter"]):
        hist = np.zeros(n_classes, dtype=np.float32)
        for c in grp.stroke_class.dropna().astype(int):
            if 0 <= c < n_classes:
                hist[c] += 1
        if hist.sum() > 0:
            hist = hist / hist.sum()

        rows.append({
            "video_name":               video,
            "hitter":                   player,
            **{f"sc_{i}": hist[i] for i in range(n_classes)},
            "backhand_rate":            grp.backhand.mean(),
            "avg_hitter_court_depth":   grp.hitter_y.mean(),
            "avg_landing_depth":        grp.landing_y.mean(),
            "smash_rate":               (grp.stroke_class_name == "smash").mean(),
            "net_shot_rate":            (grp.stroke_class_name == "net_shot").mean(),
            "avg_shot_confidence":      grp.shot_depth.abs().mean(),
            "win_rate_as_hitter":       grp.winner.mean(),
            "n_shots":                  len(grp),
        })
    return pd.DataFrame(rows)
