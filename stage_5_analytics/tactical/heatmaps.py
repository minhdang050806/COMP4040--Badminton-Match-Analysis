"""Heatmaps + landing-zone cross tabs — Section 4.4.3 / 4.4.4."""
from __future__ import annotations

import numpy as np
import pandas as pd


def court_heatmap(df: pd.DataFrame, x_col: str, y_col: str,
                  bins=(30, 15), sigma: float = 1.5) -> np.ndarray:
    pts = df[[x_col, y_col]].dropna().values
    if len(pts) == 0:
        return np.zeros(bins, dtype=np.float32)
    h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1],
                             bins=list(bins), range=[[0, 1], [0, 1]])
    try:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, sigma=sigma)
    except ImportError:
        pass
    return h


def landing_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df.hitter_court_zone, df.landing_zone, normalize="index")
