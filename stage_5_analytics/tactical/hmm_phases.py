"""HMM tactical-phase segmentation — Section 4.4.2."""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_phases(df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """Adds a ``phase`` column with HMM-decoded states {0,1,2}."""
    out = df.copy()
    try:
        from hmmlearn import hmm
    except ImportError:
        out["phase"] = (out["offensive_score"] > 0.3).astype(int)
        return out

    out["phase"] = -1
    for keys, grp in out.groupby(["video_name", "rally_id"]):
        scores = grp["offensive_score"].values.reshape(-1, 1)
        if len(scores) < n_states:
            continue
        try:
            model = hmm.GaussianHMM(n_components=n_states,
                                    covariance_type="diag",
                                    n_iter=50, random_state=42)
            model.fit(scores)
            states = model.predict(scores)
            out.loc[grp.index, "phase"] = states
        except Exception:
            continue
    return out
