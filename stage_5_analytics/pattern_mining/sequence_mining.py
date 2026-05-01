"""Pattern mining — Section 4.3.

Sequence encoding + PrefixSpan + first-order Markov chain over stroke
classes. PrefixSpan import is optional; we fall back to a naive frequent
n-gram counter if the package is missing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd


def encode_rallies(df: pd.DataFrame) -> List[List[str]]:
    """One sequence per (video_name, rally_id), ordered by shot_seq."""
    sequences = []
    for _, grp in df.sort_values("shot_seq").groupby(["video_name", "rally_id"]):
        seq = [
            f"{int(r.stroke_class)}_{int(r.hitter_court_zone)}_{int(r.ball_height)}"
            for r in grp.itertuples(index=False)
            if r.stroke_class is not None and not pd.isna(r.stroke_class)
        ]
        if seq:
            sequences.append(seq)
    return sequences


def mine_frequent_patterns(sequences: List[List[str]],
                           min_support: float = 0.05,
                           max_len: int = 6
                           ) -> List[Tuple[float, list]]:
    if not sequences:
        return []
    try:
        from prefixspan import PrefixSpan
        ps = PrefixSpan(sequences)
        ps.maxlen = max_len
        thresh = max(1, int(min_support * len(sequences)))
        return ps.frequent(thresh)
    except ImportError:
        # Fallback: count all contiguous n-grams up to max_len
        counts: Counter = Counter()
        for seq in sequences:
            for n in range(1, max_len + 1):
                for i in range(len(seq) - n + 1):
                    counts[tuple(seq[i:i + n])] += 1
        thresh = max(1, int(min_support * len(sequences)))
        return [(c, list(p)) for p, c in counts.items() if c >= thresh]


def build_markov_chain(df: pd.DataFrame, n_classes: int = 25) -> np.ndarray:
    T = np.zeros((n_classes, n_classes), dtype=np.float32)
    for _, grp in df.sort_values("shot_seq").groupby(["video_name", "rally_id"]):
        cls = grp.stroke_class.dropna().astype(int).values
        for i in range(len(cls) - 1):
            a, b = cls[i], cls[i + 1]
            if 0 <= a < n_classes and 0 <= b < n_classes:
                T[a, b] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums
