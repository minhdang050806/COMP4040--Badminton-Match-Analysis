"""Player clustering — Section 4.2.

K-Means with PCA dim-reduction and silhouette sweep for k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_players(player_df: pd.DataFrame, k: int = 4,
                    pca_components: int = 10, sweep_k: bool = True
                    ) -> dict:
    feat_cols = [c for c in player_df.columns
                 if c not in {"video_name", "hitter"}]
    X = player_df[feat_cols].fillna(0).values.astype(np.float32)
    if len(X) < 2:
        player_df = player_df.copy()
        player_df["cluster"] = 0
        return {"labels": player_df, "k": 1, "centroids": None,
                "silhouette": None}

    X_scaled = StandardScaler().fit_transform(X)
    n_pca = min(pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    X_pca = PCA(n_components=n_pca).fit_transform(X_scaled)

    sil = None
    best_k = k
    if sweep_k and X_pca.shape[0] >= 4:
        scores = {}
        for kk in range(2, min(8, X_pca.shape[0])):
            labels = KMeans(n_clusters=kk, n_init=20, random_state=42
                            ).fit_predict(X_pca)
            scores[kk] = silhouette_score(X_pca, labels)
        best_k = max(scores, key=scores.get)
        sil = scores[best_k]

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(X_pca)
    out = player_df.copy()
    out["cluster"] = km.labels_
    return {"labels": out, "k": best_k,
            "centroids": km.cluster_centers_, "silhouette": sil}
