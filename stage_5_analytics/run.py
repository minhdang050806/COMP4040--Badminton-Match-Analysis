"""Stage-5 entry point: runs all DM analyses end-to-end."""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config                                # noqa: E402
from common.io import ensure_dir, save_json                          # noqa: E402
from stage_5_analytics import (                                       # noqa: E402
    build_stroke_features, build_player_features,
    cluster_players, mine_frequent_patterns, build_markov_chain,
    score_rally, court_heatmap, landing_pivot, segment_phases,
)
from stage_5_analytics.pattern_mining import encode_rallies           # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    p.add_argument("--strokes_csv", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    csv_path = args.strokes_csv or os.path.join(
        cfg.storage.strokes_csv_dir, "strokes.csv")
    out_dir = ensure_dir(cfg.storage.analytics_dir)

    print(f"[Stage 5] Loading {csv_path}")
    df = build_stroke_features(csv_path)

    # Tactical scoring + phases
    df = score_rally(df)
    df = segment_phases(df, n_states=cfg.analytics.hmm_states)
    df.to_csv(os.path.join(out_dir, "stroke_features.csv"), index=False)

    # Player clustering
    pdf = build_player_features(df, n_classes=cfg.bst.n_classes)
    pdf.to_csv(os.path.join(out_dir, "player_features.csv"), index=False)
    cluster = cluster_players(pdf, k=cfg.analytics.clustering_n)
    cluster["labels"].to_csv(os.path.join(out_dir, "player_clusters.csv"),
                             index=False)
    save_json(os.path.join(out_dir, "cluster_meta.json"),
              {"k": int(cluster["k"]),
               "silhouette": float(cluster["silhouette"]) if cluster["silhouette"] else None})

    # Sequence patterns
    seqs = encode_rallies(df)
    patterns = mine_frequent_patterns(
        seqs,
        min_support=cfg.analytics.pattern_min_support,
        max_len=cfg.analytics.pattern_max_len,
    )
    save_json(os.path.join(out_dir, "frequent_patterns.json"),
              [{"support": s, "pattern": p} for s, p in patterns])

    # Markov transition matrix
    T = build_markov_chain(df, n_classes=cfg.bst.n_classes)
    np.save(os.path.join(out_dir, "transition_matrix.npy"), T)

    # Heatmaps + landing pivot
    np.save(os.path.join(out_dir, "hitter_heatmap.npy"),
            court_heatmap(df, "hitter_x", "hitter_y"))
    np.save(os.path.join(out_dir, "landing_heatmap.npy"),
            court_heatmap(df, "landing_x", "landing_y"))
    landing_pivot(df).to_csv(os.path.join(out_dir, "landing_pivot.csv"))

    print(f"[Stage 5] Analytics written to {out_dir}")


if __name__ == "__main__":
    main()
