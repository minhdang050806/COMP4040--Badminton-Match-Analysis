#!/usr/bin/env python3
"""Cluster player-centric movement and predicted-stroke strategy profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/movement_strategy_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
    }
)

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44"
OUTPUT = ROOT / "project" / "outputs" / "movement_strategy"

PROFILE_FEATURES = [
    "mean_forward_position",
    "lateral_position_std",
    "forward_position_std",
    "mean_interstroke_displacement",
    "rightward_displacement_rate",
    "netward_displacement_rate",
    "right_baselineward_displacement_rate",
    "smash_rate",
    "net_attack_rate",
    "lift_clear_rate",
    "smash_after_large_displacement_rate",
    "rear_court_smash_rate",
]
MIN_PROFILE_DISPLACEMENTS = 30


def clean_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = frame["feature_alias_source_clip_id"].fillna("").astype(str)
    return frame[aliases == ""].copy()


def player_centric_events(frame: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    events = []
    for row in frame.itertuples():
        true_side = str(row.true_label_name).split("_", 1)[0]
        if true_side not in {"Top", "Bottom"} or pd.isna(row.player):
            continue
        offset = min(max(int(row.event_offset_frames), 0), int(row.video_len_after_collation) - 1)
        point = positions[int(row.row_index), offset, 0 if true_side == "Top" else 1]
        if not np.any(point != 0.0) or not np.all((-0.2 <= point) & (point <= 1.2)):
            continue
        # Rotate the Top half by 180 degrees so every player faces the net at forward=1.
        lateral = float(1.0 - point[0] if true_side == "Top" else point[0])
        forward = float(2.0 * point[1] if true_side == "Top" else 2.0 * (1.0 - point[1]))
        if not 0.0 <= lateral <= 1.0 or not 0.0 <= forward <= 1.0:
            continue
        events.append(
            {
                "row_index": int(row.row_index),
                "video_id": int(row.video_id),
                "set_id": int(row.set_id),
                "rally_id": int(row.rally_id),
                "event_rank": int(row.event_rank),
                "player": str(row.player),
                "true_side_at_event": true_side,
                "predicted_stroke_type": str(row.predicted_stroke_type),
                "confidence": float(row.confidence),
                "lateral_position": lateral,
                "forward_position": forward,
            }
        )
    output = pd.DataFrame(events).sort_values(["video_id", "set_id", "rally_id", "event_rank"])
    output["delta_lateral"] = np.nan
    output["delta_forward"] = np.nan
    output["interstroke_displacement"] = np.nan
    for _, group in output.groupby(["video_id", "set_id", "rally_id", "player"]):
        indices = group.index
        output.loc[indices, "delta_lateral"] = group["lateral_position"].diff()
        output.loc[indices, "delta_forward"] = group["forward_position"].diff()
        output.loc[indices, "interstroke_displacement"] = np.hypot(
            output.loc[indices, "delta_lateral"], output.loc[indices, "delta_forward"]
        )
    return output


def build_profiles(events: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    net_attacks = {"net shot", "cross-court net shot", "push", "rush", "return net"}
    lift_clear = {"lob", "clear"}
    for key, group in events.groupby(["video_id", "set_id", "player"]):
        displacements = group.dropna(subset=["interstroke_displacement"])
        smash = group[group["predicted_stroke_type"] == "smash"]
        profiles.append(
            {
                "video_id": key[0],
                "set_id": key[1],
                "player": key[2],
                "n_events": len(group),
                "n_displacements": len(displacements),
                "mean_confidence": float(group["confidence"].mean()),
                "mean_forward_position": float(group["forward_position"].mean()),
                "lateral_position_std": float(group["lateral_position"].std(ddof=0)),
                "forward_position_std": float(group["forward_position"].std(ddof=0)),
                "mean_interstroke_displacement": float(displacements["interstroke_displacement"].mean()),
                "rightward_displacement_rate": float((displacements["delta_lateral"] > 0.05).mean()),
                "netward_displacement_rate": float((displacements["delta_forward"] > 0.05).mean()),
                "right_baselineward_displacement_rate": float(
                    ((displacements["delta_lateral"] > 0.05) & (displacements["delta_forward"] < -0.05)).mean()
                ),
                "smash_rate": float((group["predicted_stroke_type"] == "smash").mean()),
                "net_attack_rate": float(group["predicted_stroke_type"].isin(net_attacks).mean()),
                "lift_clear_rate": float(group["predicted_stroke_type"].isin(lift_clear).mean()),
                "smash_after_large_displacement_rate": float(
                    (smash["interstroke_displacement"] > 0.20).mean()
                ) if len(smash) else 0.0,
                "rear_court_smash_rate": float((smash["forward_position"] < 0.5).mean()) if len(smash) else 0.0,
            }
        )
    return pd.DataFrame(profiles)


def cluster_profiles(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    scaled = StandardScaler().fit_transform(profiles[PROFILE_FEATURES])
    rng = np.random.default_rng(0)
    diagnostics = []
    models = {}
    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init=50, random_state=0).fit(scaled)
        labels = model.labels_
        stability = []
        for _ in range(100):
            sample = rng.integers(0, len(profiles), size=len(profiles))
            if len(np.unique(sample)) < k:
                continue
            bootstrap = KMeans(n_clusters=k, n_init=20, random_state=0).fit(scaled[sample])
            stability.append(adjusted_rand_score(labels, bootstrap.predict(scaled)))
        counts = np.bincount(labels, minlength=k)
        diagnostics.append(
            {
                "k": k,
                "silhouette": silhouette_score(scaled, labels),
                "bootstrap_stability_mean_ari": np.mean(stability),
                "bootstrap_stability_std_ari": np.std(stability),
                "min_cluster_size": counts.min(),
                "max_cluster_size": counts.max(),
            }
        )
        models[k] = model
    diagnostics_frame = pd.DataFrame(diagnostics)
    eligible = diagnostics_frame[diagnostics_frame["min_cluster_size"] >= 3]
    best = eligible.sort_values(["silhouette", "bootstrap_stability_mean_ari"], ascending=False).iloc[0]
    model = models[int(best["k"])]
    output = profiles.copy()
    output["cluster"] = model.labels_
    pca = PCA(n_components=2).fit_transform(scaled)
    output["pca_1"] = pca[:, 0]
    output["pca_2"] = pca[:, 1]
    centroids = output.groupby("cluster")[PROFILE_FEATURES].mean()
    return output, centroids, diagnostics_frame, {
        "n_units": len(output),
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette"]),
        "best_bootstrap_stability_mean_ari": float(best["bootstrap_stability_mean_ari"]),
        "best_min_cluster_size": int(best["min_cluster_size"]),
    }


def save_figures(events: pd.DataFrame, assignments: pd.DataFrame, centroids: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for cluster, group in assignments.groupby("cluster"):
        ax.scatter(group["pca_1"], group["pca_2"], s=80, label=f"Cluster {cluster}")
        for row in group.itertuples():
            ax.annotate(f"{row.video_id}-S{row.set_id}-{row.player}", (row.pca_1, row.pca_2), xytext=(5, 5), textcoords="offset points", fontsize=12)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "movement_strategy_clusters.png", dpi=180)
    plt.close(fig)

    display_features = [
        "mean_forward_position",
        "mean_interstroke_displacement",
        "rightward_displacement_rate",
        "netward_displacement_rate",
        "right_baselineward_displacement_rate",
        "smash_rate",
        "net_attack_rate",
        "lift_clear_rate",
        "smash_after_large_displacement_rate",
        "rear_court_smash_rate",
    ]
    plot = centroids[display_features].T
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(plot))
    width = 0.8 / len(plot.columns)
    for index, cluster in enumerate(plot.columns):
        ax.bar(x + (index - (len(plot.columns) - 1) / 2) * width, plot[cluster], width, label=f"Cluster {cluster}")
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=40, ha="right", fontsize=13)
    ax.set_ylabel("Original-scale cluster centroid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "movement_strategy_centroids.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(assignments["right_baselineward_displacement_rate"], assignments["smash_rate"], c=assignments["cluster"], cmap="tab10", s=70)
    axes[0].set_xlabel("Right-baselineward displacement rate")
    axes[0].set_ylabel("Predicted smash rate")
    axes[1].scatter(assignments["netward_displacement_rate"], assignments["net_attack_rate"], c=assignments["cluster"], cmap="tab10", s=70)
    axes[1].set_xlabel("Netward displacement rate")
    axes[1].set_ylabel("Predicted net-attack rate")
    fig.tight_layout()
    fig.savefig(OUTPUT / "movement_stroke_relationships.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    histogram, _, _ = np.histogram2d(events["lateral_position"], events["forward_position"], bins=35, range=[[0, 1], [0, 1]])
    ax.imshow(histogram.T, origin="lower", extent=[0, 1, 0, 1], cmap="hot", aspect="auto")
    ax.axhline(1.0, color="white", linewidth=1)
    ax.set_xlabel("Player-centric lateral position (left to right)")
    ax.set_ylabel("Player-centric depth (baseline to net)")
    ax.set_title(f"General event-position occupancy, n={len(events)}")
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "general_player_centric_occupancy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = clean_predictions(pd.read_csv(INTEGRATION / "phase10_structured_strokes.csv"))
    positions = np.load(INTEGRATION / "inputs" / "pos.npy")
    events = player_centric_events(frame, positions)
    all_profiles = build_profiles(events)
    profiles = all_profiles[all_profiles["n_displacements"] >= MIN_PROFILE_DISPLACEMENTS].reset_index(drop=True)
    assignments, centroids, diagnostics, summary = cluster_profiles(profiles)
    events.to_csv(OUTPUT / "movement_events.csv", index=False)
    all_profiles.to_csv(OUTPUT / "movement_strategy_profiles_all.csv", index=False)
    assignments.to_csv(OUTPUT / "movement_strategy_profiles.csv", index=False)
    centroids.to_csv(OUTPUT / "movement_strategy_centroids.csv")
    diagnostics.to_csv(OUTPUT / "movement_strategy_cluster_diagnostics.csv", index=False)
    correlations = profiles[PROFILE_FEATURES].corr()
    correlations.to_csv(OUTPUT / "movement_strategy_correlations.csv")
    summary.update(
        {
            "n_valid_events": len(events),
            "n_valid_displacements": int(events["interstroke_displacement"].notna().sum()),
            "minimum_profile_displacements": MIN_PROFILE_DISPLACEMENTS,
            "n_excluded_sparse_profiles": int(len(all_profiles) - len(profiles)),
            "features": PROFILE_FEATURES,
            "identity_note": "Persistent player A/B and true side at event come from ShuttleSet evaluation metadata; deployment requires player re-identification.",
        }
    )
    (OUTPUT / "movement_strategy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figures(events, assignments, centroids)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
