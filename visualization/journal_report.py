#!/usr/bin/env python3
"""Generate neutral, manuscript-ready figures from current raw-video outputs."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/journal_report_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
OUTPUT = ROOT / "project" / "outputs" / "journal_report_figures"
INTEGRATION = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44"
MINING = ROOT / "project" / "outputs" / "mining" / "bst_tracked_phase09_41_44"


def save_workflow() -> None:
    boxes = [
        ("Training videos 1-39\nannotations + feature clips", (0.10, 0.82), "#d9eaf7"),
        ("SA-CNN training\nbinary rally/non-play frames\nmetric: validation F1", (0.38, 0.88), "#d9eaf7"),
        ("BST training\npose + position + shuttle\nmetric: macro-F1", (0.68, 0.88), "#d9eaf7"),
        ("Selected checkpoints\nSA-CNN + BST", (0.91, 0.82), "#d9eaf7"),
        ("Unseen raw videos 41-44\nbroadcast MP4", (0.08, 0.43), "#e4f2df"),
        ("Rally filtering\naccepted play intervals", (0.27, 0.43), "#e4f2df"),
        ("TrackNetV3 + denoising\nframe-level shuttle trajectory", (0.48, 0.43), "#e4f2df"),
        ("YOLO26x-pose + positions\njoints (T,2,17,2)\npos (T,2,2), shuttle (T,2)", (0.70, 0.43), "#e4f2df"),
        ("Integrated BST inference\nstroke, side, confidence", (0.91, 0.43), "#e4f2df"),
        ("Structured rally sequences\n3,178 clean stroke predictions", (0.70, 0.10), "#f7edcf"),
        ("Sequence mining\nMarkov transitions + frequent motifs", (0.90, 0.10), "#f7edcf"),
        ("Profile mining\ndistributions + movement + KMeans", (0.48, 0.10), "#f7edcf"),
    ]
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis("off")
    for label, (x, y), color in boxes:
        ax.text(
            x, y, label, ha="center", va="center", fontsize=13,
            bbox={"boxstyle": "round,pad=0.55", "fc": color, "ec": "#365f7d", "lw": 1.8},
        )
    positions = [position for _, position, _ in boxes]
    arrows = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (3, 8), (8, 9), (9, 10), (9, 11)]
    for source, target in arrows:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#365f7d", "shrinkA": 58, "shrinkB": 58},
        )
    ax.text(0.02, 0.96, "Model development", fontsize=18, fontweight="bold", color="#274f6b")
    ax.text(0.02, 0.62, "Unseen-video inference", fontsize=18, fontweight="bold", color="#365f35")
    ax.text(0.02, 0.18, "Analytical outputs", fontsize=18, fontweight="bold", color="#795d18")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT / "raw_video_workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def clean_predictions() -> pd.DataFrame:
    frame = pd.read_csv(INTEGRATION / "phase10_structured_strokes.csv")
    aliases = frame["feature_alias_source_clip_id"].fillna("").astype(str)
    return frame[aliases == ""].copy()


def cluster_centroids(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = sorted(predictions["predicted_stroke_type"].unique())
    units = []
    for (video_id, side), group in predictions.groupby(["video_id", "predicted_player_side"]):
        histogram = group["predicted_stroke_type"].value_counts(normalize=True)
        units.append(
            {"video_id": video_id, "predicted_player_side": side, **{state: float(histogram.get(state, 0.0)) for state in states}}
        )
    units_frame = pd.DataFrame(units)
    assignments = pd.read_csv(MINING / "clusters_players_pred.csv")
    units_frame = units_frame.merge(assignments[["video_id", "predicted_player_side", "cluster", "pca_1", "pca_2"]])
    centroids = units_frame.groupby("cluster")[states].mean()
    centroids.to_csv(OUTPUT / "cluster_centroids.csv")
    return units_frame, centroids


def save_stroke_distribution(predictions: pd.DataFrame) -> None:
    distribution = pd.crosstab(
        predictions["video_id"], predictions["predicted_stroke_type"], normalize="index"
    )
    ordered = distribution.mean().sort_values(ascending=False).index
    distribution = distribution[ordered]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    distribution.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Unseen evaluation video")
    ax.set_ylabel("Predicted stroke proportion")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(title="Stroke type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=13, title_fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "predicted_stroke_distributions.png", dpi=180)
    plt.close(fig)


def save_centroid_plot(centroids: pd.DataFrame) -> None:
    active = centroids.columns[np.ptp(centroids.to_numpy(), axis=0) > 1e-12]
    plot = centroids[active].T
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(plot))
    width = 0.38
    for offset, cluster in zip((-width / 2, width / 2), plot.columns):
        ax.bar(x + offset, plot[cluster], width, label=f"Cluster {cluster}")
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=45, ha="right", fontsize=13)
    ax.set_ylabel("Centroid stroke proportion")
    ax.set_ylim(0, max(0.22, float(plot.to_numpy().max()) * 1.15))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "cluster_centroid_comparison.png", dpi=180)
    plt.close(fig)


def save_cluster_projection(units: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for cluster, group in units.groupby("cluster"):
        ax.scatter(group["pca_1"], group["pca_2"], s=90, label=f"Cluster {cluster}")
        for row in group.itertuples():
            ax.annotate(f"{row.video_id}-{row.predicted_player_side}", (row.pca_1, row.pca_2), xytext=(5, 5), textcoords="offset points", fontsize=12)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "cluster_projection.png", dpi=180)
    plt.close(fig)


def save_quality_accuracy() -> None:
    groups = pd.read_csv(INTEGRATION / "phase10_evaluation_by_group.csv")
    selected = groups[groups["evaluation_group"].isin([
        "reference_test_top_pose_valid_ge_75",
        "reference_test_top_pose_valid_lt_50",
        "reference_test_pose_dropout_lt_25",
        "reference_test_primary",
    ])].copy()
    labels = {
        "reference_test_top_pose_valid_ge_75": "Top pose valid >=75%",
        "reference_test_top_pose_valid_lt_50": "Top pose valid <50%",
        "reference_test_pose_dropout_lt_25": "Pose dropout <25%",
        "reference_test_primary": "All clean test rows",
    }
    selected["label"] = selected["evaluation_group"].map(labels)
    selected = selected.set_index("label").loc[list(labels.values())].reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(selected["label"], selected["accuracy"], color=["#4daf4a", "#e41a1c", "#377eb8", "#984ea3"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in selected["accuracy"]], padding=4, fontsize=14)
    ax.set_ylabel("Side-aware classification accuracy")
    ax.set_ylim(0, 0.9)
    ax.tick_params(axis="x", rotation=20, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "accuracy_by_feature_quality.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = clean_predictions()
    units, centroids = cluster_centroids(predictions)
    save_workflow()
    save_stroke_distribution(predictions)
    save_centroid_plot(centroids)
    save_cluster_projection(units)
    save_quality_accuracy()
    shutil.copy2(
        ROOT / "project" / "outputs" / "visualizations" / "phase06_rally_comparison" / "phase06_rallies_side_by_side.jpg",
        OUTPUT / "tracked_rally_examples.jpg",
    )
    shutil.copy2(
        MINING / "figures" / "transition_heatmap_pred_hard.png",
        OUTPUT / "predicted_transition_heatmap.png",
    )
    shutil.copy2(
        MINING / "figures" / "player_event_occupancy_pred.png",
        OUTPUT / "player_event_occupancy.png",
    )


if __name__ == "__main__":
    main()
