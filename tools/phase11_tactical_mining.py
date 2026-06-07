#!/usr/bin/env python3
"""Phase 11: tactical mining over ground truth and integrated predictions."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase11_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_INTEGRATION = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44"
DEFAULT_BASELINE_MINING = ROOT / "project" / "outputs" / "mining" / "baseline_old_phase09"
DEFAULT_OUTPUT = ROOT / "project" / "outputs" / "mining" / "bst_tracked_phase09_41_44"

ZH2EN = {
    "放小球": "net_shot",
    "挑球": "lob",
    "擋小球": "defensive_net",
    "推球": "push",
    "長球": "clear",
    "殺球": "smash",
    "切球": "drop",
    "發短球": "short_service",
    "點扣": "tap_smash",
    "未知球種": "unknown",
    "勾球": "cross_net",
    "過渡切球": "transition_drop",
    "平球": "drive",
    "撲球": "rush",
    "後場抽平球": "backcourt_drive",
    "防守回抽": "defensive_drive",
    "發長球": "long_service",
    "防守回挑": "defensive_lob",
    "小平球": "mini_drive",
}


def clean_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    aliases = pred["feature_alias_source_clip_id"].fillna("").astype(str)
    return pred[aliases == ""].copy()


def predicted_sequences(
    pred: pd.DataFrame,
    confidence: float | None = None,
    column: str = "predicted_label_name",
) -> list[list[str]]:
    sequences: list[list[str]] = []
    for _, group in pred.groupby(["video_id", "rally_id"], sort=False):
        group = group.sort_values("event_rank")
        current: list[str] = []
        previous_rank: int | None = None
        for row in group.itertuples():
            rank = int(row.event_rank)
            accepted = confidence is None or float(row.confidence) >= confidence
            contiguous = previous_rank is None or rank == previous_rank + 1
            if accepted and contiguous:
                current.append(getattr(row, column))
            elif accepted:
                if current:
                    sequences.append(current)
                current = [getattr(row, column)]
            else:
                if current:
                    sequences.append(current)
                current = []
            previous_rank = rank
        if current:
            sequences.append(current)
    return sequences


def ground_truth_sequences(gt: pd.DataFrame) -> list[list[str]]:
    sequences: list[list[str]] = []
    for _, group in gt.groupby(["match_id", "set_id", "rally_id"], sort=False):
        group = group.sort_values("ball_round_id")
        sequences.append([ZH2EN.get(value, "unknown") for value in group["stroke_type_ground_truth"]])
    return sequences


def labeled_sequences(pred: pd.DataFrame, column: str) -> list[list[str]]:
    return [
        list(group.sort_values("event_rank")[column])
        for _, group in pred.groupby(["video_id", "rally_id"], sort=False)
    ]


def transition_matrix(sequences: Iterable[list[str]], states: list[str]) -> tuple[np.ndarray, np.ndarray]:
    state_to_id = {state: index for index, state in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=np.float64)
    for sequence in sequences:
        for current, following in zip(sequence[:-1], sequence[1:]):
            if current in state_to_id and following in state_to_id:
                counts[state_to_id[current], state_to_id[following]] += 1
    probabilities = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )
    return counts, probabilities


def soft_transition_matrix(pred: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    counts = np.zeros((probabilities.shape[1], probabilities.shape[1]), dtype=np.float64)
    for _, group in pred.groupby(["video_id", "rally_id"], sort=False):
        indices = list(group.sort_values("event_rank")["row_index"].astype(int))
        for current, following in zip(indices[:-1], indices[1:]):
            counts += np.outer(probabilities[current], probabilities[following])
    return np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )


def save_matrix(matrix: np.ndarray, states: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=states, columns=states).to_csv(path)


def plot_matrix(matrix: np.ndarray, states: list[str], title: str, path: Path, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(states) * 0.48), max(7, len(states) * 0.48)))
    image = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(states)), states, rotation=90, fontsize=7)
    ax.set_yticks(range(len(states)), states, fontsize=7)
    ax.set_xlabel("next stroke")
    ax.set_ylabel("current stroke")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def transition_gap(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    epsilon = 1e-9
    kls = []
    for ref_row, pred_row in zip(reference, predicted):
        if ref_row.sum() > 0 and pred_row.sum() > 0:
            p = ref_row + epsilon
            q = pred_row + epsilon
            kls.append(float(np.sum(p * np.log(p / q))))
    return {
        "frobenius": float(np.linalg.norm(reference - predicted)),
        "mean_row_kl": float(np.mean(kls)) if kls else 0.0,
    }


def strongest_transitions(matrix: np.ndarray, states: list[str], count: int = 20) -> pd.DataFrame:
    rows = []
    for index, state in enumerate(states):
        if matrix[index].sum() > 0:
            following = int(np.argmax(matrix[index]))
            rows.append({"current": state, "next": states[following], "probability": matrix[index, following]})
    return pd.DataFrame(rows).sort_values("probability", ascending=False).head(count)


def mine_patterns(sequences: list[list[str]], min_support: float, top_k: int = 40) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for length in (2, 3, 4):
        support: Counter[tuple[str, ...]] = Counter()
        for sequence in sequences:
            support.update({tuple(sequence[index : index + length]) for index in range(len(sequence) - length + 1)})
        for pattern, count in support.items():
            fraction = count / max(len(sequences), 1)
            if fraction >= min_support:
                rows.append(
                    {
                        "n": length,
                        "pattern": " -> ".join(pattern),
                        "support_count": count,
                        "support_frac": fraction,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["n", "pattern", "support_count", "support_frac"])
    frame = pd.DataFrame(rows).sort_values(["n", "support_frac"], ascending=[True, False])
    return frame.groupby("n", group_keys=False).head(top_k).reset_index(drop=True)


def opening_patterns(gt: pd.DataFrame, min_support: float) -> pd.DataFrame:
    rows = []
    total = 0
    counter: Counter[tuple[str, ...]] = Counter()
    wins: Counter[tuple[str, ...]] = Counter()
    for _, group in gt.groupby(["match_id", "set_id", "rally_id"], sort=False):
        group = group.sort_values("ball_round_id")
        if len(group) < 3:
            continue
        total += 1
        pattern = tuple(ZH2EN.get(value, "unknown") for value in group["stroke_type_ground_truth"].iloc[:3])
        counter[pattern] += 1
        winner = group["getpoint_player"].dropna()
        if len(winner) and winner.iloc[-1] == group.iloc[0]["player"]:
            wins[pattern] += 1
    for pattern, count in counter.items():
        if count / total >= min_support:
            rows.append(
                {
                    "opening": " -> ".join(pattern),
                    "support_count": count,
                    "support_frac": count / total,
                    "server_win_rate": wins[pattern] / count,
                }
            )
    return pd.DataFrame(rows).sort_values("support_frac", ascending=False).head(40)


def distribution_tables(gt: pd.DataFrame, pred: pd.DataFrame, high_conf: pd.DataFrame, output: Path) -> dict[str, Any]:
    gt_frame = gt.copy()
    gt_frame["stroke_type"] = gt_frame["stroke_type_ground_truth"].map(ZH2EN).fillna("unknown")
    gt_frame.groupby(["player", "stroke_type"]).size().rename("count").reset_index().to_csv(
        output / "stroke_distribution_gt.csv", index=False
    )
    pred.groupby(["predicted_player_side", "predicted_stroke_type"]).size().rename("count").reset_index().to_csv(
        output / "stroke_distribution_pred.csv", index=False
    )
    high_conf.groupby(["predicted_player_side", "predicted_stroke_type"]).size().rename("count").reset_index().to_csv(
        output / "stroke_distribution_pred_high_conf.csv", index=False
    )
    gt_lengths = gt.groupby(["match_id", "set_id", "rally_id"]).size()
    pred_lengths = pred.groupby(["video_id", "rally_id"]).size()
    return {
        "gt_strokes": int(len(gt)),
        "gt_rallies": int(len(gt_lengths)),
        "gt_mean_rally_length": float(gt_lengths.mean()),
        "pred_clean_strokes": int(len(pred)),
        "pred_high_conf_strokes": int(len(high_conf)),
        "pred_rallies": int(len(pred_lengths)),
        "pred_mean_rally_length": float(pred_lengths.mean()),
        "none_rate": float((pred["predicted_label_name"] == "none").mean()),
        "high_conf_coverage": float(len(high_conf) / len(pred)),
    }


def build_profiles(frame: pd.DataFrame, group_columns: list[str], token_column: str, states: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    units: list[dict[str, Any]] = []
    features: list[list[float]] = []
    for key, group in frame.groupby(group_columns):
        if not isinstance(key, tuple):
            key = (key,)
        histogram = group[token_column].value_counts(normalize=True)
        vector = [float(histogram.get(state, 0.0)) for state in states]
        unit = {column: value for column, value in zip(group_columns, key)}
        unit.update({"n_strokes": len(group), "mean_confidence": float(group["confidence"].mean()) if "confidence" in group else 1.0})
        units.append(unit)
        features.append(vector)
    return pd.DataFrame(units), np.asarray(features, dtype=np.float64)


def cluster_profiles(
    units: pd.DataFrame,
    features: np.ndarray,
    states: list[str],
    output_csv: Path,
    profile_csv: Path,
    diagnostics_csv: Path,
    figure_path: Path,
    title: str,
    bootstrap_repeats: int = 100,
) -> dict[str, Any]:
    if len(units) < 4:
        raise ValueError("Clustering requires at least four profile units.")

    # Exclude absent stroke classes so StandardScaler cannot amplify numerical noise.
    active = np.ptp(features, axis=0) > 1e-12
    active_features = features[:, active]
    scaled = StandardScaler().fit_transform(active_features)
    max_k = min(8, len(units) - 1)
    rng = np.random.default_rng(0)
    diagnostics: list[dict[str, Any]] = []
    fitted: dict[int, tuple[KMeans, np.ndarray]] = {}

    for clusters in range(2, max_k + 1):
        model = KMeans(n_clusters=clusters, n_init=50, random_state=0).fit(scaled)
        labels = model.labels_
        fitted[clusters] = (model, labels)
        stability: list[float] = []
        for _ in range(bootstrap_repeats):
            sample = rng.integers(0, len(units), size=len(units))
            if len(np.unique(sample)) < clusters:
                continue
            bootstrap = KMeans(n_clusters=clusters, n_init=20, random_state=0).fit(scaled[sample])
            stability.append(float(adjusted_rand_score(labels, bootstrap.predict(scaled))))
        counts = np.bincount(labels, minlength=clusters)
        diagnostics.append(
            {
                "k": clusters,
                "silhouette": float(silhouette_score(scaled, labels)),
                "bootstrap_stability_mean_ari": float(np.mean(stability)),
                "bootstrap_stability_std_ari": float(np.std(stability)),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "min_cluster_fraction": float(counts.min() / len(units)),
            }
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(diagnostics_csv, index=False)
    eligible = diagnostics_frame[diagnostics_frame["min_cluster_size"] >= 2]
    selection_pool = eligible if not eligible.empty else diagnostics_frame
    best_row = selection_pool.sort_values(
        ["silhouette", "bootstrap_stability_mean_ari", "min_cluster_fraction"],
        ascending=False,
    ).iloc[0]
    best_k = int(best_row["k"])
    model, labels = fitted[best_k]
    units = units.copy()
    units["cluster"] = labels
    coordinates = PCA(n_components=2).fit_transform(scaled)
    units["pca_1"] = coordinates[:, 0]
    units["pca_2"] = coordinates[:, 1]
    units["distance_to_centroid"] = np.linalg.norm(scaled - model.cluster_centers_[labels], axis=1)
    units.to_csv(output_csv, index=False)

    global_histogram = features.mean(axis=0)
    profiles = []
    for cluster in sorted(set(labels)):
        mean_histogram = features[labels == cluster].mean(axis=0)
        difference = mean_histogram - global_histogram
        strongest = np.argsort(-difference)[:5]
        weakest = np.argsort(difference)[:3]
        profiles.append(
            {
                "cluster": int(cluster),
                "n_units": int(np.sum(labels == cluster)),
                "top_strokes": ", ".join(f"{states[index]}({mean_histogram[index]:.2f})" for index in strongest),
                "above_global_average": ", ".join(f"{states[index]}({difference[index]:+.3f})" for index in strongest),
                "below_global_average": ", ".join(f"{states[index]}({difference[index]:+.3f})" for index in weakest),
                "mean_distance_to_centroid": float(units.loc[units["cluster"] == cluster, "distance_to_centroid"].mean()),
            }
        )
    pd.DataFrame(profiles).to_csv(profile_csv, index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=55)
    ax.set_title(
        f"{title}\nk={best_k}, silhouette={best_row['silhouette']:.3f}, "
        f"bootstrap ARI={best_row['bootstrap_stability_mean_ari']:.3f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {
        "n_units": len(units),
        "n_active_stroke_features": int(active.sum()),
        "best_k": best_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_bootstrap_stability_mean_ari": float(best_row["bootstrap_stability_mean_ari"]),
        "best_min_cluster_size": int(best_row["min_cluster_size"]),
        "diagnostics": diagnostics,
    }


def draw_court(ax: plt.Axes) -> None:
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="white", linewidth=1)
    ax.axhline(0.5, color="white", linewidth=1)
    ax.axvline(0.5, color="white", linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")


def event_positions(pred: pd.DataFrame, positions: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {"Top": [], "Bottom": []}
    for row in pred.itertuples():
        index = min(max(int(row.event_offset_frames), 0), int(row.video_len_after_collation) - 1)
        clip = positions[int(row.row_index)]
        for player, name in ((0, "Top"), (1, "Bottom")):
            point = clip[index, player]
            if np.any(point != 0.0) and np.all((-0.2 <= point) & (point <= 1.2)):
                output[name].append(point)
    return {name: np.asarray(points) for name, points in output.items()}


def plot_predicted_occupancy(pred: pd.DataFrame, positions: np.ndarray, figure: Path) -> dict[str, int]:
    points = event_positions(pred, positions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    counts = {}
    for ax, name in zip(axes, ("Top", "Bottom")):
        values = points[name]
        counts[name] = len(values)
        if len(values):
            histogram, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=35, range=[[0, 1], [0, 1]])
            ax.imshow(histogram.T, origin="upper", extent=[0, 1, 1, 0], cmap="hot")
        draw_court(ax)
        ax.set_title(f"{name} event-frame occupancy, n={len(values)}")
    fig.tight_layout()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    return counts


def plot_gt_heatmaps(gt: pd.DataFrame, figures: Path) -> dict[str, int]:
    numeric = gt.copy()
    for column in ("landing_x", "landing_y", "player_location_x", "player_location_y"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    outputs = {}
    for x_column, y_column, name, title in (
        ("landing_x", "landing_y", "landing_heatmap_gt_all.png", "GT shuttle landing density"),
        ("player_location_x", "player_location_y", "player_position_heatmap_gt.png", "GT player position density"),
    ):
        valid = numeric[[x_column, y_column]].dropna()
        outputs[name] = len(valid)
        histogram, x_edges, y_edges = np.histogram2d(valid[x_column], valid[y_column], bins=45)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(histogram.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="hot", aspect="auto")
        ax.invert_yaxis()
        ax.set_title(f"{title}, n={len(valid)}")
        fig.tight_layout()
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)
    frame = numeric.copy()
    frame["stroke_type"] = frame["stroke_type_ground_truth"].map(ZH2EN).fillna("unknown")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, stroke in zip(axes.ravel(), ("smash", "drop", "clear", "net_shot")):
        valid = frame[frame["stroke_type"] == stroke][["landing_x", "landing_y"]].dropna()
        if len(valid):
            histogram, x_edges, y_edges = np.histogram2d(valid["landing_x"], valid["landing_y"], bins=35)
            ax.imshow(histogram.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="hot", aspect="auto")
            ax.invert_yaxis()
        ax.set_title(f"{stroke}, n={len(valid)}")
    fig.suptitle("GT shuttle landing density by stroke")
    fig.tight_layout()
    fig.savefig(figures / "landing_heatmap_gt_by_stroke.png", dpi=160)
    plt.close(fig)
    return outputs


def plot_approximate_shuttle_landings(pred: pd.DataFrame, figure: Path) -> int:
    points = []
    for row in pred.itertuples():
        path = Path(row.shuttle_npy)
        if not path.is_absolute():
            path = ROOT / path
        shuttle = np.load(path)
        visible = shuttle[np.any(shuttle != 0.0, axis=1)]
        if len(visible):
            point = visible[-1]
            if np.all((0 <= point) & (point <= 1)):
                points.append(point)
    values = np.asarray(points)
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(values):
        histogram, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=40, range=[[0, 1], [0, 1]])
        ax.imshow(histogram.T, origin="upper", extent=[0, 1, 1, 0], cmap="hot")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title(f"Approximate predicted shuttle endpoint, n={len(values)}")
    ax.set_xlabel("normalized image x")
    ax.set_ylabel("normalized image y")
    fig.tight_layout()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    return len(values)


def plot_accuracy_diagnostics(pred: pd.DataFrame, figures: Path) -> None:
    frame = pred.copy()
    frame["correct_bool"] = frame["correct"].astype(str).str.lower() == "true"
    frame["confidence_bin"] = pd.cut(frame["confidence"], [0, 0.4, 0.6, 0.8, 1.01], right=False)
    confidence = frame.groupby("confidence_bin", observed=True).agg(rows=("correct_bool", "size"), accuracy=("correct_bool", "mean")).reset_index()
    confidence.to_csv(figures.parent / "accuracy_by_confidence.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(confidence["confidence_bin"].astype(str), confidence["accuracy"], color="#377eb8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Tracked Phase 10 accuracy by confidence")
    fig.tight_layout()
    fig.savefig(figures / "accuracy_by_confidence.png", dpi=160)
    plt.close(fig)

    videos = frame.groupby("video_id").agg(accuracy=("correct_bool", "mean"), top_pose_valid=("p1_pose_valid_rate", "mean")).reset_index()
    videos.to_csv(figures.parent / "accuracy_by_video_and_top_pose.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(videos["top_pose_valid"], videos["accuracy"], s=90)
    for row in videos.itertuples():
        ax.annotate(str(row.video_id), (row.top_pose_valid, row.accuracy), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("mean Top-player pose-valid rate")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy tracks far-side Top-player pose quality")
    fig.tight_layout()
    fig.savefig(figures / "accuracy_vs_top_pose.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11 tactical mining over tracked Phase 10 outputs.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--integration-root", type=Path, default=DEFAULT_INTEGRATION)
    parser.add_argument("--baseline-mining-root", type=Path, default=DEFAULT_BASELINE_MINING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--min-pattern-support", type=float, default=0.02)
    parser.add_argument("--cluster-bootstrap-repeats", type=int, default=100)
    args = parser.parse_args()

    output = args.output_root
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(args.ground_truth)
    all_predictions = pd.read_csv(args.integration_root / "phase10_structured_strokes.csv")
    pred = clean_predictions(all_predictions)
    high_conf = pred[pred["confidence"] >= args.confidence_threshold].copy()
    probabilities = np.load(args.integration_root / "phase10_probabilities.npy")
    positions = np.load(args.integration_root / "inputs" / "pos.npy")
    if len(all_predictions) != len(probabilities) or len(all_predictions) != len(positions):
        raise RuntimeError("Phase 10 rows, probabilities, and positions are not aligned.")

    states25 = [
        name
        for _, name in sorted(
            {(int(row.predicted_label), row.predicted_label_name) for row in all_predictions.itertuples()}
        )
    ]
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "integration_root": str(args.integration_root),
        "output_root": str(output),
        "confidence_threshold": args.confidence_threshold,
        "cluster_bootstrap_repeats": args.cluster_bootstrap_repeats,
        "rows": {"all_predictions": len(all_predictions), "clean_predictions": len(pred), "alias_rows_excluded": len(all_predictions) - len(pred)},
    }
    summary["distribution"] = distribution_tables(gt, pred, high_conf, output)

    gt_sequences = ground_truth_sequences(gt)
    _, gt19 = transition_matrix(gt_sequences, sorted(set(ZH2EN.values())))
    states19 = sorted(set(ZH2EN.values()))
    save_matrix(gt19, states19, output / "transition_matrix_gt.csv")
    plot_matrix(gt19, states19, "Ground-truth stroke transitions, all 44 videos", figures / "transition_heatmap_gt.png")

    true_sequences = labeled_sequences(pred, "true_label_name")
    hard_sequences = labeled_sequences(pred, "predicted_label_name")
    _, true25 = transition_matrix(true_sequences, states25)
    _, hard25 = transition_matrix(hard_sequences, states25)
    soft25 = soft_transition_matrix(pred, probabilities)
    high_sequences = predicted_sequences(pred, args.confidence_threshold)
    high_true_sequences = predicted_sequences(pred, args.confidence_threshold, "true_label_name")
    _, high25 = transition_matrix(high_sequences, states25)
    _, high_true25 = transition_matrix(high_true_sequences, states25)
    for name, matrix in (("gt25_labeled", true25), ("pred_hard", hard25), ("pred_soft", soft25), ("pred_high_conf", high25)):
        save_matrix(matrix, states25, output / f"transition_matrix_{name}.csv")
        plot_matrix(matrix, states25, name.replace("_", " "), figures / f"transition_heatmap_{name}.png")
    difference = np.abs(true25 - hard25)
    save_matrix(difference, states25, output / "transition_matrix_absolute_error.csv")
    plot_matrix(difference, states25, "Absolute transition error: GT minus tracked prediction", figures / "transition_error_heatmap.png", "magma")
    comparison = {
        "hard": transition_gap(true25, hard25),
        "soft": transition_gap(true25, soft25),
        "high_confidence": transition_gap(high_true25, high25),
    }
    baseline_path = args.baseline_mining_root / "transition_comparison.json"
    if baseline_path.exists():
        comparison["old_baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
    (output / "transition_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    strongest_transitions(true25, states25).to_csv(output / "strongest_transitions_gt25.csv", index=False)
    strongest_transitions(hard25, states25).to_csv(output / "strongest_transitions_pred.csv", index=False)
    summary["transition_comparison"] = comparison

    patterns_gt = mine_patterns(gt_sequences, args.min_pattern_support)
    patterns_pred = mine_patterns(hard_sequences, args.min_pattern_support)
    patterns_high = mine_patterns(high_sequences, args.min_pattern_support)
    openings = opening_patterns(gt, 0.01)
    patterns_gt.to_csv(output / "patterns_gt.csv", index=False)
    patterns_pred.to_csv(output / "patterns_pred.csv", index=False)
    patterns_high.to_csv(output / "patterns_pred_high_conf.csv", index=False)
    openings.to_csv(output / "patterns_opening_gt.csv", index=False)
    summary["patterns"] = {"gt": len(patterns_gt), "predicted": len(patterns_pred), "high_confidence": len(patterns_high)}

    gt_profiles = gt.copy()
    gt_profiles["stroke_token"] = gt_profiles["stroke_type_ground_truth"].map(ZH2EN).fillna("unknown")
    gt_profiles["confidence"] = 1.0
    gt_units, gt_features = build_profiles(gt_profiles, ["match_id", "player"], "stroke_token", states19)
    summary["clustering_gt"] = cluster_profiles(
        gt_units,
        gt_features,
        states19,
        output / "clusters_players_gt.csv",
        output / "player_profiles_gt.csv",
        output / "cluster_diagnostics_gt.csv",
        figures / "player_clusters_gt.png",
        "GT player-match style clusters",
        args.cluster_bootstrap_repeats,
    )
    pred_states = sorted(pred["predicted_stroke_type"].unique())
    pred_units, pred_features = build_profiles(
        pred,
        ["video_id", "predicted_player_side"],
        "predicted_stroke_type",
        pred_states,
    )
    summary["clustering_pred"] = cluster_profiles(
        pred_units,
        pred_features,
        pred_states,
        output / "clusters_players_pred.csv",
        output / "player_profiles_pred.csv",
        output / "cluster_diagnostics_pred.csv",
        figures / "player_clusters_pred.png",
        "Predicted video-side style clusters",
        args.cluster_bootstrap_repeats,
    )

    summary["heatmaps_gt"] = plot_gt_heatmaps(gt, figures)
    summary["predicted_event_occupancy"] = plot_predicted_occupancy(pred, positions, figures / "player_event_occupancy_pred.png")
    summary["approximate_shuttle_endpoints"] = plot_approximate_shuttle_landings(
        pred,
        figures / "shuttle_endpoint_pred_approximate.png",
    )
    plot_accuracy_diagnostics(pred[pred["reference_split"] == "test"], figures)
    (output / "phase11_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
