#!/usr/bin/env python3
"""Phase 12: validate the *mining* outputs, not just the perception.

For every tactical-mining view we run the analysis on the predicted stroke
stream and on the matched ground-truth stroke stream over the same 3,178
evaluated events (videos 41-44) and report the predicted-vs-true gap:

  * composition   -- KL / chi-square / total-variation between predicted and
                     true stroke-type distributions;
  * transitions   -- Frobenius + mean row-KL of the predicted matrix against
                     ground truth, with unigram / uniform-random / earlier-
                     pipeline / current-pipeline / ground-truth baselines;
  * motifs        -- top-k motif-set overlap (Jaccard, overlap@k) and Spearman
                     rank correlation of motif supports;
  * movement      -- Pearson correlations re-reported (already validated);
  * clustering    -- adjusted Rand index between predicted-derived and
                     truth-derived movement-style cluster assignments.

Outputs land in project/outputs/mining_validation/.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase12_matplotlib")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44"
OLD_BASELINE = ROOT / "project" / "outputs" / "mining" / "baseline_old_phase09"
OUTPUT = ROOT / "project" / "outputs" / "mining_validation"

EPS = 1e-9
RANDOM_SEEDS = 50
TOP_K = 10
MIN_SUPPORT = 0.02

# Movement-profile feature set (identical to movement_strategy.py).
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
N_MOVEMENT_PROFILES = 16  # player-set profiles surviving the displacement filter


# --------------------------------------------------------------------------- #
# Sequence / distribution helpers
# --------------------------------------------------------------------------- #
def clean_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = frame["feature_alias_source_clip_id"].fillna("").astype(str)
    return frame[aliases == ""].copy()


def side_agnostic(label: str) -> str:
    """'Top_net shot' -> 'net shot', 'none' -> 'none'."""
    if not isinstance(label, str) or label == "none":
        return "none"
    parts = label.split("_", 1)
    return parts[1] if len(parts) == 2 else label


def labeled_sequences(frame: pd.DataFrame, column: str) -> list[list[str]]:
    """One ordered token list per rally (no contiguity filter), matching phase 11."""
    return [
        list(group.sort_values("event_rank")[column])
        for _, group in frame.groupby(["video_id", "rally_id"], sort=False)
    ]


def stroke_sequences(frame: pd.DataFrame, column: str, drop_none: bool) -> list[list[str]]:
    sequences = []
    for _, group in frame.groupby(["video_id", "rally_id"], sort=False):
        tokens = [side_agnostic(value) for value in group.sort_values("event_rank")[column]]
        if drop_none:
            tokens = [token for token in tokens if token != "none"]
        if tokens:
            sequences.append(tokens)
    return sequences


def distribution(labels: Iterable[str], vocabulary: list[str]) -> np.ndarray:
    counts = Counter(labels)
    vector = np.array([counts.get(token, 0) for token in vocabulary], dtype=np.float64)
    return vector


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((p + EPS) * np.log((p + EPS) / (q + EPS))))


def chi_square_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + EPS)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * np.sum(np.abs(p - q)))


def composition_gap(frame: pd.DataFrame, column_true: str, column_pred: str, drop_none: bool) -> dict:
    true_tokens = [side_agnostic(v) for v in frame[column_true]]
    pred_tokens = [side_agnostic(v) for v in frame[column_pred]]
    if drop_none:
        true_tokens = [t for t in true_tokens if t != "none"]
        pred_tokens = [t for t in pred_tokens if t != "none"]
    vocabulary = sorted(set(true_tokens) | set(pred_tokens))
    p = distribution(true_tokens, vocabulary)  # true
    q = distribution(pred_tokens, vocabulary)  # predicted
    uniform = np.ones_like(p)
    return {
        "vocabulary_size": len(vocabulary),
        "n_true_tokens": int(p.sum()),
        "n_pred_tokens": int(q.sum()),
        "kl_true_pred": kl_divergence(p, q),
        "kl_pred_true": kl_divergence(q, p),
        "kl_true_uniform": kl_divergence(p, uniform),  # baseline: no knowledge
        "chi_square_distance": chi_square_distance(p, q),
        "total_variation": total_variation(p, q),
        "true_distribution": {token: float(v) for token, v in zip(vocabulary, p / p.sum())},
        "pred_distribution": {token: float(v) for token, v in zip(vocabulary, q / q.sum())},
    }


# --------------------------------------------------------------------------- #
# Transition matrices + baselines
# --------------------------------------------------------------------------- #
def transition_counts(sequences: Iterable[list[str]], states: list[str]) -> np.ndarray:
    index = {state: i for i, state in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=np.float64)
    for sequence in sequences:
        for current, following in zip(sequence[:-1], sequence[1:]):
            if current in index and following in index:
                counts[index[current], index[following]] += 1
    return counts


def row_normalize(counts: np.ndarray) -> np.ndarray:
    return np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )


def transition_gap(reference: np.ndarray, candidate: np.ndarray) -> dict:
    kls = []
    for ref_row, cand_row in zip(reference, candidate):
        if ref_row.sum() > 0 and cand_row.sum() > 0:
            p = ref_row + EPS
            q = cand_row + EPS
            kls.append(float(np.sum(p * np.log(p / q))))
    return {
        "frobenius": float(np.linalg.norm(reference - candidate)),
        "mean_row_kl": float(np.mean(kls)) if kls else 0.0,
    }


def unigram_matrix(true_counts: np.ndarray) -> np.ndarray:
    """Memoryless baseline: every active row equals the marginal next-stroke distribution."""
    active = true_counts.sum(axis=1) > 0
    marginal = true_counts.sum(axis=0)
    marginal = marginal / marginal.sum() if marginal.sum() > 0 else marginal
    matrix = np.zeros_like(true_counts)
    matrix[active] = marginal
    return matrix


def random_matrix_gap(true_matrix: np.ndarray, n_states: int, seeds: int) -> dict:
    """Uniform-random row-normalized transitions on the active rows, averaged over seeds."""
    active = true_matrix.sum(axis=1) > 0
    frob, kl = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        candidate = np.zeros_like(true_matrix)
        random_rows = rng.random((n_states, n_states))
        random_rows = random_rows / random_rows.sum(axis=1, keepdims=True)
        candidate[active] = random_rows[active]
        gap = transition_gap(true_matrix, candidate)
        frob.append(gap["frobenius"])
        kl.append(gap["mean_row_kl"])
    return {"frobenius": float(np.mean(frob)), "mean_row_kl": float(np.mean(kl))}


# --------------------------------------------------------------------------- #
# Motif mining
# --------------------------------------------------------------------------- #
def motif_supports(sequences: list[list[str]], n: int) -> dict[tuple[str, ...], float]:
    support: Counter[tuple[str, ...]] = Counter()
    for sequence in sequences:
        present = {tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)}
        support.update(present)
    total = max(len(sequences), 1)
    return {motif: count / total for motif, count in support.items()}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def motif_gap(true_sequences: list[list[str]], pred_sequences: list[list[str]], top_k: int) -> dict:
    per_n = {}
    for n in (2, 3, 4):
        true_support = motif_supports(true_sequences, n)
        pred_support = motif_supports(pred_sequences, n)
        true_top = {m for m, _ in sorted(true_support.items(), key=lambda kv: -kv[1])[:top_k]}
        pred_top = {m for m, _ in sorted(pred_support.items(), key=lambda kv: -kv[1])[:top_k]}
        union_top = true_top | pred_top
        intersect_top = true_top & pred_top
        # Spearman over motifs above MIN_SUPPORT in either stream (0-fill).
        union_motifs = sorted(
            {m for m, s in true_support.items() if s >= MIN_SUPPORT}
            | {m for m, s in pred_support.items() if s >= MIN_SUPPORT}
        )
        true_vec = np.array([true_support.get(m, 0.0) for m in union_motifs])
        pred_vec = np.array([pred_support.get(m, 0.0) for m in union_motifs])
        per_n[n] = {
            "n_true_motifs_above_support": int(sum(1 for s in true_support.values() if s >= MIN_SUPPORT)),
            "n_pred_motifs_above_support": int(sum(1 for s in pred_support.values() if s >= MIN_SUPPORT)),
            "jaccard_top_k": len(intersect_top) / len(union_top) if union_top else 0.0,
            "overlap_at_k": len(intersect_top) / top_k,
            "spearman_support": spearman(true_vec, pred_vec),
            "n_union_motifs": len(union_motifs),
            "top_true_motifs": [(" -> ".join(m), round(true_support[m], 4)) for m in sorted(true_top, key=lambda m: -true_support[m])],
            "top_pred_motifs": [(" -> ".join(m), round(pred_support[m], 4)) for m in sorted(pred_top, key=lambda m: -pred_support[m])],
        }
    return per_n


# --------------------------------------------------------------------------- #
# Movement profiles (predicted-derived vs truth-derived) -> ARI
# --------------------------------------------------------------------------- #
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
                "predicted_stroke_type": str(row.predicted_stroke_type),
                "true_stroke_type": side_agnostic(str(row.true_label_name)),
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


def build_profiles(events: pd.DataFrame, stroke_column: str) -> pd.DataFrame:
    profiles = []
    net_attacks = {"net shot", "cross-court net shot", "push", "rush", "return net"}
    lift_clear = {"lob", "clear"}
    for key, group in events.groupby(["video_id", "set_id", "player"]):
        displacements = group.dropna(subset=["interstroke_displacement"])
        smash = group[group[stroke_column] == "smash"]
        profiles.append(
            {
                "video_id": key[0],
                "set_id": key[1],
                "player": key[2],
                "n_events": len(group),
                "n_displacements": len(displacements),
                "mean_forward_position": float(group["forward_position"].mean()),
                "lateral_position_std": float(group["lateral_position"].std(ddof=0)),
                "forward_position_std": float(group["forward_position"].std(ddof=0)),
                "mean_interstroke_displacement": float(displacements["interstroke_displacement"].mean()),
                "rightward_displacement_rate": float((displacements["delta_lateral"] > 0.05).mean()),
                "netward_displacement_rate": float((displacements["delta_forward"] > 0.05).mean()),
                "right_baselineward_displacement_rate": float(
                    ((displacements["delta_lateral"] > 0.05) & (displacements["delta_forward"] < -0.05)).mean()
                ),
                "smash_rate": float((group[stroke_column] == "smash").mean()),
                "net_attack_rate": float(group[stroke_column].isin(net_attacks).mean()),
                "lift_clear_rate": float(group[stroke_column].isin(lift_clear).mean()),
                "smash_after_large_displacement_rate": float((smash["interstroke_displacement"] > 0.20).mean())
                if len(smash)
                else 0.0,
                "rear_court_smash_rate": float((smash["forward_position"] < 0.5).mean()) if len(smash) else 0.0,
            }
        )
    return pd.DataFrame(profiles)


def cluster_labels(profiles: pd.DataFrame, k: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(profiles[PROFILE_FEATURES])
    return KMeans(n_clusters=k, n_init=50, random_state=0).fit_predict(scaled), scaled


def clustering_gap(events: pd.DataFrame, k: int) -> dict:
    all_pred = build_profiles(events, "predicted_stroke_type")
    keep = all_pred["n_displacements"] >= MIN_PROFILE_DISPLACEMENTS
    pred_profiles = all_pred[keep].reset_index(drop=True)
    # Truth-derived profiles share the same units (position filter is label-independent).
    true_profiles = build_profiles(events, "true_stroke_type")
    true_profiles = true_profiles[keep.values].reset_index(drop=True)

    pred_labels, pred_scaled = cluster_labels(pred_profiles, k)
    true_labels, true_scaled = cluster_labels(true_profiles, k)
    ari = float(adjusted_rand_score(true_labels, pred_labels))

    # Per-feature fidelity: how well does each predicted-derived profile feature
    # track its truth-derived value across the 16 units? Position features are
    # label-independent (r=1 by construction); the stroke-rate features are
    # where classifier error enters, so we report those explicitly.
    stroke_features = [
        "smash_rate",
        "net_attack_rate",
        "lift_clear_rate",
        "smash_after_large_displacement_rate",
        "rear_court_smash_rate",
    ]
    feature_fidelity = {}
    for feature in stroke_features:
        true_values = true_profiles[feature].to_numpy()
        pred_values = pred_profiles[feature].to_numpy()
        pearson = float(np.corrcoef(true_values, pred_values)[0, 1]) if np.std(pred_values) > 0 else float("nan")
        feature_fidelity[feature] = {
            "pearson": pearson,
            "mean_abs_error": float(np.mean(np.abs(true_values - pred_values))),
        }

    # Random-labeling baseline for ARI (should be ~0).
    rng = np.random.default_rng(0)
    random_ari = []
    for _ in range(1000):
        random_ari.append(adjusted_rand_score(true_labels, rng.integers(0, k, size=len(true_labels))))
    contingency = pd.crosstab(pd.Series(true_labels, name="truth"), pd.Series(pred_labels, name="pred"))
    return {
        "n_units": int(len(pred_profiles)),
        "k": k,
        "ari_pred_vs_true": ari,
        "random_ari_mean": float(np.mean(random_ari)),
        "random_ari_p95": float(np.percentile(random_ari, 95)),
        "stroke_feature_fidelity": feature_fidelity,
        "stroke_feature_fidelity_mean_pearson": float(
            np.nanmean([v["pearson"] for v in feature_fidelity.values()])
        ),
        "pred_silhouette": float(silhouette_score(pred_scaled, pred_labels)) if len(set(pred_labels)) > 1 else float("nan"),
        "true_silhouette": float(silhouette_score(true_scaled, true_labels)) if len(set(true_labels)) > 1 else float("nan"),
        "contingency": contingency.to_dict(),
        "pred_labels": pred_labels.tolist(),
        "true_labels": true_labels.tolist(),
    }, pred_profiles.assign(pred_cluster=pred_labels, true_cluster=true_labels)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = clean_predictions(pd.read_csv(INTEGRATION / "phase10_structured_strokes.csv"))
    positions = np.load(INTEGRATION / "inputs" / "pos.npy")

    summary: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_clean_events": int(len(frame)),
        "n_rallies": int(frame.groupby(["video_id", "rally_id"]).ngroups),
    }

    # ---- 1. Composition --------------------------------------------------- #
    summary["composition"] = {
        "side_agnostic_no_none": composition_gap(frame, "true_label_name", "predicted_label_name", drop_none=True),
        "side_agnostic_with_none": composition_gap(frame, "true_label_name", "predicted_label_name", drop_none=False),
        "side_aware_with_none": {
            **_side_aware_composition(frame),
        },
    }

    # ---- 2. Transitions + baselines -------------------------------------- #
    states25 = sorted(set(frame["true_label_name"]) | set(frame["predicted_label_name"]))
    true_seq = labeled_sequences(frame, "true_label_name")
    pred_seq = labeled_sequences(frame, "predicted_label_name")
    true_counts = transition_counts(true_seq, states25)
    pred_counts = transition_counts(pred_seq, states25)
    true_T = row_normalize(true_counts)
    pred_T = row_normalize(pred_counts)
    unigram_T = unigram_matrix(true_counts)

    transitions = {
        "n_states": len(states25),
        "ground_truth": {"frobenius": 0.0, "mean_row_kl": 0.0},
        "current_pipeline": transition_gap(true_T, pred_T),
        "unigram": transition_gap(true_T, unigram_T),
        "uniform_random": random_matrix_gap(true_T, len(states25), RANDOM_SEEDS),
    }
    # Earlier pipeline, two ways:
    #  * earlier_pipeline          -- earlier pred vs its OWN 2,986-event ground
    #                                 truth (self-consistent, but a different
    #                                 reference matrix than the current set);
    #  * earlier_pipeline_common   -- earlier pred re-scored against the SAME
    #                                 3,178-event ground-truth reference used for
    #                                 every other row, so the table is reference-
    #                                 consistent (the residual difference is the
    #                                 candidate event sample, not the reference).
    transitions["earlier_pipeline"] = _earlier_pipeline_gap()
    transitions["earlier_pipeline_common_ref"] = _earlier_pipeline_common_ref(true_T, states25)
    summary["transitions"] = transitions

    # ---- 3. Motifs -------------------------------------------------------- #
    true_motif_seq = stroke_sequences(frame, "true_label_name", drop_none=True)
    pred_motif_seq = stroke_sequences(frame, "predicted_label_name", drop_none=True)
    summary["motifs"] = motif_gap(true_motif_seq, pred_motif_seq, TOP_K)

    # ---- 4. Movement correlations (with significance) --------------------- #
    corr_path = ROOT / "project" / "outputs" / "movement_strategy" / "movement_strategy_correlations.csv"
    if corr_path.exists():
        summary["movement_correlations"] = _movement_correlation_stats(corr_path)

    # ---- 5. Clustering ARI ------------------------------------------------ #
    events = player_centric_events(frame, positions)
    cluster_summary, cluster_table = clustering_gap(events, k=3)
    summary["clustering"] = cluster_summary
    cluster_table.to_csv(OUTPUT / "movement_cluster_agreement.csv", index=False)
    events.to_csv(OUTPUT / "movement_events_dual_labels.csv", index=False)

    (OUTPUT / "mining_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _side_aware_composition(frame: pd.DataFrame) -> dict:
    vocabulary = sorted(set(frame["true_label_name"]) | set(frame["predicted_label_name"]))
    p = distribution(list(frame["true_label_name"]), vocabulary)
    q = distribution(list(frame["predicted_label_name"]), vocabulary)
    uniform = np.ones_like(p)
    return {
        "vocabulary_size": len(vocabulary),
        "kl_true_pred": kl_divergence(p, q),
        "kl_true_uniform": kl_divergence(p, uniform),
        "chi_square_distance": chi_square_distance(p, q),
        "total_variation": total_variation(p, q),
    }


def _earlier_pipeline_gap() -> dict:
    gt_path = OLD_BASELINE / "transition_matrix_gt25_labeled.csv"
    pred_path = OLD_BASELINE / "transition_matrix_pred25_labeled.csv"
    if gt_path.exists() and pred_path.exists():
        gt = pd.read_csv(gt_path, index_col=0)
        pr = pd.read_csv(pred_path, index_col=0)
        common = sorted(set(gt.index) & set(pr.index))
        gap = transition_gap(gt.loc[common, common].to_numpy(), pr.loc[common, common].to_numpy())
        gap["recomputed_from"] = "baseline_old_phase09 matrices"
        gap["reference"] = "earlier 2,986-event ground truth (self)"
        gap["n_events"] = 2986
        return gap
    stored = json.loads((OLD_BASELINE / "transition_comparison.json").read_text())
    return {"frobenius": stored["frobenius_gt_minus_pred"], "mean_row_kl": stored["mean_row_kl_gt_pred"]}


def _movement_correlation_stats(corr_path: Path) -> dict:
    """Report each highlighted movement correlation with its sample size,
    two-sided p-value, Fisher 95% CI, and Benjamini-Hochberg q-value computed
    over the full family of C(12,2)=66 pairwise correlations (we select five
    from this matrix, so the correction is over the whole matrix). With only
    n=16 profiles, |r| must reach ~0.50 to clear raw significance."""
    from scipy import stats

    corr = pd.read_csv(corr_path, index_col=0)
    features = list(corr.index)
    n = N_MOVEMENT_PROFILES

    def stat(r: float) -> dict:
        t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
        p = float(2 * stats.t.sf(abs(t), n - 2))
        z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), stats.norm.ppf(0.975)
        return {"r": float(r), "p": p, "ci95": [float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))]}

    # Full family of 66 unique pairs -> BH q-values.
    pairs, pvals = [], []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append((features[i], features[j]))
            pvals.append(stat(corr.iloc[i, j])["p"])
    order = np.argsort(pvals)
    m = len(pvals)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, pvals[idx] * m / (rank + 1))
        q[idx] = running
    qmap = {pair: float(q[k]) for k, pair in enumerate(pairs)}

    def lookup_q(a: str, b: str) -> float:
        return qmap.get((a, b), qmap.get((b, a), float("nan")))

    reported = {
        "netward_vs_smash": ("netward_displacement_rate", "smash_rate"),
        "baselineward_vs_smash": ("right_baselineward_displacement_rate", "smash_rate"),
        "lateral_std_vs_netward": ("lateral_position_std", "netward_displacement_rate"),
        "forward_std_vs_net_attack": ("forward_position_std", "net_attack_rate"),
        "mean_forward_vs_rear_smash": ("mean_forward_position", "rear_court_smash_rate"),
    }
    out = {"n_profiles": n, "family_size": m, "fdr_method": "benjamini_hochberg"}
    for name, (a, b) in reported.items():
        s = stat(corr.loc[a, b])
        s["q_bh_full_matrix"] = lookup_q(a, b)
        s["raw_significant"] = s["p"] < 0.05
        s["fdr_significant"] = s["q_bh_full_matrix"] < 0.05
        out[name] = s
    return out


def _align_to_states(df: pd.DataFrame, states: list[str]) -> np.ndarray:
    """Embed a labeled transition matrix into the current state ordering."""
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    for row_label in df.index:
        for col_label in df.columns:
            if row_label in index and col_label in index:
                matrix[index[row_label], index[col_label]] = float(df.loc[row_label, col_label])
    return matrix


def _earlier_pipeline_common_ref(true_T: np.ndarray, states: list[str]) -> dict:
    """Re-score the earlier pipeline's predicted matrix against the *current*
    3,178-event ground-truth reference, so it sits on the same reference as the
    unigram / random / current rows. The earlier predicted matrix is still
    estimated from its 2,986-event run; only the reference is now shared."""
    pred_path = OLD_BASELINE / "transition_matrix_pred25_labeled.csv"
    if not pred_path.exists():
        return {}
    pred = _align_to_states(pd.read_csv(pred_path, index_col=0), states)
    gap = transition_gap(true_T, pred)
    gap["reference"] = "shared 3,178-event ground truth"
    gap["candidate_n_events"] = 2986
    return gap


if __name__ == "__main__":
    main()
