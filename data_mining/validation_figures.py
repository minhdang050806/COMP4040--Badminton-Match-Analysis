#!/usr/bin/env python3
"""Render the mining-validation figures used in the final report.

Produces, in project/outputs/mining_validation/figures/:
  * transition_baselines.png      -- Frobenius / row-KL for 5 transition models
  * composition_true_vs_pred.png  -- true vs predicted stroke-type distribution
  * motif_support_scatter.png     -- true vs predicted motif support (n=2,3)
  * cluster_feature_fidelity.png  -- stroke-feature fidelity + ARI agreement
  * mining_validation_overview.png-- 2x2 panel combining all four views
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase12_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_mining.validation as v

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

ROOT = v.ROOT
OUT = ROOT / "project" / "outputs" / "mining_validation"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY = "#183252"
STEEL = "#3a5678"
CORAL = "#d66853"
RED = "#8c2d2d"
GREY = "#9aa6b2"


def load_summary() -> dict:
    return json.loads((OUT / "mining_validation_summary.json").read_text())


def panel_transitions(ax_f, ax_k, summary) -> None:
    # Earlier pipeline is shown on the SHARED 3,178-event reference so all bars
    # are comparable (earlier_pipeline_common_ref), not its own 2,986-event GT.
    order = ["uniform_random", "unigram", "earlier_pipeline_common_ref", "current_pipeline", "ground_truth"]
    labels = ["Uniform\nrandom", "Unigram\n(memoryless)", "Earlier\n(none 56%)", "Current\n(none 7%)", "Ground\ntruth"]
    colors = [GREY, STEEL, RED, CORAL, NAVY]
    frob = [summary["transitions"][k]["frobenius"] for k in order]
    kl = [summary["transitions"][k]["mean_row_kl"] for k in order]

    bars = ax_f.bar(labels, frob, color=colors)
    ax_f.set_ylabel("Frobenius distance to GT")
    ax_f.set_title("Transition recovery: Frobenius $\\Vert T-\\hat T\\Vert_F$")
    ax_f.axhline(frob[-2], color=CORAL, ls="--", lw=1, alpha=0.6)
    for bar, value in zip(bars, frob):
        ax_f.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax_f.set_ylim(0, max(frob) * 1.18)

    bars = ax_k.bar(labels, kl, color=colors)
    ax_k.set_ylabel("Mean row KL to GT (log)")
    ax_k.set_yscale("log")
    ax_k.set_title("Transition recovery: mean row KL")
    for bar, value in zip(bars, kl):
        ax_k.text(bar.get_x() + bar.get_width() / 2, max(value, 1e-3) * 1.1, f"{value:.2f}", ha="center", va="bottom", fontsize=10)


def panel_composition(ax, summary) -> None:
    comp = summary["composition"]["side_agnostic_no_none"]
    true_dist = comp["true_distribution"]
    pred_dist = comp["pred_distribution"]
    tokens = sorted(true_dist, key=lambda t: -true_dist[t])
    x = np.arange(len(tokens))
    width = 0.4
    ax.bar(x - width / 2, [true_dist[t] for t in tokens], width, label="Ground truth", color=NAVY)
    ax.bar(x + width / 2, [pred_dist[t] for t in tokens], width, label="Predicted", color=CORAL)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=40, ha="right")
    ax.set_ylabel("Proportion of strokes")
    ax.set_title("Stroke composition: predicted vs true")
    ax.legend()
    ax.text(
        0.97,
        0.95,
        f"KL(true$\\Vert$pred)={comp['kl_true_pred']:.3f}\n"
        f"(uniform={comp['kl_true_uniform']:.3f})\n"
        f"$\\chi^2$={comp['chi_square_distance']:.3f}  TV={comp['total_variation']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#fff9ed", ec=GREY),
    )


def motif_pairs(frame, n):
    true_seq = v.stroke_sequences(frame, "true_label_name", drop_none=True)
    pred_seq = v.stroke_sequences(frame, "predicted_label_name", drop_none=True)
    ts = v.motif_supports(true_seq, n)
    ps = v.motif_supports(pred_seq, n)
    motifs = sorted({m for m, s in ts.items() if s >= v.MIN_SUPPORT} | {m for m, s in ps.items() if s >= v.MIN_SUPPORT})
    return np.array([ts.get(m, 0.0) for m in motifs]), np.array([ps.get(m, 0.0) for m in motifs])


def panel_motifs(ax, frame, summary) -> None:
    markers = {2: ("o", STEEL), 3: ("s", CORAL)}
    lim = 0
    for n, (marker, color) in markers.items():
        tv, pv = motif_pairs(frame, n)
        ax.scatter(tv, pv, s=45, marker=marker, color=color, alpha=0.75,
                   label=f"n={n}  $\\rho$={summary['motifs'][str(n)]['spearman_support']:.2f}, "
                         f"J@10={summary['motifs'][str(n)]['jaccard_top_k']:.2f}")
        lim = max(lim, tv.max(), pv.max())
    lim *= 1.08
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True motif rally support")
    ax.set_ylabel("Predicted motif rally support")
    ax.set_title("Motif support: predicted vs true")
    ax.legend(loc="lower right", fontsize=10)


def panel_clusters(ax, summary) -> None:
    cl = summary["clustering"]
    fidelity = cl["stroke_feature_fidelity"]
    names = list(fidelity.keys())
    short = [n.replace("_rate", "").replace("_", " ") for n in names]
    pearson = [fidelity[n]["pearson"] for n in names]
    bars = ax.barh(short, pearson, color=STEEL)
    for bar, value in zip(bars, pearson):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10)
    ax.axvline(cl["stroke_feature_fidelity_mean_pearson"], color=RED, ls="--", lw=1.2,
               label=f"mean r={cl['stroke_feature_fidelity_mean_pearson']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson r (truth- vs predicted-derived)")
    ax.set_title("Movement-profile feature fidelity")
    ax.legend(loc="lower right", fontsize=10)
    ax.text(
        0.03,
        0.04,
        f"cluster ARI={cl['ari_pred_vs_true']:.2f} (chance={cl['random_ari_mean']:.2f})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#fff9ed", ec=GREY),
    )


def main() -> None:
    summary = load_summary()
    frame = v.clean_predictions(pd.read_csv(v.INTEGRATION / "phase10_structured_strokes.csv"))

    # Standalone transition baselines.
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    panel_transitions(a, b, summary)
    fig.tight_layout()
    fig.savefig(FIG / "transition_baselines.png", dpi=170)
    plt.close(fig)

    # Standalone composition.
    fig, ax = plt.subplots(figsize=(8, 5))
    panel_composition(ax, summary)
    fig.tight_layout()
    fig.savefig(FIG / "composition_true_vs_pred.png", dpi=170)
    plt.close(fig)

    # Standalone motif scatter.
    fig, ax = plt.subplots(figsize=(7, 6))
    panel_motifs(ax, frame, summary)
    fig.tight_layout()
    fig.savefig(FIG / "motif_support_scatter.png", dpi=170)
    plt.close(fig)

    # Standalone cluster fidelity.
    fig, ax = plt.subplots(figsize=(8, 5))
    panel_clusters(ax, summary)
    fig.tight_layout()
    fig.savefig(FIG / "cluster_feature_fidelity.png", dpi=170)
    plt.close(fig)

    # Combined 2x2 overview (report headline validation figure).
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)
    ax_comp = fig.add_subplot(gs[0, 0])
    ax_motif = fig.add_subplot(gs[0, 1])
    ax_frob = fig.add_subplot(gs[1, 0])
    ax_clu = fig.add_subplot(gs[1, 1])
    panel_composition(ax_comp, summary)
    panel_motifs(ax_motif, frame, summary)

    # Single Frobenius panel for the overview (KL noted in caption).
    # Earlier pipeline is shown on the SHARED 3,178-event reference so all bars
    # are comparable (earlier_pipeline_common_ref), not its own 2,986-event GT.
    order = ["uniform_random", "unigram", "earlier_pipeline_common_ref", "current_pipeline", "ground_truth"]
    labels = ["Uniform\nrandom", "Unigram", "Earlier\n(none 56%)", "Current\n(none 7%)", "Ground\ntruth"]
    colors = [GREY, STEEL, RED, CORAL, NAVY]
    frob = [summary["transitions"][k]["frobenius"] for k in order]
    bars = ax_frob.bar(labels, frob, color=colors)
    for bar, value in zip(bars, frob):
        ax_frob.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax_frob.set_ylabel("Frobenius distance to GT")
    ax_frob.set_title("Transition recovery vs baselines")
    ax_frob.set_ylim(0, max(frob) * 1.18)
    panel_clusters(ax_clu, summary)

    for ax, tag in zip([ax_comp, ax_motif, ax_frob, ax_clu], ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(-0.08, 1.05, tag, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    fig.savefig(FIG / "mining_validation_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
