# Data Mining Algorithms for the Badminton Analytics Output

This document maps concrete data-mining algorithms onto the output produced by the
end-to-end system described in [systems.md](systems.md), specifically the Phase 10 /
Phase 11 structured stroke tables. It is the menu for **Phase 11: Tactical Mining**.

---

## 0. What the output actually is (the data contract for mining)

### 0.1 Row = one stroke; rally = an ordered sequence of rows

The integrated output
(`project/outputs/integration/<run>/phase10_structured_strokes.csv`, 6,282 rows in the
baseline) is a **per-stroke event log**. One row is one detected stroke event. To
recover a rally as a *sequence*, sort rows by `(video_id, rally_id, event_rank)` (or
`ball_round_id` when the Phase 02 join exists); a match is the set of its rallies. Every
mining method below consumes this ordering.

### 0.2 The hard constraint: on the prediction path we only get *labels*, not tactics

This is the central design fact for Phase 11. There are **two different tables** and
they are not interchangeable:

- **Phase 02 ground-truth table** (`shuttleset_ground_truth_strokes.csv`) — produced by
  a **human annotator** in the ShuttleSet dataset. It is the *only* place the rich
  tactical fields exist: `hit_x/y`, `hit_area`, `hit_height`, `landing_x/y`,
  `landing_area`, `landing_height`, `player_location_*`, `opponent_location_*`,
  `aroundhead`, `backhand`, and every **outcome** field (`win_reason`, `lose_reason`,
  `getpoint_player`, `score_state`, `roundscore_A/B`, `flaw`).
- **Phase 10 predicted/integrated table** (`phase10_structured_strokes.csv`) — produced
  by the **perception + classification pipeline** of Phases 05–10. **None of those
  tactical/outcome columns are in it.** No model in the pipeline detects who won a
  point, why a rally was lost, or the annotator's hit/landing area code. So for a true
  unseen broadcast video (Track B), those columns simply do not exist — there is no
  annotator in the loop.

**Why we mine the predicted labels, not the ground truth:** the whole point of Track B
is to analyze *new, unseen video* where no ShuttleSet annotation will ever exist. If
Phase 11 leaned on `win_reason` / `hit_area` / `landing_area`, it would only ever work
on the 44 already-annotated videos and would prove nothing about the end-to-end system.
So the deployable mining set must be built from **what the pipeline itself emits**.

### 0.3 What the pipeline actually emits (the mineable signal for unseen video)

| Tier | Columns / artifacts | Where it comes from |
| --- | --- | --- |
| **Model output (always available)** | `predicted_label`, `predicted_label_name`, `predicted_stroke_type`, `predicted_player_side` (Top/Bottom) | BST classifier, Phase 04/10 |
| **Model uncertainty** | `confidence`, `top2_label_name`/`top2_score`, `top3_*`, full softmax matrix `phase10_probabilities.npy` `(N,25)`, `phase10_logits.npy` | BST head |
| **Sequence keys / order** | `video_id`, `rally_id`, `event_rank`, `clip_id` | Phase 05 rallies + Phase 07 windows |
| **Time** | `event_frame_original`, `window_start_original`, `window_end_original` (`time` only if fps known) | Phase 07 |
| **Derivable geometry** (needs a small extra step, not in the CSV as tactics) | shuttle trajectory `shuttle.npy (T,2)`, player court positions `pos.npy (T,2,2)`, pose `joints.npy (T,2,17,2)` | Phase 06 / 08 / 09 |
| **Quality / provenance** | `zero_pose_rate`, `shuttle_visible_rate`, `quality_group`, `feature_source` | Phase 09/10 |
| **GT join (ShuttleSet only, for evaluation)** | `stroke_type_ground_truth`, `true_label`, `player`, `server`, `reference_split`, `correct` — **populated only when `has_phase02_label == True`** | join to Phase 02 |

Two consequences:

1. **The deployable rally object is a sequence of `(predicted_stroke_type,
   predicted_player_side)` pairs** plus optional geometry derived from `pos.npy` /
   `shuttle.npy`. That is the substrate for all four algorithms kept below: §1
   sequential patterns, §2 Markov transitions, §3 clustering, and §4 heatmaps (from
   *derived* positions on the predicted path, *direct* coordinates on the GT path).
2. **Outcome-based mining is not available on the prediction path.** Anything keyed on
   `getpoint_player` / `win_reason` (win-probability, winners-vs-losers contrast,
   point-outcome rules) requires fields the pipeline does not produce. Where one of the
   four kept algorithms has an outcome-flavoured variant (e.g. absorbing-state Markov,
   win-rate-weighted patterns), that variant runs on the **Phase 02 GT table only**; the
   core variant still runs on the predicted table.

### 0.4 Two-track reporting rule + current quality caveat

Run every *predicted-OK* algorithm on the **predicted** table; run *GT-only* algorithms
on the **Phase 02** table. They answer different questions and must never be silently
merged. On predicted ShuttleSet data, keep the `1-34 / 35-39 / 40-44` split groups
separate and report model coverage + accuracy context beside any tactical claim
(systems.md Phase 11).

> **Quality caveat.** The active tracked integrated run reaches **66.40%** clean
> reference-test top-1 accuracy, while the old baseline reached only 9.23%.
> Prediction-based mining now preserves substantial tactical structure, but residual
> errors still compound across sequences. Report GT and predicted findings separately,
> exclude shared-feature aliases, compare hard/soft/high-confidence variants, and
> stratify results by confidence and Top-player pose quality.

---

We keep **four** algorithms — the ones that are both tactically meaningful *and*
feasible on what the pipeline actually emits (§0.3). They are ordered by build effort
(cheapest first). Each entry ends with a concrete **"Will we implement it?"** decision
that fixes the scope we will actually code in Phase 11.

A trivial precursor common to all four — **stroke frequency / distribution** (counts of
`predicted_stroke_type` per `predicted_player_side`, per video, rally-length histogram)
— is *not* a separate algorithm; it is a one-pass `groupby` we compute first and reuse
as the feature substrate for §2 and §3.

---

## 1. Markov Chains & Transition Modeling — *"what tends to follow what?"*  `[predicted-OK; absorbing variant GT-only]`

**What it answers.** Given the current stroke, what is the player likely to play next —
the rally's flow as a `P(next stroke | current stroke)` matrix. Directly fulfils the
`transition_matrix.csv` Phase 11 deliverable.

**Track & inputs.** Predicted table: ordered `(predicted_player_side, predicted_stroke_type)`
tokens per `(video_id, rally_id)`, sorted by `event_rank`. GT table: same from
`stroke_type_ground_truth`. 25 side-aware classes → a 25×25 matrix.

**Method & library.** Pure `numpy`/`pandas` counting + row-normalization; render with
`seaborn.heatmap`. No external solver needed.

**Implementation plan.**
1. Load the table; build per-rally label-index sequences (map the 25 class names → 0–24).
2. Accumulate adjacent pairs `(s_t, s_{t+1})` into a 25×25 count matrix `C`; row-normalize
   to `P`.
3. **Soft / uncertainty-aware variant (mitigates the 9% accuracy):** instead of the hard
   top-1, accumulate `Σ_t outer(p_t, p_{t+1})` from `phase10_probabilities.npy` so
   low-confidence strokes contribute fractionally rather than as a wrong hard label.
4. Build the same matrix on the **GT table**, then report `||P_pred − P_gt||` (Frobenius
   + per-row KL) — this *quantifies how perception error distorts rally flow*, a headline
   error-propagation result.
5. (Optional, GT-only) second-order n-gram transitions for 2-stroke memory.

**Output.** `project/outputs/mining/transition_matrix.csv` (+ `_soft`, `_gt`), and a
`figures/transition_heatmap.png`.

**Will we implement it?** **Yes — first and definitely.** It is the cheapest deliverable,
it is explicitly required by systems.md, and we run all three variants (predicted hard,
predicted soft, GT) because the GT-vs-predicted comparison is itself a reportable finding.
We **skip** the absorbing-Markov / HMM / MDP extensions: they need outcome labels or add
training complexity with little payoff at current accuracy.

## 2. Sequential Pattern Mining — *"what stroke sequences recur?"*  `[predicted-OK; win-rate weighting GT-only]`

**What it answers.** The recurring multi-stroke *motifs* a transition matrix can't see —
e.g. `serve(short) → net → lift → smash` — and the classic serve + third-shot opening
patterns. This is the "data-mining headline" result for the report.

**Track & inputs.** Same per-rally ordered token lists as §1. Each rally is one sequence;
each item is the `side_stroke` token. GT table gives cleaner patterns; predicted table
shows what survives the noise.

**Method & library.** **PrefixSpan** for frequent + **closed** sequential patterns
(`pip install prefixspan`), with a `min_support` threshold; fall back to `mlxtend`
itemset mining for a bag-of-strokes relaxation if needed.

**Implementation plan.**
1. Build `sequences = [[tok, tok, ...], ...]`, one list per rally.
2. Run PrefixSpan at a support floor (e.g. ≥1% of rallies); keep **closed** patterns to
   drop redundant sub-sequences.
3. Compute, per pattern: support count, support fraction, mean length.
4. **Opening-tactics slice:** restrict to the first 3 strokes (`event_rank ≤ 3`) and mine
   serve/return/third-shot combos separately.
5. (GT-only) join `getpoint_player` to tag each pattern with the rally win-rate →
   "patterns that tend to win the point"; on the predicted table this column is omitted.
6. Run on GT first (trust), then predicted with the same thresholds, and note the support
   drop as a noise indicator.

**Output.** `project/outputs/mining/patterns.csv` (pattern, support, length, [win_rate]).

**Will we implement it?** **Yes — the core mining result.** Scope: frequent + closed
patterns + the opening-tactics slice, on both tables. We **skip** discriminative /
contrast-set sequence mining (winners-vs-losers) on the predicted path — it needs outcome
labels — and only do its lightweight win-rate-weighted form on the GT table.

## 3. Clustering — *"what natural player/rally groupings exist?"*  `[player-style & rally-seq predicted-OK; shot-placement GT/derived]`

**What it answers.** Whether players fall into **style archetypes** (aggressive smasher
vs patient rallyer) and whether rallies fall into recurring **templates**.

**Track & inputs.** *Player-style (predicted-OK):* one feature vector per `player` /
per `predicted_player_side` = normalized 25-class stroke histogram (from the §0 precursor)
+ mean rally length + serve rate. Aggregation over many strokes makes this **robust to
per-stroke misclassification**, which is exactly why it survives the 9% accuracy.
*Rally-sequence (predicted-OK):* per-rally token sequences from §1.

**Method & library.**
- Player-style: `sklearn` `StandardScaler` → `KMeans` / Ward `AgglomerativeClustering`;
  pick `k` by silhouette / elbow.
- Rally-sequence (optional): Levenshtein/DTW distance matrix → `KMedoids`
  (`scikit-learn-extra`).
- Visualization: `PCA`/`UMAP` to 2D scatter (folds the old "embedding" step in here).

**Implementation plan.**
1. Build the per-player feature matrix; standardize.
2. Sweep `k=2..8`, choose by silhouette; fit final KMeans; name each cluster from its
   dominant strokes.
3. 2D PCA/UMAP scatter coloured by cluster for the report figure.
4. (Optional stretch) rally edit-distance → k-medoids → a handful of rally templates.

**Output.** `project/outputs/mining/clusters_players.csv` + `figures/player_clusters.png`
(and optional `clusters_rallies.csv`).

**Will we implement it?** **Yes for player-style clustering** — it is robust to label
noise and gives an interpretable result. **Rally-sequence clustering is a stretch goal**
(distance-matrix cost + noise). We **skip** pose/motion clustering from `joints.npy` and
spatial shot-zone clustering (§4 of the old list) as out of scope / heavy.

## 4. Heatmaps & Spatial Density — *"where do shots and players sit on court?"*  `[GT-direct; predicted via derived pos.npy / shuttle.npy]`

**What it answers.** The spatial story: where players stand and where the shuttle goes.
The main *visual* deliverable.

**Track & inputs.**
- **GT table (direct, trustworthy):** `landing_x/y`, `hit_x/y`, `player_location_x/y`,
  `opponent_location_x/y` — already court-normalized.
- **Predicted path (derived):** player court position from `pos.npy (T,2,2)` at the event
  frame → occupancy heatmap; approximate shuttle landing = last visible point of
  `shuttle.npy` in the window. Shuttle landing is an *approximation*, flagged as such.

**Method & library.** `numpy.histogram2d` + `scipy.stats.gaussian_kde` /
`seaborn.kdeplot`, drawn over a normalized badminton-court template in `matplotlib`.

**Implementation plan.**
1. GT: KDE of `landing_x/y` per player and per stroke type (small multiples).
2. Predicted: read `pos.npy` per clip, take the two players' court coords at the event
   frame → 2D player-occupancy heatmap per video.
3. **Difference heatmap GT − predicted** for the 44 ShuttleSet videos → visualizes spatial
   error propagation (pairs with §1 step 4).
4. Render to `figures/`.

**Will we implement it?** **Yes, primarily on the GT table** where coordinates are exact —
landing and player-position heatmaps per player/stroke. On the **predicted** path we will
implement the **player-occupancy** heatmap from `pos.npy` (clean, available) and include
shuttle-landing only as an explicitly-approximate overlay. We **skip** score-state and
winner/loser heatmaps (outcome-dependent, GT-only and lower priority).

---

## Mapping to Phase 11 deliverables & build order

| systems.md Phase 11 output | Kept algorithm |
| --- | --- |
| `transition_matrix.csv` | §1 Markov |
| `patterns.csv` | §2 Sequential patterns |
| `player_profiles.csv` | §0 precursor + §3 Clustering |
| `stroke_distribution.csv` | §0 precursor (groupby) |
| `figures/` heatmaps | §4 Heatmaps |

**Build order (cheapest, most-certain first):**
1. §0 stroke-distribution precursor + **§1 Markov** on GT, then predicted (hard + soft).
2. **§2 sequential patterns** on GT, then predicted; add the opening-tactics slice.
3. **§4 heatmaps** on GT (direct), then predicted player-occupancy + GT−pred difference.
4. **§3 player-style clustering**; rally-sequence clustering only if time allows.

Every step runs on the **GT table first** (real tactical findings) and the **predicted
table second**, always reporting the GT-vs-predicted gap as the error-propagation result.
