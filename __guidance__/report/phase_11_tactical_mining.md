# Phase 11: Tactical Mining on Tracked Phase 10

## Status

Implemented and executed using:

```text
project/outputs/integration/bst_tracked_phase09_41_44
```

Generated mining bundle:

```text
project/outputs/mining/bst_tracked_phase09_41_44
```

The implementation follows the four algorithms selected in
[data_mining.md](../data_mining.md):

1. Markov transition modeling.
2. Frequent contiguous sequential patterns.
3. Player/style clustering.
4. Spatial density and heatmaps.

## Implementation

```text
project/tools/phase11_tactical_mining.py
```

Run command:

```bash
python3 project/tools/phase11_tactical_mining.py \
  --integration-root project/outputs/integration/bst_tracked_phase09_41_44 \
  --output-root project/outputs/mining/bst_tracked_phase09_41_44 \
  --baseline-mining-root project/outputs/mining/baseline_old_phase09 \
  --confidence-threshold 0.8
```

Phase 11 excludes the 14 shared-feature alias rows identified by Phase 10.

Three prediction tracks are kept separate:

- **hard predicted:** every clean top-1 prediction;
- **soft predicted:** probability-weighted transitions;
- **high-confidence predicted:** contiguous prediction segments where every
  retained stroke has confidence `>=0.8`.

High-confidence filtering never joins strokes across removed low-confidence
events.

## Data Scale

| Source | Strokes | Rallies | Mean rally length |
|---|---:|---:|---:|
| Ground truth, all 44 videos | 36,482 | 3,683 | 9.91 |
| Clean tracked predictions, videos 41-44 | 3,178 | 343 | 9.27 |
| High-confidence tracked predictions | 1,131 | fragmented segments | - |

Tracked prediction `none` rate is **6.70%**, down from **56.43%** in the old
integrated baseline.

High-confidence coverage is **35.59%** of clean predicted strokes. Phase 10
test accuracy within this confidence range is **92.80%**.

## 1. Markov Transition Modeling

Transition error against exact 25-class labels:

| Predicted transition method | Frobenius error | Mean row KL |
|---|---:|---:|
| Old baseline hard predictions | 3.652 | 11.723 |
| New tracked hard predictions | **1.232** | **0.511** |
| New tracked soft probabilities | 1.617 | 0.623 |
| New high-confidence contiguous segments | **1.211** | 1.372 |

The tracked hard transition matrix is the best general-purpose estimator.
Probability-weighted transitions are worse because the model's residual
probability mass blurs strong tactical transitions. High-confidence transitions
have the lowest Frobenius error but higher KL because only 35.59% of strokes
remain, leaving rare transition rows sparse.

Strong transitions recovered from tracked predictions:

| Current | Most likely next | Probability |
|---|---|---:|
| Top smash | Bottom return net | 0.667 |
| Bottom net shot | Top lob | 0.622 |
| Bottom long service | Top smash | 0.534 |
| Bottom smash | Top return net | 0.530 |
| Bottom lob | Top smash | 0.471 |
| Bottom clear | Top smash | 0.448 |
| Top drive | Bottom return net | 0.448 |

These preserve the expected attack/defense grammar. The largest remaining
transition errors concentrate around net-shot versus return-net distinctions,
drop shots, rushes, and long services.

## 2. Sequential Pattern Mining

Ground-truth motifs remain the authoritative tactical findings:

| Pattern | Rally support |
|---|---:|
| net shot -> lob | 48.76% |
| smash -> defensive net | 30.30% |
| net shot -> lob -> smash | 18.11% |
| lob -> smash -> defensive net | 17.05% |
| net shot -> lob -> smash -> defensive net | 11.16% |

The tracked predicted stream now recovers the same side-aware structure:

| Predicted pattern | Rally support |
|---|---:|
| Top smash -> Bottom return net | 28.86% |
| Bottom lob -> Top smash | 18.08% |
| Bottom clear -> Top smash | 17.49% |
| Bottom smash -> Top return net | 15.45% |
| Top net shot -> Bottom lob -> Top smash | 7.58% |
| Top net shot -> Bottom lob -> Top smash -> Bottom return net | 4.66% |

Unlike the old baseline, the dominant predicted motif is no longer
`none -> none`. The deployable predicted stream now contains recognizable
tactical sequences.

High-confidence filtering retains only six patterns above the 2% support
threshold. It is suitable for reliable examples, but not for broad pattern
discovery because rally fragmentation removes too many adjacent strokes.

### Ground-Truth Opening Tactics

| Opening | Support | Server win rate |
|---|---:|---:|
| short service -> net shot -> lob | 12.21% | 49.13% |
| short service -> net shot -> net shot | 8.58% | 49.82% |
| short service -> push -> clear | 3.64% | **37.50%** |
| short service -> lob -> smash | 2.15% | **53.52%** |
| short service -> lob -> tap smash | 2.12% | **60.00%** |

Outcome-weighted opening findings remain GT-only because the deployment
prediction path does not infer point winners.

## 3. Clustering

### Ground-Truth Player-Match Profiles

- Units: `88` `(match, player)` profiles.
- Best `k`: `2`.
- Silhouette: **0.152**, indicating weak separation.
- Bootstrap stability: mean adjusted Rand index **0.440**.
- Cluster sizes: `20` and `68`.

The two groups are gradual stroke-mix tendencies, not distinct player
archetypes:

- a smaller backcourt-oriented tendency with more clear (`+4.9` percentage
  points over global), smash (`+1.8`), and long service (`+1.5`);
- a larger front/midcourt tendency with slightly more net shot (`+1.4`), lob
  (`+1.1`), and short service (`+0.7`).

The low silhouette and moderate-to-low bootstrap stability mean many
player-match profiles can move between these groups under resampling. The
result does not support naming stable elite-player archetypes.

### Predicted Video-Side Profiles

- Units: `8` `(video, Top/Bottom side)` profiles.
- Selected `k`: `2`.
- Silhouette: **0.200**.
- Bootstrap stability: mean adjusted Rand index **0.505**.
- Cluster sizes: `4` and `4`.

The raw silhouette maximum is `k=3` at `0.225`, but that solution contains a
singleton cluster. Phase 11 therefore selects the balanced `k=2` solution
instead of interpreting one outlying video-side as a style.

The selected groups are:

- videos `42-43`, both sides: more clear (`+5.2` percentage points), long
  service (`+3.4`), and drop (`+2.1`);
- videos `41` and `44`, both sides: more lob (`+3.9`), net shot (`+3.9`), and
  short service (`+1.4`).

Because both sides from each video stay together, the split is primarily a
**video/match-domain or perception-quality grouping**, not evidence of
individual player styles. It is useful as a diagnostic signal that videos
`42-43` produce a different predicted stroke mix; it must not be generalized
as a player taxonomy.

Clustering diagnostics are exported to:

```text
cluster_diagnostics_gt.csv
cluster_diagnostics_pred.csv
player_profiles_gt.csv
player_profiles_pred.csv
clusters_players_gt.csv
clusters_players_pred.csv
```

The diagnostics sweep `k=2..8` where feasible and report silhouette,
bootstrap adjusted Rand stability, and minimum cluster size. Assignment files
include PCA coordinates and distance to the assigned centroid for outlier
inspection.

## 4. Spatial Density

Generated spatial visualizations:

- GT landing density over `35,297` annotated points.
- GT landing density by smash, drop, clear, and net shot.
- GT player-position density over `35,314` points.
- Predicted event-frame Top occupancy over `1,965` valid points.
- Predicted event-frame Bottom occupancy over `3,146` valid points.
- Approximate predicted shuttle endpoints for all `3,178` clean clips.

The predicted player occupancy visualization validates stable ordering:

```text
Top player -> far court half
Bottom player -> near court half
```

Top occupancy has substantially fewer valid samples, matching the Phase 09
far-side pose/position weakness.

The predicted shuttle endpoint plot uses normalized broadcast-image
coordinates. It is an approximate final visible point inside each stroke
window, not annotated landing ground truth.

## System Error Propagation

The old baseline demonstrated that severe perception failure destroyed tactical
mining:

```text
old transition Frobenius error: 3.652
old transition mean row KL: 11.723
old predicted none rate: 56.43%
```

Tracked Phase 09 and BST-compatible temporal windows changed the result:

```text
new transition Frobenius error: 1.232
new transition mean row KL: 0.511
new predicted none rate: 6.70%
```

Relative reduction:

- Frobenius error: **66.3% lower**.
- Mean row KL: **95.6% lower**.

The pipeline now preserves enough sequence structure for predicted tactical
mining to be informative, but findings should still be accompanied by Phase 10
accuracy and confidence.

## Visualizations

Generated under:

```text
project/outputs/mining/bst_tracked_phase09_41_44/figures/
```

Important figures:

```text
transition_heatmap_gt25_labeled.png
transition_heatmap_pred_hard.png
transition_heatmap_pred_soft.png
transition_heatmap_pred_high_conf.png
transition_error_heatmap.png
accuracy_by_confidence.png
accuracy_vs_top_pose.png
player_event_occupancy_pred.png
shuttle_endpoint_pred_approximate.png
landing_heatmap_gt_all.png
landing_heatmap_gt_by_stroke.png
player_position_heatmap_gt.png
player_clusters_gt.png
player_clusters_pred.png
```

## Validation

- Phase 11 script compiles and completed successfully.
- Clustering diagnostics use `100` deterministic bootstrap repeats.
- The focused Phase 11 clustering test and all `17` project tests pass.
- `3,192` prediction, probability, and position rows align.
- `14` shared-feature aliases are excluded.
- Every populated transition row sums to one within floating-point tolerance.
- GT and predicted findings remain separate.
- High-confidence filtering preserves original rally adjacency.
- Visualizations were generated and manually inspected.

## Limitations

- Primary tracked Phase 10 test accuracy is `66.40%`; sequence mining still
  compounds residual errors.
- Far-side Top-player pose quality remains the largest upstream limitation.
- Predicted player-style clustering has only eight video-side units.
- Predicted shuttle endpoints are approximations, not landings.
- GT image-pixel spatial plots and predicted normalized-coordinate plots are
  not directly subtractable.

Detailed interpretation: [interpretation.md](../interpretation.md).
