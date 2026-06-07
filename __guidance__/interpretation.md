# Phase 11 Interpretation

This interpretation uses the tracked integration mining output:

```text
project/outputs/mining/bst_tracked_phase09_41_44
```

Ground-truth findings describe badminton tactics. Predicted findings describe
what the deployable pipeline can now recover from video.

## 1. Badminton Singles Has an Attack-Defense Grammar

The ground-truth transitions and patterns show a consistent rally structure:

```text
net shot -> lob -> smash -> defensive net
```

This four-stroke motif appears in **11.16%** of rallies. The pair
`net shot -> lob` appears in **48.76%**, while `smash -> defensive net` appears
in **30.30%**.

The interpretation is tactical:

- net pressure frequently forces a lift;
- lifts create attacking opportunities;
- smashes usually force a defensive reply rather than immediately ending the
  rally;
- the next decisive contest often returns to the net.

The tracked predicted stream now recovers this same grammar in side-aware form:

```text
Top net shot -> Bottom lob -> Top smash -> Bottom return net
```

That predicted four-stroke sequence appears in **4.66%** of tracked-video
rallies. Its lower support reflects residual classification error, but its
structure is coherent and was absent from the old baseline.

## 2. The Third Shot After Service Matters

Ground-truth opening patterns show that short service itself is not enough.
The continuation determines whether the server gains an advantage:

| Opening | Server win rate |
|---|---:|
| short service -> push -> clear | **37.50%** |
| short service -> net shot -> lob | 49.13% |
| short service -> lob -> smash | **53.52%** |
| short service -> lob -> tap smash | **60.00%** |

A passive push-clear continuation performs poorly. Openings that create an
attack by the third shot perform better.

This conclusion is GT-only because the deployed pipeline does not predict point
winners.

## 3. The New Pipeline Preserves Tactical Structure

The old integrated pipeline predicted `none` for **56.43%** of strokes. Its
transition matrix had:

```text
Frobenius error = 3.652
mean row KL = 11.723
```

The tracked pipeline reduces the `none` rate to **6.70%** and reduces transition
error to:

```text
Frobenius error = 1.232
mean row KL = 0.511
```

This is the central Phase 11 systems result:

- transition Frobenius error fell by **66.3%**;
- transition KL fell by **95.6%**;
- dominant predicted patterns changed from meaningless `none` chains to
  recognizable attack-defense sequences.

The pipeline has moved from an error-propagation demonstration to a useful
predicted tactical baseline.

## 4. Hard Predictions Are Better Than Soft Transitions

The probability-weighted soft transition matrix performs worse than hard
top-1 predictions:

| Method | Frobenius error | Mean row KL |
|---|---:|---:|
| Hard predictions | **1.232** | **0.511** |
| Soft probabilities | 1.617 | 0.623 |

The BST model is sufficiently accurate that its top prediction preserves
transition structure better than spreading probability mass across several
classes.

High-confidence predictions have slightly lower Frobenius error (`1.211`) but
retain only **35.59%** of strokes. They are best used for reliable examples,
not complete-rally mining.

## 5. Remaining Errors Are Concentrated Around Fine Stroke Distinctions

The transition-error visualization shows that remaining errors are not random.
They concentrate around:

- net shot versus return net;
- drop and push responses;
- rushes;
- long-service continuations.

These are finer tactical distinctions than smash, clear, or service detection.
Improving these classes should reduce sequence error more efficiently than
uniform model changes.

## 6. Far-Side Top-Player Quality Still Controls Reliability

The event-frame occupancy heatmap confirms stable side ordering, but contains:

```text
Top valid event positions: 1,965
Bottom valid event positions: 3,146
```

Phase 10 accuracy follows Top-player pose quality:

| Video | Mean Top pose-valid rate | Accuracy |
|---|---:|---:|
| 42 | 31.64% | 63.97% |
| 43 | 27.75% | **53.27%** |
| 44 | 87.65% | **76.57%** |

The strongest next improvement is therefore not another mining algorithm. It
is better far-side player extraction, especially for video 43.

## 7. Player Styles Remain a Weak Clustering Result

Ground-truth player-match clustering selects two groups but has silhouette
`0.152` and bootstrap stability ARI `0.440`. The groups describe weak
backcourt-oriented versus front/midcourt-oriented stroke-mix tendencies.
Elite singles player-match profiles do not separate cleanly into stable,
simple archetypes.

Predicted video-side clustering selects a balanced `k=2` solution with
silhouette `0.200` and bootstrap stability ARI `0.505`. The raw best
silhouette is `k=3` at `0.225`, but it creates a singleton cluster and is
rejected as an outlier split. The balanced result groups both sides of videos
`42-43` against both sides of videos `41/44`. This is best interpreted as a
video/match-domain or perception-quality effect, not a player-style taxonomy.

## 8. Spatial Findings

Ground-truth landing heatmaps confirm expected stroke roles:

- clears land deep;
- net shots and drops concentrate forward;
- smashes occupy attacking landing regions.

Predicted occupancy cleanly separates the two court halves, validating Phase
09 player ordering. The approximate shuttle endpoint visualization shows the
broadcast-image region where tracked windows terminate, but it must not be
interpreted as true landing position.

## Practical Use

Use the tracked predicted output for:

- transition analysis with hard top-1 predictions;
- common side-aware tactical motifs;
- high-confidence example retrieval;
- court occupancy visualization;
- comparing videos and diagnosing perception quality.

Do not use it as authoritative evidence for:

- point-winning causal claims;
- rare-pattern frequencies;
- persistent player-style identities;
- exact shuttle landing analysis.

## Bottom Line

The project now has two valid analytical layers:

1. **Ground truth:** authoritative tactical conclusions across all 44 videos.
2. **Tracked predictions:** a deployable video-analysis baseline that preserves
   meaningful tactical structure at `66.40%` primary test accuracy.

The old pipeline destroyed rally structure. The new pipeline recovers it. The
remaining bottleneck is far-side player perception and fine net/drive/push
stroke discrimination.
