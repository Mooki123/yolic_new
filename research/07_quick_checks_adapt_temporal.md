# 07 — Quick checks: session self-adaptation (A) and temporal persistence (B)

Written 2026-10-22, **before running either check**. Thresholds must not change after results come in.

## Setting
Outdoor Hazard, session-grouped split (`splits/outdoor_grouped`, 48 test sessions that share no session with train),
the existing B1 checkpoints (MobileNetV2 + GAP+FC, protocol v1), seeds 0-2. No retraining.
Metric: object macro-F1 at 0.5 over the 11 hazard classes, mean of 3 seeds, paired bootstrap over test frames.
Frames are ordered within a session by their 8-digit frame counter.

## A — re-estimate BatchNorm statistics on the new session (unlabelled)
- **A-session (transductive upper bound):** each test session's BN statistics are recomputed from all its frames
  (AdaBN), then the session is evaluated.
- **A-online (deployable):** frames are processed in temporal order in batches of 16; each BN layer normalises with
  the cumulative statistics of every frame of this session seen so far, current batch included. Nothing from later
  frames is used; statistics reset at each new session.

## B — temporal persistence of hazards
- **B-causal (deployable):** p'_t = a * p_t + (1 - a) * p'_{t-1} within a session, reset when the frame counter jumps by
  more than 10x the session's median step. a is chosen on the grouped **val** split from {0.3, 0.5, 0.7, 0.9}.
- **B-window (non-causal upper bound):** mean over frames t-1, t, t+1.
- Also A-online followed by B-causal.

## What counts as promising (deployable variants only: A-online, B-causal, and both)
| Outcome | Condition (test, vs B1 as trained, 3-seed mean) | Next step |
|---|---|---|
| **Promising** | gain >= **3.0** F1 points, bootstrap 95% CI excluding 0 | Develop it into the paper's method (proper adaptation baselines: TENT, etc.) |
| **Weak** | gain in [1.0, 3.0) | Keep as a component; not a headline |
| **Flat** | gain < 1.0 | Drop; move to idea C (layout transfer) |

## Result (2026-10-22): both **Flat** (negative) — drop A and B
Test object macro-F1, 3 seeds (B1 as trained: 50.35 / 50.78 / 49.70, mean 50.28):
| Variant | Mean | Gain vs B1 [95% CI] |
|---|---|---|
| A-session (upper bound) | 29.83 | −20.45 [−21.27, −19.59] |
| A-online | 24.37 | −25.91 [−26.83, −24.95] |
| B-causal (a = 0.9, chosen on val) | 50.13 | −0.15 [−0.20, −0.10] |
| B-window (upper bound) | 41.30 | −8.97 [−9.54, −8.35] |
| A-online + B-causal | 24.03 | −26.25 [−27.19, −25.29] |

Diagnostics (seed 0), ruling out a bug:
- BN re-estimated on 3,000 *training* frames: 50.78 (baseline 50.34), so the adaptation code is correct.
- BN re-estimated on all 4,071 test frames pooled: 42.16, so adaptation hurts even without small-session noise
  (sessions: min 2, median 49, max 613 frames).
- Consecutive frames are 1 s apart (counter step 30); 16.6% of all cells change their hazard bits frame to frame,
  against 24.3% of cells containing any hazard, so per-cell temporal smoothing blurs moving hazards.
Per the rule above: drop both; next is idea C (layout transfer).
