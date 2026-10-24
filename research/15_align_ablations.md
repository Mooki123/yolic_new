# 15 — What in the Align head matters? (ablations on F+AL)

Written 2026-10-24, **before any ablation run**. Thresholds must not change after results come in.

Reference: **F+AL** (research/13; foveated 224x112 + 320x96 views, per-view BN, Align head per view, per-cell view
blend): Cityscapes test People+Vehicle macro-F1 86.83 (3 seeds: 86.99 / 86.70 / 86.81).

## Variants (each removes exactly one part; everything else identical to F+AL; protocol v1; 3 seeds)
| Variant | Change |
|---|---|
| **no-lse** (mean only) | cell readout = mask-weighted mean (the soft max and the learned mix are removed) |
| **no-mean** (soft max only) | cell readout = mask-weighted log-sum-exp (T = 4) only |
| **no-lateral** | the stride-16 lateral branch is removed; the classifier sees only the upsampled stride-32 map (1280 ch) |
| **no-cellbias** | the per-(cell, class) bias is removed (the classifier keeps its per-class bias) |

## Reading (vs F+AL, 3-seed means, paired bootstrap over test images, seeds pooled)
| Outcome for a part | Condition on (variant − F+AL) |
|---|---|
| **Needed** | <= −1.0 with the 95% CI excluding 0 |
| **Not needed** | within ±0.5 |
| **Better without it** | >= +0.5 with the CI excluding 0 |
| **Minor** | otherwise |
Also reported: far / near cells, parameters, MACs.

## Result (2026-10-24 23:57)
Cityscapes test, People+Vehicle macro-F1, 3 seeds; delta vs F+AL (86.83) with paired-bootstrap 95% CI.
| Variant | Seeds | Mean | Far / near | Δ vs F+AL | Reading |
|---|---|---|---|---|---|
| F+AL (full) | 86.99 / 86.70 / 86.81 | 86.83 | 86.9 / 85.8 | — | — |
| no-lse (mean only) | 86.99 / 87.04 / 86.97 | **87.00** | 87.0 / 86.9 | +0.18 [−0.03, +0.37] | **not needed** |
| no-mean (soft max only) | 84.08 / 84.03 / 83.90 | 84.00 | 84.2 / 79.6 | −2.83 [−3.14, −2.55] | **needed** |
| no-lateral | 84.49 / 84.80 / 84.54 | 84.61 | 84.8 / 81.0 | −2.22 [−2.49, −1.96] | **needed** |
| no-cellbias | 86.60 / 86.75 / 86.60 | 86.65 | 86.6 / 86.2 | −0.18 [−0.37, +0.01] | **not needed** |
Reading: two parts carry the gain — the **stride-16 lateral fusion** (fine spatial detail at ~no compute) and the
**mask-weighted mean readout**. The log-sum-exp "soft max" and the per-cell bias add nothing measurable; a simplified
head (mean readout over the fused stride-16 map, no LSE, no cell bias) should match F+AL. The soft-max-only readout
hurts most on near (large) cells (79.6 vs 85.8), where averaging over many locations is the robust choice.
