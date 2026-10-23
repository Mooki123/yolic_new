# 09 — Confirming layout transfer: compute-matched, same-architecture oracles, 3 seeds, irregular cells

Written 2026-10-22, **before any run in this document**. Follows research/08 (GO on seed 0). Thresholds must not change
after results come in.

## What changes from 08
1. **Compute-matched method, "T":** MIL-max (MobileNetV2 at stride 16, class maps upsampled to 112, max per cell,
   per-class bias), input resized on the GPU from the 224 cache to **168x168** (11x11 feature map): **1.064x** YOLIC's
   MACs. Trained on L1 labels only, 3 seeds.
2. **Same-architecture oracle, "O_T":** T's exact architecture retrained on each target layout's own labels — the
   real "retraining" ceiling. 3 seeds on shift and fine; seed 0 on poly.
3. **Irregular layout, poly:** 96 perspective trapezoids receding to a vanishing point at (1024, 300), 12 per row, 8 rows
   from y = 400 to 1024 (layouts/cityscapes_poly.json). No edge coincides with an L1 cell.
4. **Seeds:** 3 for T, for O_T on shift/fine, for YOLIC's oracle O_A on shift/fine (seeds 1-2 added), and for the
   224-input MIL-max of 08 (seeds 1-2 added). Overlap transfer uses YOLIC A's 3 existing seeds.

## Metric
People+Vehicle macro-F1 at 0.5 on Cityscapes test (official val), void cells masked, mean over seeds; paired bootstrap
over test images (1,000 resamples, seeds pooled).

## Claims and thresholds
| Check | Condition |
|---|---|
| **C1 (transfer works at matched compute)**, per target in {shift, fine, poly} | T >= **0.90 x O_T** and T − overlap(A) >= **3.0** with the 95% CI excluding 0 |
| **C2 (no cost on the original layout)** | T on L1 >= A − **1.0** |

| Outcome | Condition |
|---|---|
| **CONFIRMED** | C1 on all three targets, and C2 |
| **PARTIAL** | C1 on two of three targets (C2 either way), or C1 on all three without C2 |
| **NOT CONFIRMED** | otherwise |

Also reported, not decisive: union; T relative to O_A (YOLIC retrained); the 224-input MIL-max at 3 seeds against O_A
(the 08 comparison); the pooled segmenter; MACs.

## Result (2026-10-22 23:05): **NOT CONFIRMED**
P+V macro-F1, Cityscapes test, 3-seed means (O_A and O_T on poly: seed 0 only; O_A on union: seed 0 only).
| Layout | Overlap (A) | O_A (YOLIC retrained) | O_T (T retrained) | **T** (MIL-max s16, 168; 1.06x) | MIL-max s16, 224 (1.76x) |
|---|---|---|---|---|---|
| L1 | 77.77 | = A | = T | 75.18 | 78.54 |
| union | 79.80 | 80.41 | — | 77.17 | 80.09 |
| shift | 73.82 | 77.65 | 74.80 | 74.89 | 78.05 |
| fine | 72.91 | 75.09 | 72.52 | 72.68 | 76.10 |
| poly | 74.47 | 79.66 | 75.16 | 76.38 | 78.99 |
Checks: shift T/O_T 1.001, T − overlap +1.06 [0.28, 1.75]; fine 1.002, −0.23 [−0.89, 0.35]; poly 1.016, +1.91
[0.87, 2.79]. C1 fails on all three (gain over overlap < 3.0); C2 fails (T on L1 75.18 vs A 77.77, −2.59).

Reading:
- **Re-aggregation is lossless relative to retraining the same architecture**: T/O_T = 1.00-1.02 on every target,
  and the 224-input MIL-max is >= YOLIC retrained on shift and fine (99% on poly). That part replicates over 3 seeds.
- **The advantage over the naive overlap baseline needs spatial resolution, which costs compute.** At matched compute
  (168 input) the dense head loses 2.6 points on L1 and its margin over overlap shrinks to about 0-2 points. The
  research/08 GO (+4.1 / +3.3 over overlap) holds over 3 seeds only at 1.76x MACs.
