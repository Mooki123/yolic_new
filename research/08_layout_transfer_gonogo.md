# 08 — Go/no-go: layout transfer from cell labels alone (idea C, review survivor #2)

Written 2026-10-22, **before any code or run for this idea**. Thresholds must not change after results come in.

## Claim under test
A model trained only with cell-level presence labels on one layout (L1 = YOLIC's 256-cell Cityscapes layout) can be
re-aggregated to an **unseen** layout L2 at deployment, with no L2 labels and no retraining, and recover most of the
accuracy of a model retrained on L2's labels. The review found no prior work that re-aggregates a cell classifier to
a different layout; the pooling methods themselves (MIL max / LSE / mean) are known and are not the claim.

## Target layouts (Cityscapes 2048x1024, all left-right symmetric; labels from full-res masks with the v1 rule)
| L2 | Small band (L1: 16x10 cells of 64x32 over x 512-1536, y 320-640) | Large band (L1: 16x6 cells of 128x64 over x 0-2048, y 640-1024) | Cells |
|---|---|---|---|
| union | 2x2 merges: 8x5 of 128x64 | 2x2 merges: 8x3 of 256x128 | 64 |
| shift | 15x10 of 64x32 over x 544-1504, y 336-656 | 15x6 of 128x64 over x 64-1984, y 672-1024 (last row clipped to 32 px) | 240 |
| fine | 2x2 splits: 32x20 of 32x16 | 2x2 splits: 32x12 of 64x32 | 1024 |

## Methods (all trained on L1 labels only, except the oracle and the segmenter)
- **Overlap (baseline):** A = YOLIC GAP+FC (existing checkpoints); an L2 cell's score = max of the L1 cells it overlaps.
- **Dense, mean pooling:** D = CellPool-s16 (existing checkpoints); class maps re-pooled with L2 area weights; the
  per-(cell, class) bias is replaced by its per-class mean (it cannot transfer).
- **Dense, MIL max / LSE (r = 5):** new heads on MobileNetV2 at stride 16 (as D): 1x1 conv class maps, bilinearly
  upsampled x4, pooled over each cell's coverage, per-class bias only. Trained on L1 with protocol v1.
- **Ceiling with masks:** B8 LR-ASPP-MNV3-S (existing), max over the pixels of each L2 cell.
- **Oracle:** A's architecture retrained directly on each L2 layout's labels.

Selection: the dense pooling (mean / max / LSE) is chosen on **val'** by mean P+V macro-F1 over shift and fine.

## Go/no-go runs (seed 0; Cityscapes test = official val, P+V macro-F1 at 0.5, void cells masked)
New training: MIL-max, MIL-LSE (on L1), and the oracle on each of union, shift, fine. Everything else reuses checkpoints.

## Decision (the selected dense transfer, "T")
| Outcome | Condition |
|---|---|
| **GO** | On **both** shift and fine: T >= **0.90 x oracle**, **and** T − overlap >= **3.0** points with the bootstrap 95% CI excluding 0 |
| **PARTIAL** | GO's conditions hold on one of shift / fine |
| **NO-GO** | otherwise |
Also reported, not decisive: union (expected easy for every method), T's own L1 score vs A, and MACs (the stride-16
heads cost about 1.76x A, which a GO would have to address with a compute-matched variant).
If GO: 3 seeds, random-polygon layouts and a compute-matched dense head before any paper claim.

## Result (2026-10-22 18:20, seed 0): **GO**
P+V macro-F1 on test (val' in brackets). Selected dense pooling on val': **MIL-max**.
| Layout | Overlap (A) | Oracle (A arch, retrained) | Mean (D) | **MIL-max** | MIL-LSE | Segmenter (masks) |
|---|---|---|---|---|---|---|
| L1 (training layout) | 77.65 (76.30) | = A | 77.30 (75.36) | 78.55 (76.83) | 78.80 (76.67) | 70.63 (68.20) |
| union | 79.65 (78.09) | 80.41 (79.40) | 66.62 (63.76) | 79.84 (77.92) | 80.12 (77.89) | 71.11 (69.26) |
| shift | 73.73 (71.85) | 77.52 (75.77) | 77.25 (74.59) | **77.81** (75.73) | 78.39 (75.47) | 70.45 (67.98) |
| fine | 72.92 (71.62) | 75.07 (73.33) | 74.32 (72.68) | **76.26** (74.33) | 75.51 (73.35) | 70.06 (67.50) |
Checks: shift T/oracle 1.004, T − overlap +4.08 [3.31, 4.79]; fine T/oracle 1.016, T − overlap +3.34 [2.70, 3.94].
Both pass → GO.

Caveats that the follow-up must address before any claim (as required above):
- **Compute:** MIL-max runs at stride 16 (1.76x A's MACs) while the oracle is A's architecture; MIL-max also beats A
  on L1 itself (+0.90). Part of "transfer >= oracle" may be the extra compute. Needed: a compute-matched dense head,
  and an oracle of the same architecture retrained on each L2 (the true "retraining" ceiling).
- **One seed** for every model.
- Mean pooling collapses on union (66.6): averaging dilutes evidence in larger cells; max/LSE do not.
- The pooled segmenter is below every cell model (as in research/04: recall loss under the >= 1 px rule at 224).
