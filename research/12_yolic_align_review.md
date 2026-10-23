# 12 — Review of the earlier YOLIC-Align runs (indoor, outdoor; single seed)

Written 2026-10-23 from the files in `~/dev/yolic/` (results) and `~/dev/yolic/code` (the code that ran: YOLIC-Align
`main` at 3d074c4 plus uncommitted path edits). Nothing was re-run.

## What YOLIC-Align does
`yolic_align.py` replaces YOLIC's head (global average pool + one Linear producing every (cell, class) logit, 1.25M
parameters on outdoor) with a **shared cell mask-pooling head**:
1. MobileNetV2 is split at stride 16: features[:14] (96 ch, 14x14) and features[14:] (1280 ch, 7x7).
2. The stride-32 map is bilinearly upsampled to 14x14 and concatenated with a 1x1-conv lateral of the stride-16 map
   (128 ch) — an FPN-style fusion that reuses features the backbone computes anyway, so there is no dilation cost.
3. A 1x1 conv turns the fused map into per-location class logits.
4. Each cell reads them out through fixed bilinear mask weights computed exactly from the cell geometry (rectangles,
   or rasterised polygons for indoor), as a learned per-class blend of a weighted mean and a weighted log-sum-exp
   (soft max, temperature 4), plus a learned per-(cell, class) bias.
An older variant (`align-v1`: stride-32 mask-average pooling + shared Linear) exists; both trained checkpoints use
`align` (verified from their keys).

The training script also fixes two released bugs: the flip (now left-right, `flip(2)`) and the random flip on
validation/test (now training only). Everything else is the released protocol: unsorted `os.listdir` +
`train_test_split(random_state=2)` per-frame random split, no ImageNet normalisation, hue jitter 0.5, checkpoint
selected on validation exact-cell accuracy, one seed.

## Cost
| | Baseline | Align | Change |
|---|---|---|---|
| Outdoor params | 3.823M (15.5 MB file) | 2.255M (9.3 MB) | **−41%** |
| Indoor params | 2.493M | 2.246M | −10% |
| Cityscapes params (built, not trained) | 3.536M | 2.243M | −37% |
| MACs at 224x224 (fvcore) | 314.5M | 319.9M | +1.7% (fvcore does not count the log-sum-exp readout) |
| CPU latency, 1 thread (laptop proxy, outdoor) | 17.1 ms | 14.7 ms | about the same (noisy) |
Align saves parameters/storage, not compute: the backbone dominates MACs.

## Results as reported (test split, single seed)
Object macro-F1 = mean F1 over the hazard classes (Road and the dummy Background excluded), computed from the reports.
| Dataset | Baseline | Align | Δ | Exact-cell acc. (test) | Risk F1 (binary) |
|---|---|---|---|---|---|
| Outdoor (11 classes) | 84.57 | **90.11** | **+5.54** | 95.77 → 96.98 | 93.35 → 94.98 |
| Indoor (6 classes) | 90.70 | **93.36** | **+2.66** | 93.97 → 96.08 | 96.16 → 97.06 |
Outdoor per-class gains are largest on rare/small classes: TrafficSign 70.1 → 86.1, TrafficCone 77.9 → 89.3,
Column 73.2 → 82.1, Creature 81.3 → 88.5; no class gets worse.

## Why these numbers cannot yet support a claim
1. **Leaky split.** Per-frame random split; research/04 measured that this inflates outdoor object macro-F1 by 37
   points (87.6 vs 50.3). Absolute numbers are memorisation-heavy, and a head with more spatial capacity may simply
   memorise near-duplicates better — the gain needs to be shown on the session-grouped split.
2. **One seed.** Seed noise on these setups is about 0.5-1 point (research/04/05); +2.7 is not separable from it
   without more seeds and a paired bootstrap.
3. **Unequal training.** Indoor: baseline 42 epochs vs Align 65. Outdoor: both 65 epochs, but the baseline checkpoint
   (2026-09-14) predates this code (2026-09-27), so it was probably trained with the released script — vertical-flip
   bug and augmented validation — while Align had both fixed. Its best epoch was the last one (still improving).
   Neither run reached the planned 150 epochs, so the LR drops at 100/125 never happened.
4. **Indoor test sets differ.** Per-class supports differ between the two indoor reports (Sofa 918 vs 934, Wall 7,597
   vs 7,880, Background 1,438 vs 1,068) with the same number of test images (1,282): the unsorted `os.listdir` split
   produced different test images for the two evaluations. Outdoor supports match exactly.
5. Some outdoor labels came from a `yoliclabel/` directory that no longer exists, so they cannot be re-checked.

## Reading
The outdoor gain (+5.5) is large and consistent across every class, and outdoor Align (90.1) also exceeds our properly
trained B1 on the same random split (87.6, research/04), so it is probably not only the baseline's bugs. The design is
interesting for a reason the size framing misses: it gets stride-16 spatial detail at ~1.0x MACs by reusing the
backbone's own stride-16 features, whereas our CellPool-s16 / MIL-s16 needed 1.76x via dilation (research/04, 08-09).
That makes it a compute-matched candidate for exactly the far-band resolution bottleneck found on Cityscapes.

## What would make it a result
Port the `align` head into yolic_bench unchanged and run it under protocol v1 against B1, 3 seeds, paired bootstrap:
Cityscapes (People+Vehicle, where resolution is the bottleneck) and the outdoor session-grouped split; plus the
outdoor random split for continuity with these runs. Report parameters, MACs and latency.
