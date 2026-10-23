# 06 — Can foveation become a real method? (budget sweep + transfer test)

Written 2026-10-18, **before any run in this document**. Thresholds must not change after results come in.
Builds on research/05 (GO, narrowly: F − A = +2.23, F − R = +1.19 People+Vehicle macro-F1 on Cityscapes test).

## The method, stated as a rule (so it can transfer without tuning)
Given a fixed cell layout and a pixel budget matched to the baseline (MACs <= 1.2x the 224x224 GAP+FC model):
1. **Fovea** = the bounding box of the layout's smallest-cell group (Cityscapes: small_64x32, x 512-1536 y 320-640;
   outdoor: far_34px, x 220-628 y 166-268). Both are symmetric about the frame's vertical midline, so the flip
   augmentation stays exact.
2. **Context view** = the full frame, aspect-preserving.
3. **Split**: a fraction *s* of the pixel budget goes to the fovea, and each view keeps its source aspect ratio.
4. One shared backbone, GAP per view, concatenation, one Linear (FoveaGapFc).

The only free parameter is *s*. It is chosen on Cityscapes `val'` (stage 1) and then frozen for every other dataset.

## Stage 1 — choose s on Cityscapes val' only (seed 0, protocol v1)
| Variant | Context view | Fovea | Pixels | s | MACs vs A |
|---|---|---|---|---|---|
| V1 (= F of research/05; already run) | 224x112 | 320x96 | 55,808 | 0.55 | 1.13x |
| V2 | 160x80 | 352x112 | 52,224 | 0.75 | 1.09x |
| V3 | 128x64 | 384x120 | 54,272 | 0.85 | 1.12x |
| V4 | 96x48 | 416x128 | 57,856 | 0.92 | 1.16x |
| V5 (two-band control: near band x 0-2048 y 640-1024 instead of the full frame) | 256x48 | 352x112 | 51,712 | 0.76 | 1.08x |

**Selection:** the variant with the highest `val'` People+Vehicle macro-F1 at 0.5 (seed 0; V1 uses its seed 0).
Test-set numbers are not looked at for V2-V5 until the winner is fixed. If V5 wins, the rule's context view
becomes "the bands of the other cell groups" instead of the full frame.

## Stage 2 — the winner, 3 seeds, Cityscapes test (one evaluation)
## Stage 3 — transfer to outdoor, grouped split, 3 seeds, no outdoor tuning
- Baseline: B1 (MobileNetV2 + GAP+FC, 224x224, protocol v1) on `splits/outdoor_grouped`; seed 0 already exists.
- Foveated: the stage-1 rule with the chosen *s*, applied to the outdoor layout (sizes rounded to multiples of 8,
  keeping each view's source aspect ratio; MACs <= 1.2x B1, measured).
- Metric: object macro-F1 at 0.5 over the 11 hazard classes (the outdoor primary metric).

## Verdict (all deltas in F1 points, 3-seed means, paired-bootstrap 95% CI over test images)
| Check | Condition |
|---|---|
| **M1 — size (Cityscapes)** | winner − A >= **3.0** with CI excluding 0, **and** winner − R >= **1.5** (R = 336x168 control of 05) |
| **M2 — transfer (outdoor)** | foveated − B1 >= **2.0** with CI excluding 0, with *s* frozen from stage 1 |

| Outcome | Reading |
|---|---|
| M1 and M2 | **Real method.** The paper leads with layout-driven foveated input; the 04 protocol and leakage results support it. |
| M2 only | Real but modest. A method section with a cross-dataset claim; the headline still needs the detector comparison. |
| M1 only | Cityscapes-specific. Report as a positive finding inside the protocol paper; no general method claim. |
| neither | Tops out around +2. Protocol paper; foveation reported as one controlled result. |

## Addendum — stage 1 outcome and the frozen rule (written 2026-10-18 19:05, before any outdoor foveated run)
Stage 1 (Cityscapes val', seed 0, People+Vehicle macro-F1): V1 78.54, V2 76.26, V3 72.54, V4 64.96, **V5 79.89**
(A 76.30, R 77.42). V5 wins, so, as specified above, the context view is replaced by the bands of the other cell
groups. Test numbers of V2-V5 were not looked at.

V5 gives the smallest-cell group 39,424 of 51,712 pixels (s = 0.76). For a layout with more than one other group
(outdoor has near_53px and top_60px), the rule is fixed as:
1. one view per cell group, cropped to the bounding box of that group's cells (each must be symmetric about the
   vertical midline);
2. the smallest-cell group gets s = 0.76 of the pixel budget; the other groups share the remaining 0.24 in
   proportion to their bounding-box areas;
3. each view keeps its source aspect ratio, sides rounded to multiples of 8, no side below 32 px;
4. pixel budget = 51,712 (V5's total); MACs must stay <= 1.2x the 224x224 baseline.
Stage 2 runs V5 (seeds 1-2 added to seed 0). Stage 3 applies this rule to outdoor unchanged.

## Result and a validity problem (2026-10-19)
Pre-registered verdict as computed: **TOPS OUT** — M1: V5 − A = +2.48 [1.85, 3.22], V5 − R = +1.45 [0.89, 2.08];
M2: outdoor bands − B1 = −21.7 [−22.5, −20.9].

**This verdict is not a valid test of the method.** The FoveaGapFc implementation shares one MobileNetV2, including its
BatchNorm layers, across views with very different image statistics. Training normalises each view with its own batch
statistics; evaluation uses running statistics averaged over all views. Diagnostics (not used for any decision):
- outdoor bands seed 0 on its own *training* frames: F1 65.3 with running stats vs 86.4 with batch statistics
  (B1: 94.5) — a train/eval mismatch, not overfitting;
- Cityscapes V5 seed 0 on val': 79.89 → 81.29 with batch statistics; large cells 67.18 → 80.74.
The large-cell drop reported for every foveated variant (F, V1-V5) is this bug. All multi-view numbers so far are
affected; single-view models (A, R, B1) are not.
Fix (an implementation correction, not a design change; views, s and thresholds stay frozen): per-view BatchNorm —
shared convolution weights, separate BN layers per view. Re-run V5 and outdoor bands, 3 seeds each, and re-apply the
verdict above unchanged.

## Re-run plan with per-view BatchNorm (written 2026-10-20, before any per-view-BN run)
The shared-BN bug may also have changed the stage-1 ranking, so the whole procedure is repeated with
`per_view_bn: true`; nothing else changes (views, sizes, protocol v1, thresholds).
1. Stage 1 again: V1-V5, seed 0, selection on val' People+Vehicle macro-F1 only (tools/fovea_select.py --suffix _pvbn).
   V1 is F of research/05, so its seeds 1-2 are also run to replace research/05's numbers.
2. Stage 2: the winner, 3 seeds, Cityscapes test.
3. Stage 3, outdoor grouped split, 3 seeds, no outdoor tuning: if V5 wins, the band rule of the addendum (s = 0.76);
   if V1-V4 wins, the context rule (full frame + fovea = far_34px box) with that variant's s and pixel total.
4. Verdict: M1 and M2 exactly as above (tools/fovea_method_verdict.py pointed at the per-view-BN runs).

### Per-view-BN stage 1 outcome (2026-10-20 04:30, before phase B)
val' P+V, seed 0: V1 82.55, V2 83.00, V3 83.01, **V4 83.36**, V5 83.26 (A 76.30, R 77.42). Winner by the rule: V4
(context 96x48 + fovea 416x128, s = 0.92, 57,856 px); V5 is 0.10 behind, within seed noise. Stage 3 therefore uses the
context rule; for outdoor it derives context 88x48 + fovea (far_34px box, x 220-628 y 166-268) 464x112, 1.18x B1 MACs.
Known risk, noted before running: outdoor near cells (about 45% of hazard positives) are then seen only through the
88x48 context view.

## Per-view-BN result (2026-10-20 11:20) — the valid verdict: **CITYSCAPES-SPECIFIC (M1 only)**
- **M1 passes.** V4, Cityscapes test, P+V macro-F1 83.85 / 83.75 / 83.77 (mean 83.79) vs A 77.76: **+6.03 [5.42, 6.75]**;
  vs R 78.81: **+4.99 [4.46, 5.63]**; MACs 1.16x A. Small cells 84.32 vs 77.88; large cells 71.51 vs 74.69 (the 96x48
  context view is too small for them).
- **M2 fails.** Outdoor grouped, context rule with V4's split: 46.80 / 44.78 / 45.84 (mean 45.81) vs B1 50.28:
  **−4.47 [−5.13, −3.74]**. Far cells 51.96 vs 52.98 (no gain); near cells 45.41 vs 56.51 — the risk noted before
  the run (near cells only seen through the 88x48 context view).
- research/05 corrected (F = V1 with per-view BN, 3 seeds): 83.32 / 83.26 / 83.27 (mean 83.28); F − A **+5.51
  [4.91, 6.18]**, F − R **+4.48 [3.97, 5.07]**; small cells 83.47, large 77.73 (no large-cell loss with the 224x112
  context view). The shared-BN numbers in research/05 are superseded.
Anything run from here on outdoor with a different split or rule is exploratory, not a test of this pre-registration.
