# 05 — Go/no-go: foveated input for fixed-cell classifiers (direction 1)

Written 2026-10-11, **before any run**. Thresholds must not change after results come in.

## Motivation (from the 04 go/no-go)
At matched compute, GAP+FC (A, 77.77 People+Vehicle macro-F1) beat spatial heads. The heads lost on the 160 small
64x32 cells (about 7x7 px after the 224x224 squash) and won on the 96 large cells. CellPool at stride 16 recovered
the small cells, at 1.76x the MACs. So the bottleneck is input resolution on the far band, not the head.
The cell layout is fixed and known in advance, so the far band can be sampled at high resolution for free.

## Models (Cityscapes, protocol v1 exactly as in 04 section 2.3, 3 seeds each)
- **A** = B1, already run: MobileNetV2 + GAP+FC, 224x224 squash.
- **F** = foveated: a shared MobileNetV2 runs on two views, GAP on each, concat (2560-d) -> Dropout(0.2) -> FC to 256x4.
  - global: the full 2048x1024 frame -> 224x112 (aspect-preserving);
  - fovea: the small-cell band x 512-1536, y 320-640 -> 320x96 (aspect about 3.3, source 3.2).
  - 55,808 px in total vs 50,176 for A; matched means MACs <= 1.2x A (measured with fvcore).
- **R** = resolution control: MobileNetV2 + GAP+FC on the full frame at 336x168 (56,448 px), aspect-preserving.
  This separates "foveation helps" from "more pixels or the correct aspect ratio help".

Both views of one sample get the same flip and the same colour-jitter draw.

## Decision rule (People+Vehicle macro-F1 at 0.5 on Cityscapes val = test, mean of 3 seeds, in points)
| Outcome | Condition | Action |
|---|---|---|
| **GO (method paper)** | F - A >= 2.0 **and** F - R >= 1.0, all 3 F seeds above all 3 seeds of both A and R, and both paired-bootstrap 95% CIs (F vs A, F vs R) exclude 0 | The method becomes the paper's headline; 04's protocol and leakage results become supporting sections. |
| **RESOLUTION ONLY** | R - A >= 2.0 and F - R < 1.0 | The gain is from pixels or aspect ratio, not foveation. Report it as a finding; no method claim. |
| **AMBIGUOUS** | F - A in [1.0, 2.0), or the seeds overlap | Add 2 seeds to F, A and R; GO if the 5-seed F - A >= 1.5 and F - R >= 1.0 with both CIs excluding 0. |
| **FAIL** | otherwise | Drop direction 1; continue the 04 protocol paper. |

Secondary (reported, not used in the decision): small-cell and large-cell P+V macro-F1, the paper's All F1, and MACs.
