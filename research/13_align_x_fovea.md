# 13 — Align head x foveated input: a 2x2 on Cityscapes

Written 2026-10-23, **before any code or run for this experiment**. Thresholds must not change after results come in.

## Design (Cityscapes, protocol v1, People+Vehicle macro-F1 at 0.5 on test, 3 seeds, paired bootstrap)
| | YOLIC head (GAP + FC) | Align head (research/12) |
|---|---|---|
| **224x224 input** | A — done (77.77) | **AL — new** |
| **Foveated input (F: 224x112 full frame + 320x96 far band, per-view BN)** | F — done (83.28) | **F+AL — new** |

- **AL** is YOLIC-Align's `align` head ported unchanged: MobileNetV2 split at stride 16; stride-32 map upsampled to
  stride 16 and concatenated with a 1x1-conv lateral (128 ch) of the stride-16 map; 1x1 classifier to per-location
  logits; per cell, a learned per-class blend of a mask-weighted mean and a mask-weighted log-sum-exp (T = 4) over
  bilinear (RoIAlign-style) cell masks, plus a per-(cell, class) bias. The port must reproduce the original module's
  outputs on identical weights (unit test) before any run.
- **F+AL** applies the same head per view: each view has its own backbone + lateral BatchNorm (shared convolutions,
  research/06 fix), one shared classifier, cell masks computed in that view's crop coordinates. A cell's logit is a
  per-cell learned softmax blend of the readouts of the views that cover it (equal at initialisation), plus the cell
  bias. Far cells are covered by both views; near cells only by the full-frame view.

## Questions and thresholds
| Question | Measure | "Yes" if |
|---|---|---|
| Q1: does Align help on its own? | AL − A | >= +1.0 with the 95% CI excluding 0 |
| Q2: is the combination worth keeping? | F+AL − F | >= +1.0 with CI excluding 0, **or** within ±0.5 while using <= 0.6x F's parameters |
Reported, not decisive: parameters, MACs, far/near cell breakdown, AL − A on the far band.

## Result (2026-10-23 21:30): **Q1 YES, Q2 YES**
Cityscapes test, 3 seeds each, People+Vehicle macro-F1:
| | YOLIC head | Align head |
|---|---|---|
| 224 input | A 77.65 / 77.76 / 77.88 = **77.77** (3.54M params, 1.00x MACs) | AL 79.61 / 79.40 / 79.42 = **79.48** (2.24M, 1.01x) |
| Foveated | F 83.32 / 83.26 / 83.27 = **83.28** (4.88M, 1.13x) | F+AL 86.99 / 86.70 / 86.81 = **86.83** (2.28M, 1.14x) |
- Q1: AL − A = **+1.71 [1.35, 2.10]** → yes.
- Q2: F+AL − F = **+3.55 [3.23, 3.88]** → yes (and 0.47x F's parameters).
- F+AL − A = **+9.06 [8.41, 9.80]**: about 41% fewer errors than YOLIC, with 0.64x its parameters at 1.14x its MACs.
- Far / near cells: A 77.9 / 74.7, AL 79.1 / 86.5, F 83.5 / 77.7, **F+AL 86.9 / 85.8** — foveation fixes the far band,
  Align fixes the near band, and they add up. Paper-style "All" F1: A 82.67, F+AL **89.15** (the paper reports 82.02).
- Unlike the spatial heads of research/04 (CellPool-s32, −5.7), Align does not lose the far band: it keeps stride-16
  detail through the FPN-style lateral at no dilation cost, and its LSE readout plus per-cell bias keep YOLIC's prior.
