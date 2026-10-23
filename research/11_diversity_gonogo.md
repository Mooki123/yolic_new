# 11 — Go/no-go: attacking the lack of training diversity on outdoor (two ideas)

Written 2026-10-23, **before any code or run for either idea**. Thresholds must not change after results come in.
Motivation (research/10): outdoor fails across sessions mainly because training is one session (56% of frames, top 3 =
80%) and the class mix shifts between sessions; not label noise, not something test-time adaptation fixes.

Setting for both: Outdoor Hazard, `splits/outdoor_grouped`; baseline **B1 = 50.28** object macro-F1 (11 hazard classes,
at 0.5; 3 seeds: 50.35 / 50.78 / 49.70); paired bootstrap over test frames, seeds pooled. Selection only on the grouped
**val** split.

## Idea 1 — foundation-model features (feasibility of a teacher)
Question: do features pretrained on far more diverse images generalise across sessions much better than an ImageNet
MobileNetV2 fine-tuned on our three streets? If yes, distilling them into MobileNetV2 becomes the method.
- Backbone: **DINOv2 ViT-S/14** (public weights), **frozen**, input 224x224 (the same squash as B1). Features cached once
  for every frame and its horizontal flip: the CLS token plus the 16x16 patch tokens average-pooled to 8x8.
- Heads (trained on cached features, flip augmentation via the cached flipped features, Adam, BCE, selection on val):
  - **linear:** [CLS, mean of patches] (768-d) → Linear → 104x12 (the GAP+FC analogue);
  - **spatial:** 8x8 patch grid → 1x1 conv to 64 → flatten → Linear → 104x12.
  The head with the better val score is the one reported; 3 head seeds.
- **GO (worth distilling):** best head − B1 >= **5.0** with the 95% CI excluding 0.
  **WEAK:** gain in [2.0, 5.0). **NO:** < 2.0.
  This is a feasibility check of the teacher only; it is not a deployable model (ViT-S costs ~15x B1's MACs).

## Idea 2 — session-balanced training
Question: if every training session counts more equally, does the model generalise better?
- B1 exactly (protocol v1, 150 epochs), except each epoch draws len(train) frames **with replacement**, frame weight
  proportional to 1 / (frames in its session)^beta. beta = 1: every session has equal total weight; beta = 0.5: softer.
- Stage 1: beta in {0.5, 1.0}, seed 0, pick by val object macro-F1. Stage 2: the winner, seeds 1-2.
- **GO:** 3-seed mean − B1 >= **5.0** with CI excluding 0. **MODEST:** gain in [2.0, 5.0) with CI excluding 0.
  **NO:** otherwise.

## Idea 1 result (2026-10-23 13:30): **NO**
Frozen DINOv2 ViT-S/14, outdoor grouped test, 3 head seeds: spatial head (chosen on val, 53.61 vs linear 22.82)
50.12 / 49.37 / 50.04 (mean 49.84) vs B1 50.28: **−0.44 [−1.40, +0.47]**. Linear head: 23.34 (global pooling discards
the location that cell labels need).
Reading: features pretrained on far more diverse images, with a head trained on our three streets, generalise to new
sessions exactly as well as the fine-tuned MobileNetV2 — no better. This supports research/10's diagnosis that the
outdoor gap is set by the training data (few sessions, class-prevalence shift), not by the representation. Not tested:
fine-tuning DINOv2, higher-resolution patch grids. Distillation is not pursued.

## Idea 2 result (2026-10-23 19:00): **NO**
Stage 1 (val, seed 0): beta 0.5 56.34, beta 1.0 55.80 (B1 seed 0 val 56.43) → beta 0.5. Test, 3 seeds:
49.25 / 51.26 / 50.10 (mean 50.20) vs B1 50.28: **−0.08 [−0.47, +0.38]**. Per-class F1 moves by at most ±3 points in
either direction (TrafficSign still 0.3).
Reading: reweighting cannot add diversity the data does not contain — the model still sees the same 26 sessions, and
balancing them neither helps nor hurts.

## Overall reading of research/07-11 on outdoor
Nothing tried moves cross-session F1 off ~50: test-time BN adaptation (−20 to −26), temporal smoothing (≤ 0),
foveation (−4.5), frozen DINOv2 features (−0.4), session balancing (−0.1); research/04 heads were on Cityscapes. The
outdoor ceiling is set by the training data (26 sessions, one at 56%, class-prevalence shift), not by the model, the
representation or the sampling.
