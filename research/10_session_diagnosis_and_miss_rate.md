# 10 — Why outdoor fails across sessions (diagnosis), and do miss-rate guarantees survive session shift?

Written 2026-10-22, **before running either part**. Uses existing predictions only (no training).

## Part 1 — diagnosis of the cross-session gap (descriptive; no thresholds)
Model: B1 (MobileNetV2 + GAP+FC, protocol v1) on `splits/outdoor_grouped`, 3 seeds; B1 on the random split (seed 0)
for contrast. Questions and measurements:
1. **Spread:** object macro-F1 per test session (classes with positives in that session); is the gap uniform or driven by
   a few sessions?
2. **Class shift:** per-class positives per frame in train vs test sessions; per-class F1, random vs grouped split.
3. **Visual distance:** per test session, the median nearest-neighbour pHash distance of its frames to the training
   frames, against its F1 (Spearman). A strong negative correlation means genuine appearance shift.
4. **Label consistency:** for near-identical frame pairs (pHash Hamming <= 4), the cell-label agreement for pairs from the
   **same** session against pairs from **different** sessions. Much lower cross-session agreement would mean annotation
   conventions differ between sessions, which no model can learn.

## Part 2 — conformal miss-rate control under session shift (idea 2, review survivor #3)
Target: **cell-level hazard miss rate** (FNR). A cell is a hazard if any of the 11 classes is present; its score is the
max class probability. Per image, loss = missed hazard cells / hazard cells (0 if none). Conformal risk control picks the
largest threshold whose calibration risk satisfies (n/(n+1)) R_hat + 1/(n+1) <= alpha; it promises E[test FNR] <= alpha
when calibration and test images are exchangeable.

Data: the 68 sessions never used for B1 grouped training (its val + test sessions), B1 grouped seed 0 probabilities.
500 random draws for each scheme, alpha in {0.05, 0.10, 0.20} (primary 0.10), calibration share 50%:
- **frame-level split (control):** frames assigned to calibration/test at random, ignoring sessions (the leaky way);
- **session-level split (deployment):** whole sessions assigned to calibration/test (new sessions at test time).
Also reported: the fixed 0.5 threshold, the naive empirical threshold (no +1 correction), and the flagged-cell rate.

**Decision (primary alpha = 0.10):**
| Outcome | Condition |
|---|---|
| **Material** | session-level: mean realised FNR >= alpha + 0.02, **or** P(realised FNR > alpha + 0.05) >= 0.20; **and** frame-level: P(realised FNR > alpha + 0.05) <= 0.05 |
| **Not material** | otherwise — the guarantee is robust to session shift at this data size |

## Results (2026-10-23)
### Part 1 — diagnosis
- **Broad failure:** per-session object macro-F1 (48 test sessions) quartiles 21.4 / 36.7 / 44.1 / 60.1 / 97.7; 64.5% of
  test frames are in sessions scoring below 40.
- **Training data has almost no session diversity:** 26 train sessions, but one holds 56% of training frames (7,945)
  and the top 3 hold 80%; test spans 48 sessions (largest 15%). The giant session is 39% of the whole dataset, so every
  grouped 70/10/20 split puts it in training.
- **Class mix shifts:** positives per frame, train → test: Bump 2.78 → 7.55, Fence 0.26 → 1.27, Creature 0.64 → 2.29,
  Wall 1.41 → 3.83, TrafficCone 0.16 → 0.44. Grouped-split F1 collapses most on rare or session-specific classes
  (TrafficSign 0.3, TrafficCone 24.1, Dent 27.7, Fence 35.6; random split 77-97).
- **Visual distance does not predict failure:** session median pHash NN distance vs F1, Spearman rho = −0.08 (p = 0.57).
- **Labels are consistent within sessions:** pixel-identical frame pairs (181) agree on 97.7% of hazard cells; the lower
  agreement of merely similar frames (73-81%) is camera motion. No near-identical pairs exist across sessions, so
  cross-session annotation conventions cannot be checked this way.
Reading: the gap is mainly **too little training diversity plus class-prevalence shift**, not annotation noise and not
something low-level appearance distance captures.

### Part 2 — conformal miss-rate control: **NOT MATERIAL**
6,107 images, 68 unseen sessions, 500 draws, alpha = 0.10: frame-level mean FNR 0.0973 (5-95%: 0.087-0.107,
P(>0.15) = 0); session-level mean FNR 0.0954 (5-95%: 0.052-0.150, P(>0.15) = 0.054). The guarantee holds on average
under session shift; only its spread widens (about 4x). Side findings: the default 0.5 threshold misses **33.5%** of
hazard cells on new sessions; meeting alpha = 0.10 requires flagging about 30-33% of all cells (24% truly contain a
hazard); alpha = 0.05 requires flagging almost every cell.
