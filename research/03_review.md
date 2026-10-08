# Three YOLIC extensions survive hostile review

None of the YOLIC extension ideas survives peer review as written. Note that `research/02_ideas.md` lists **21 ideas** (A1–A3, P1–P3, S1–S3, E1–E3, R1–R3, N1–N3, C1–C3), not 24, and this review covers all 21. Three survive, each only after merging or reframing:

1. **A leakage-free, multi-seed benchmark** (built on N2) that asks whether YOLIC's GAP+FC head actually beats three alternatives at matched compute: a spatial cell-pooling head, a light segmenter pooled into cells, and a detector trained on real boxes and rasterised to cells.
2. **Layout transfer from cell labels only** (P1 with A1 merged in).
3. **A stress test of conformal miss-rate guarantees under shift** (P2 with R1 merged in).

The other ideas fail for one of three reasons:

- **Already published.** Fixed saliency warps are FOVEA and LZU. Confidence cascades are IDK cascades and NoScope. CRC's own worked example is multi-label miss-rate control. Foundation-model auto-labelling is Autodistill.
- **A trivial baseline probably matches them** (reviewer judgement, not measured). A tuned EMA filter probably ties a GRU, cropping to the cell region probably captures most of the warp gain, and rectifying the test image probably matches pose conditioning. Threshold tuning is provably equivalent to post-hoc logit adjustment.
- **The data can't test the claim.** Cityscapes pitch barely varies, so A2 can't be tested on it.

The base authors' own follow-ups pre-empt E3 ([Selective Multi-Branch, Electronics 2024](https://www.mdpi.com/2079-9292/13/8/1472)) and partly cover P2 and C1 ([Cost-Sensitive YOLIC, MobileCloud 2024](https://doi.org/10.1109/MobileCloud62079.2024.00011)), S1 ([YOLIC labeling, SoftwareX 2026](https://doi.org/10.1016/j.softx.2026.102577)) and R3 ([dual ternary YOLIC, MCSoC 2024](https://doi.org/10.1109/MCSoC64144.2024.00061)). Only survivor 1 fits one RTX 4050 and three months with margin, so start there.

Without a Raspberry Pi, no survivor may claim Pi FPS. The paper should report FLOPs, parameter counts and a laptop-CPU proxy latency, labelled as a proxy.

## Rejection table: eighteen of 21 die or merge

| Idea | Verdict | One-line reason | Closest citation |
|---|---|---|---|
| A1 Layout-agnostic head | Merge-into-P1 | A pooled segmenter is layout-agnostic for free; A1 only matters when no masks exist, which is P1's setting | [FC-CLIP mask pooling](https://arxiv.org/html/2308.02487) |
| A2 Ground-plane cells | Reject | Cityscapes pitch spans 0.038–0.05 rad, so real pose change can't be tested; rectifying the test image is the obvious baseline | [Klinghoffer ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Klinghoffer_Towards_Viewpoint_Robustness_in_Birds_Eye_View_Segmentation_ICCV_2023_paper.pdf) |
| A3 Temporal YOLIC | Reject-unless-reframed | A tuned EMA, k-of-n or hysteresis filter likely ties a GRU; the only label sources are CamVid at 1 Hz and unconfirmed outdoor frame order | [Linardos BMVC 2019](https://arxiv.org/abs/1907.01869) |
| P1 Dense map from cell labels | **Survives** (#2, with A1/E1) | MIL is old, but re-aggregating to unseen layouts from cell labels alone is unreported | [Pinheiro & Collobert CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pinheiro_From_Image-Level_to_2015_CVPR_paper.pdf) |
| P2 Recall-guaranteed thresholds | **Survives** (#3, only as a shift stress test) | Plain CRC on cells repeats CRC's own multi-label worked example; where the guarantee breaks is the only new part | [CRC, ICLR 2024](https://arxiv.org/abs/2208.02814) |
| P3 Automatic layout design | Reject | Changing the layout changes the metric; a perspective grid at equal N is the guessed equal | [Jia et al. CVPR 2012](https://www.icsi.berkeley.edu/icsi/node/4872) |
| S1 Open-vocabulary auto-labels | Reject-unless-reframed | The pipeline is productised; the hazard classes (dent, bump, weed) are where auto-labels fail | [Griffin et al. 2025](https://arxiv.org/html/2506.02359) |
| S2 Coverage soft labels | Merge-into-N2 | One-line change; masking boundary cells is the baseline; label-rule sensitivity becomes one table | [MixPatch](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9221905/) |
| S3 Semi-supervised YOLIC | Reject | Semi-supervised multi-label methods with class-aware thresholds exist, and near-duplicate frames make the label-efficiency curve measure leakage | [CAP NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71510) |
| E1 N-independent head | Merge-into-N2 | A 1×1 conv plus cell pooling is the CAM identity (RoI average pooling); a necessary baseline, not a contribution | [CAM](https://arxiv.org/abs/1512.04150) |
| E2 Foveated warp | Reject | FOVEA and LZU-fixed already do this; crop-to-cell-union plus aspect matching probably captures most of the gain (kept as an input control in #1) | [LZU CVPR 2023](https://arxiv.org/pdf/2303.15390) |
| E3 Uncertainty cascade | Reject | The authors' own Selective Multi-Branch, plus IDK cascades and NoScope, already cover it; worst-case latency gets worse | [Selective Multi-Branch](https://www.mdpi.com/2079-9292/13/8/1472) |
| R1 Cityscapes→ACDC | Merge-into-P2 | Mature benchmark; weather is confounded with a different camera; usable only as the shift source for #3 | [ACDC](https://arxiv.org/abs/2104.13395) |
| R2 City-held-out + TTA | Merge-into-N2 | Cityscapes splits are already city-disjoint; only the outdoor-split leakage test survives | [Cityscapes paper](https://arxiv.org/pdf/1604.01685) |
| R3 Quantization × corruption | Reject-unless-reframed | 8-bit barely amplifies corruption sensitivity; an ICVES 2026 paper with a near-identical title exists (unread) | [Xiao et al. CVPR-W 2023](https://arxiv.org/abs/2304.03968) |
| N1 How GAP+FC localises | Merge-into-N2 | Islam et al. answer it in general form; only a cell-size and border-distance analysis is left | [Islam ICCV 2021](https://arxiv.org/abs/2108.07884) |
| N2 Re-evaluation | **Survives** (#1, reframed as a head/baseline benchmark) | No YOLIC reproduction exists; the likely effects (bugs, leakage, broken baselines) exceed seed noise | [Asselin et al. 2024](https://arxiv.org/abs/2405.06911) |
| N3 Granularity scaling | Reject | Confounded by positive rate; 120–360 runs is over budget; a per-cell-size breakdown comes free in #1 | [DeepLabv3 output-stride table](https://ar5iv.labs.arxiv.org/html/1706.05587) |
| C1 Imbalance losses | Reject | For independent sigmoids, logit adjustment equals threshold tuning; gains are likely within noise | [ASL ICCV 2021](https://ar5iv.labs.arxiv.org/html/2009.14119) |
| C2 Segmentation-teacher KD | Merge-into-N2 | Distillation is incremental; the pooled light segmenter is the baseline that matters | [MobileNetV3 / LR-ASPP](https://arxiv.org/pdf/1905.02244) |
| C3 Cell-graph CRF/GNN | Reject | Structured modules give small gains over strong CNNs; a logit MLP or majority filter likely matches | [DeepLabv3 (dropped DenseCRF)](https://arxiv.org/abs/1706.05587) |

## Ranked survivors and what each must still prove

### 1. A benchmark of cell-presence heads (N2 + E1 + C2 baseline + N1 + R2 + S2)

**Claim, reframed:**

> "Under leakage-free splits, at least three seeds and one shared per-cell metric, we test whether YOLIC's GAP+FC cell classifier beats three alternatives at matched FLOPs and input size: the same backbone with a spatial cell-pooling head, a light segmentation network pooled into cells, and a ShuffleNetV2-class detector trained on real instance boxes and rasterised to cells. We also show where each one fails as cells shrink."

This is the strongest survivor. The paper's own evidence is fragile:

- Each result comes from a single seed.
- The flip augmentation in the code is vertical while the label permutation is horizontal (D1).
- Validation is augmented during checkpoint selection (D2).
- The outdoor split is a random per-frame split of continuous video.
- On Cityscapes People+Vehicle, YOLIC-M2 reaches a macro-F1 of about 0.768 against about 0.831 for YOLOv8-N (brief arithmetic).

No reproduction exists. All 11 citing works found are non-critical, and the only extensions are the authors' own ([S2 citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.imavis.2024.105095/citations?fields=title,year,venue&limit=100)). The missing comparison is cheap. LR-ASPP-MobileNetV3-Small reaches **68.38 mIoU** on Cityscapes val at 2.90B MAdds for a 512×1024 input ([MobileNetV3](https://arxiv.org/pdf/1905.02244)). Scaled to a 224-equivalent input, that is about 0.28B MAdds, the same order as YOLIC-S2's 0.30 GFLOPs (my inference from the notes).

**Strongest remaining objection.** "A re-evaluation of a method with about 10 citations is not a journal contribution." Pure reproductions go to ReScience C or MLRC/TMLR ([MLRC](https://reproml.org/)). The defence is to frame the paper as a protocol and head-design benchmark for the whole class of grid/cell classifiers (UFLD, StixelNet, YOLIC), not as a takedown, and to target IEEE Access. The second objection is that every latency number is a laptop-CPU proxy. A Pi 4B purchase would remove this.

**Baselines it must beat, or must report honestly:**

- YOLIC with the bugs fixed.
- A flatten head in the style of UFLD (1×1 reduction, then flatten, then FC) ([UFLD model.py](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/model/model.py)).
- A 1×1 conv plus a cell-pooling head at stride 16 and stride 32.
- LR-ASPP-MNV3-S trained on Cityscapes and pooled with the same cell rule. Torchvision ships no Cityscapes checkpoint, so it must be trained ([torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)).
- FastestDet or Yolo-FastestV2 (ShuffleNetV2 backbones) trained on real Cityscapes instance boxes at 224 and rasterised to cells ([FastestDet](https://github.com/dog-qiuqiu/FastestDet)).
- BCE with per-class tuned thresholds, standing in for C1.
- A crop-to-cell-union input with matched aspect ratio, standing in for E2.

**Minimum experiment set:**

1. Fix D1, D2, D11 and D12, then re-run YOLIC-M2 with 3 seeds on Cityscapes and the outdoor set.
2. On the outdoor set, compare a random split against a session-based split (from filename timestamps if they turn out to hold order) or a perceptual-hash split, with the same model. Report the size of the gap.
3. On Cityscapes, run the head comparison and the segmenter and detector baselines with 3 seeds. Report AP and threshold sweeps as well as F1 at 0.5.
4. Break per-cell F1 down by cell size and distance from the image border. This is the useful part of N1 and N3.
5. Add one table showing how sensitive the results are to the label rule (≥1 pixel vs ≥k pixels; S2).
6. Report parameters, FLOPs, and single-thread ONNX Runtime laptop latency with warm-up and repeat counts.

**Feasibility.** All data exists:

- Cityscapes needs registration and about 11 GB.
- The outdoor set is public on Kaggle (about 13 GB, DbCL licence).
- Cityscapes instance masks give real boxes. The outdoor set has only cell labels, so the detector comparison is Cityscapes-only.

The brief estimates 3–5 h per 150-epoch run. That is likely shorter once 224 px images and labels are cached and the per-epoch evaluation on the training set is dropped. The total is roughly 5 YOLIC-family configurations × 3 seeds × 2 datasets ≈ 30 runs, plus 3 detector runs at the brief's 10–20 h each. This is a few hundred GPU hours (my arithmetic), which fits in three months. RAM (7.6 GB) is the real constraint: use 2–4 workers and cached tensors. Before claiming errors, contact the authors about D1–D3.

### 2. Layout transfer from cell labels alone (P1 + A1, with E1 as the averaging ablation)

**Claim, reframed:**

> "A dense head trained only through cell-level presence labels, with MIL aggregation, can be re-aggregated to unseen cell layouts. On Cityscapes we measure how much of the accuracy of retraining on the new layout's labels it recovers, as the new layout moves from unions of training cells, to shifted cells, to finer cells."

The defensible niche is narrow but real. The notes found **no prior work that re-aggregates a cell detector to a different layout** (limited searching), and no 2024–2026 layout-conditioned cell classifier for edge devices. The method itself is not new: it is MIL with max, LSE or noisy-OR pooling ([Pinheiro & Collobert](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pinheiro_From_Image-Level_to_2015_CVPR_paper.pdf), [WILDCAT](https://openaccess.thecvf.com/content_cvpr_2017/html/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.html)), and patch-label weak segmentation in histopathology ([Han et al. MedIA 2022](https://arxiv.org/abs/2110.08048)). The paper must claim the capability and the failure analysis, not the method.

**Strongest remaining objection.** "On Cityscapes, a segmenter trained on pixel masks wins. Where masks don't exist (the outdoor set), you can only score union layouts exactly, and those are trivial." The honest framing is "simulate on Cityscapes, demonstrate on outdoor", with finer layouts reported as a measured failure. A second objection is that a stride-8/16 dense head raises FLOPs, which erodes YOLIC's tiny-compute selling point, so the paper must report FLOPs.

**Baselines:**

- Overlap transfer: an L2 cell is positive if any overlapping L1 cell is predicted positive. This needs no model and is recall-optimal.
- Average-pooling MIL, i.e. E1.
- A comparison of max, LSE and noisy-OR pooling.
- A segmenter trained on pixel masks, as the upper bound.
- An oracle retrained on L2 labels.

**Minimum experiments:**

1. Train on YOLIC's 256-cell Cityscapes layout (L1).
2. Test on four kinds of L2 layout: unions, half-cell shifts, 2× finer cells, and random polygons. Report F1/AP as a fraction of the oracle, with 3 seeds.
3. Run the analysis of bag-size and noisy-OR saturation (bigger cells mean more OR terms).
4. On the outdoor set, score the union layout quantitatively and show the rest qualitatively.

**Feasibility.** Relabelling Cityscapes for any layout is free. Compute is about 10–20 runs. A hand-labelled outdoor L2 subset is optional, but it brings back some annotation work.

### 3. When cell-wise miss-rate guarantees break (P2 + R1)

**Claim, reframed:**

> "Conformal risk control gives a cell-wise hazard detector a finite-sample miss-rate bound per class and per near/far cell group. We measure how far realised miss rates exceed the target under city shift, adverse weather and temporally correlated calibration data, and what precision each guarantee costs."

Plain CRC on YOLIC is CRC's own §3.2 example: FNR control for multi-label MS-COCO, over 1000 trials ([CRC](https://arxiv.org/pdf/2208.02814)). Its threshold differs from a naive empirical threshold only by B/(n+1), about 0.004 at n ≈ 250. What isn't published is how the guarantee degrades on a cell detector in three settings:

- **Weather shift:** calibrate on Cityscapes val, test on ACDC conditions.
- **Near/far groups:** group-conditional control per cell region, where the closest prior work is [CRA 2025](https://arxiv.org/abs/2504.07611).
- **Correlated video:** calibration on correlated frames, where β-mixing theory predicts a penalty ([Oliveira et al. JMLR 2024](https://jmlr.org/papers/v25/23-1553.html)).

**Strongest remaining objection.** "This measures known theory: CRC already provides a total-variation bound under shift." There are also data limits:

- ACDC has only about 100 labelled val images per condition, so confidence intervals will be wide.
- ACDC's camera is a GoPro, mounted in front of the windshield at night but behind it in fog, rain and snow, which confounds weather with geometry ([ACDC](https://ar5iv.labs.arxiv.org/html/2104.13395)).
- Cost-Sensitive YOLIC already makes near cells matter more ([MobileCloud 2024](https://doi.org/10.1109/MobileCloud62079.2024.00011)).

**Baselines:**

- Fixed 0.5 threshold.
- Naive empirical threshold.
- CRC.
- LTT, which gives a high-probability guarantee ([LTT](https://arxiv.org/abs/2110.01052)).
- Distance-weighted training as a re-implementation of Cost-Sensitive YOLIC, whose weighting details are unread.
- The same calibration applied to the segmenter pooled into cells.

**Minimum experiments:**

1. Run 1000 random calibration/test splits of Cityscapes val, reporting the distribution of realised FNR.
2. Run group-conditional (near/far) control.
3. Calibrate on Cityscapes and test per ACDC condition. Use ACDC's annotated normal-condition reference images as the same-camera control.
4. On the outdoor set, compare sequence-aware against random calibration. This needs session IDs.

**Feasibility.** Compute is trivial because everything is post-hoc, and there is zero inference cost. The ACDC licence and registration terms are unread. The layout must be re-fitted to ACDC's camera framing.

Realistically, one paper fits in three months. Survivors #2 and #3 reuse #1's models, so treat them as follow-ups or as a second paper.

## Per-idea notes

**A1, layout-agnostic head**
- *Prior art:* mask pooling plus a shared classifier is standard ([FC-CLIP](https://arxiv.org/html/2308.02487), [Region-Based Representations Revisited 2024](https://arxiv.org/abs/2402.02352)). There is no author follow-up, and no paper training on random layouts was found.
- *Baseline:* a light segmenter aggregated per cell is layout-agnostic by construction at about 0.07B MAdds at 224 (notes' scaling). It likely matches A1 whenever pixel masks exist.
- *Failure mode:* at stride 32, several 7×3.5 px upper cells share one feature location and get identical features. The fix (stride 8/16) adds compute. The position embedding reintroduces layout dependence.
- *Data:* Cityscapes only.

**A2, ground-plane cells**
- *Prior art:* BEV occupancy ([PON](https://arxiv.org/abs/2003.13402), [Lu 2019](https://arxiv.org/abs/1804.02176)); camera-conditioned nets ([CAM-Convs](https://arxiv.org/abs/1904.02028)); viewpoint robustness (Klinghoffer: a 10° pitch cut gave a 17% IoU drop); rotation augmentation ([3DRot](https://arxiv.org/abs/2508.01423)).
- *Baseline:* warp the test image back with H = K R⁻¹ K⁻¹ and run the unchanged model. This is likely strong for pitch and roll.
- *Data:* Cityscapes pitch is 0.038–0.05 rad and calibrated per session ([Butt & Taj](https://arxiv.org/pdf/2211.12432)), so it has no real pose variation. Height change is not a homography, and a projected footprint is not occupancy.
- *Selling point:* it needs per-frame pose at runtime, i.e. an IMU the brief excludes.

**A3, temporal YOLIC**
- *Prior art:* [GRFP](https://arxiv.org/abs/1612.08871), [CVPR 2025 similarity propagation](https://arxiv.org/abs/2503.15676), and ECCV 2020 temporal-consistency distillation. In [Linardos](https://arxiv.org/abs/1907.01869), EMA ≈ ConvLSTM.
- *Baseline:* tuned EMA, k-of-n or hysteresis filtering, with zero parameters. A GRU likely ties on F1.
- *Failure modes:* CamVid's 1 Hz labels give one supervised step per 30 frames. Smoothing adds detection delay for sudden hazards. A GAP vector has no spatial map to warp.
- *Data:* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) has 701 labelled frames; the raw video is 7.42 GB of MXF needing codecs. Outdoor frame order is only a guess from filenames. The only surviving reframe is delay-vs-flicker curves against tuned filters, which is still thin.

**P1, dense map from cell labels** (survivor #2)
- *Prior art:* MIL from 2015–2017; [CS-SUNet](https://arxiv.org/abs/2207.08022) shows coarse-label training needs smoothness regularisation.
- *Baseline:* overlap transfer; on Cityscapes, a fully supervised segmenter.
- *Failure modes:* peaky max/noisy-OR maps, noisy-OR saturation with cell area, finer layouts the loss never constrained, and cells smaller than the feature stride.
- *Capability:* genuinely new if it clearly beats overlap transfer.

**P2, recall-guaranteed thresholds** (survivor #3)
- *Prior art:* [CRC](https://arxiv.org/abs/2208.02814) §3.2 covers multi-label FNR; [SeqCRC 2025](https://arxiv.org/abs/2505.24038) covers detection; Cost-Sensitive YOLIC covers "safety regions".
- *Baseline:* an empirical threshold on val, nearly identical in practice.
- *Failure modes:* rare far-cell classes drive the threshold toward 0. Per-image FNR is not pooled recall. Temperature scaling doesn't change a single-class threshold.
- *Selling point:* kept, at zero cost.

**P3, automatic layout design**
- *Prior art:* learned pooling regions (Jia 2012), [Learning to Zoom](https://openaccess.thecvf.com/content_ECCV_2018/html/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.html), non-uniform occupancy grids.
- *Baseline:* a perspective (equal-ground-area) grid at equal N. My guess is that greedy search rediscovers it.
- *Fatal flaw:* per-cell F1 depends on the layout, so easy cells game it. It needs a layout-independent pixel-level target.
- *Compute:* proxy training inside the search loop is expensive and noisy.

**S1, open-vocabulary auto-labelling**
- *Prior art:* [Autodistill](https://github.com/autodistill/autodistill); the authors' SAM labelling tool (SoftwareX 2026); [Griffin 2025](https://arxiv.org/html/2506.02359), where BDD mAP50 was 0.434 with human labels vs 0.271 with auto-labels.
- *Baseline:* YOLIC trained on 1–5% of human labels. On a leaky split this may beat 20k auto-labels.
- *Failure mode:* no zero-shot detection benchmark exists for bump, dent or weed. The unknown human labelling policy confounds F1.
- *Compute:* Grounding DINO memory on 6 GB is unverified (an 8.5 GB snippet conflicts). YOLO-World is about 10× faster. The best reframe, a per-class auditability study, is workshop-level.

**S2, coverage soft labels**
- *Prior art:* proportion labels for mixed patches ([MixPatch](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9221905/)); ≥10%-area grid rule ([COLD-CI](https://arxiv.org/pdf/2606.20767)); GFL/QFL soft targets (from memory).
- *Baseline:* binary labels at threshold k, or masking boundary cells. Masking likely matches.
- *Failure modes:* soft targets shrink outputs for thin or far objects, so they are missed at 0.5. Changing the evaluation rule can fake a win.
- *Data:* Cityscapes only. Keep it as one table in #1.

**S3, semi-supervised YOLIC**
- *Prior art:* [CAP](https://neurips.cc/virtual/2023/poster/71510), [D2L ECCV 2024](https://arxiv.org/pdf/2407.18624), [DiCaP 2025](https://arxiv.org/pdf/2511.20225). Fixing the flip is a bug fix (D1).
- *Baseline:* a tuned supervised-only model ([Oliver et al.](https://arxiv.org/pdf/1804.09170)) and self-supervised pretraining.
- *Failure modes:* with a random frame split, a 5% labelled subset already covers every scene. Strong geometric views break the pixel-to-cell map. Rare classes suffer confirmation bias.
- *Data:* needs sequence IDs (unknown) or Cityscapes trainextra (unverified).

**E1, N-independent head**
- *Prior art:* FCN, the YOLOv2 conv head ([arXiv](https://arxiv.org/abs/1612.08242)), and the CAM identity P·(F·W) = (P·F)·W.
- *Claim check:* the head shrinks from about 1.31M to about 5k parameters, but the backbone (about 2.2M) dominates. Total size drops about 35–40% and FPS barely moves (notes' arithmetic).
- *Failure modes:* the same stride/cell-size collision as A1, and per-cell priors survive only through the bias.
- *Role:* a mandatory baseline in #1.

**E2, foveated warp**
- *Prior art:* [FOVEA](https://arxiv.org/abs/2108.12102) S_D prior; LZU-fixed on Cityscapes gains +0.3 to +2.4 mIoU over uniform downsampling.
- *Baseline:* cropping to [512:1536]×[320:1024] gives about 2× horizontal and 1.45× vertical resolution for free; then match the aspect ratio at equal FLOPs.
- *Failure modes:* it shrinks the safety-critical near cells and is tied to one camera. Remap cost on a Pi is unmeasured.

**E3, uncertainty cascade**
- *Prior art:* the authors' Selective Multi-Branch and the [ICCE-Asia 2022 multi-branch net](https://doi.org/10.1109/ICCE-Asia57006.2022.9954875); [IDK cascades](https://www.auai.org/uai2018/proceedings/papers/212.pdf); [NoScope](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf). [Jitkrittum 2023](https://arxiv.org/abs/2307.02764) finds max-confidence deferral hard to beat.
- *Baseline:* a single model at the cascade's average latency, or the big model every k frames.
- *Selling point:* worst-case latency becomes t1 + t2 ([Daghero 2026](https://arxiv.org/abs/2604.26470)). Two models must fit in Pi RAM. Skip rates are low on a moving camera. Without a Pi, p99 latency can't be measured.

**R1, Cityscapes→ACDC**
- *Prior art:* ACDC itself and [CoTTA](https://ar5iv.labs.arxiv.org/html/2203.13591), where BN-adapt *hurt* SegFormer, plus MIC, HRDA and Refign.
- *Baseline:* a pooled light segmenter evaluated identically.
- *Failure modes:* camera and mount differ from Cityscapes, the void/hood rule (D12) mislabels the bottom rows, and only 406 labelled val images exist.
- *Role:* the shift source for #3.

**R2, city-held-out + TTA**
- *Prior art:* the Cityscapes splits are city-disjoint, and its subsets vary by less than 1.5 points ([Cityscapes](https://arxiv.org/pdf/1604.01685)). Large gaps appear only across continents: 64.6 drops to 35–39 mIoU in [Chen 2017](https://arxiv.org/pdf/1704.08509).
- *Failure mode:* TENT and BN adaptation fail at batch size 1 on correlated streams ([NOTE](https://arxiv.org/pdf/2208.05117)).
- *What survives:* the outdoor random-vs-sequence split, moved into #1.

**R3, quantization × corruption**
- *Prior art:* [Xiao 2023](https://arxiv.org/abs/2304.03968) found 8-bit ResNet18 mean corrupted accuracy of 31.7 vs 32.78 for FP32. [Karimov 2025](https://arxiv.org/html/2508.19600v3) found static INT8 more noise-sensitive, and degradation-aware calibration no better than clean calibration. A Cityscapes corruption benchmark already exists ([Kamann & Rother](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kamann_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models_CVPR_2020_paper.pdf)). The [ICVES 2026 repo](https://github.com/AymenBOUGUERRA/Quantization-and-Corruption-Robustness-in-Deployed-Road-Scene-Perception) is unread.
- *Failure modes:* laptop x86 INT8 is not ARM/ncnn INT8. YOLIC's QAT already costs 1–2 F1, so any amplification sits within noise.
- *Reframe:* a negative result for noise-only effects, and only after reading the ICVES paper.

**N1, how GAP+FC localises**
- *Prior art:* [Islam ICCV 2021](https://arxiv.org/abs/2108.07884) (position is encoded channel-wise and survives GAP), [IJCV 2024](https://arxiv.org/abs/2101.12322), [Lin 2022](https://arxiv.org/abs/2206.01202) (the padding pattern is a learning artefact).
- *Confound:* with a fixed camera, content correlates with position. Swapping padding at train time breaks the ImageNet weights.
- *Role:* the cell-size and border-distance breakdown in #1.

**N2, re-evaluation** (survivor #1)
- *Prior art:* no reproduction exists. Near-duplicate re-evaluations have precedent ([Barz & Denzler](https://arxiv.org/abs/1902.00423)), and the split is the largest source of variance ([Bouthillier 2021](https://arxiv.org/pdf/2103.03098)).
- *Noise:* Cityscapes margins of under 1 F1 (0.8202 vs 0.8134) may vanish with seeds, while the bug and leakage effects likely won't.
- *Data:* real boxes exist only on Cityscapes.

**N3, granularity scaling**
- *Prior art:* the [DeepLabv3](https://ar5iv.labs.arxiv.org/html/1706.05587) output-stride table runs from 75.18 mIoU (stride 8) to 20.29 (stride 256). No cell-layout scaling study was found.
- *Confound:* smaller cells have lower positive rates and more sliver label noise.
- *Compute:* 120–360 runs is over budget.

**C1, imbalance losses**
- *Prior art:* ASL beats CE by +2.6 mAP on COCO in single runs. Logit adjustment exists, as does Cost-Sensitive YOLIC.
- *Baseline:* BCE with per-class tuned thresholds. For sigmoids, σ(z−τ) > 0.5 ⇔ z > τ, so it is equivalent.
- *Noise:* gains under 1 mAP are likely noise, and focal loss may amplify sliver-label noise.
- *Role:* a control row in #1.

**C2, segmentation-teacher distillation**
- *Prior art:* the authors' KD road-risk paper ([DASC 2020](https://doi.org/10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00032)), plus structured KD and CWD (from memory).
- *Baseline:* the pooled segmenter itself. Teachers are available: [mmseg SegFormer](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/README.md) B0 reaches 76.54 mIoU.
- *Failure modes:* on Cityscapes train, the teacher's output ≈ ground truth. The 19 classes lack bump, dent, cone and zebra. Mean-pooling gives coverage, not presence.

**C3, cell-graph CRF/GNN**
- *Prior art:* DeepLabv3 dropped DenseCRF; ASL without label correlation beat ML-GCN on VOC (an uncontrolled comparison).
- *Baseline:* a 2-layer logit MLP or a 3×3 majority filter.
- *Noise:* gains are likely under 1 point, and self-defined fragmentation metrics will be discounted.

## What was verified, what was inferred, and what to check first

**Verified.** These sources were opened and tagged [V], [V-full], [V-abs] or VERIFIED in the notes:

- **Author follow-ups**, from abstracts or metadata: Cost-Sensitive YOLIC, dual ternary YOLIC, YOLIC labeling (SoftwareX 2026), Selective Multi-Branch (abstract only), KD 2020, ICCE-Asia 2022, and the citation lists (S2: 11, OpenAlex: 8).
- **Baseline numbers:** the UFLD head code; MobileNetV3 Table 7 (LR-ASPP 68.38/72.36 mIoU); the PyTorch Pi 4 tutorial (quantized MobileNetV2 at 33.7 FPS); Qengineering's Pi 4 table (Yolo-FastestV2 at 18.8 FPS); the FastestDet and NanoDet READMEs; the mmseg SegFormer table; torchvision having no Cityscapes LR-ASPP.
- **Cityscapes facts:** camera.json contents, the pitch range ([Butt & Taj](https://arxiv.org/pdf/2211.12432)), the city-disjoint splits and homogeneity result.
- **Related work and data:** the CRC body (§3.1, §3.2, selection rule); ACDC splits and camera; CoTTA numbers; Xiao 2023 tables; Karimov 2025; FOVEA and LZU numbers; CamVid page and raw-video inventory; Griffin 2025; Linardos; Jitkrittum; Daghero; Islam ICCV 2021 and IJCV 2024 abstracts; DeepLabv3 output-stride table; ASL numbers.
- **Brief items:** the code bugs D1–D19 come from `research/01_brief.md` and were verified there against the code.

**Seen only in snippets or recalled (not verified).** Treat these as provisional:

- **Snippets:** ICVES 2026 (title only; the repo returned 404); Kamann & Rother details; WILDCAT, Pinheiro and Oquab failure analyses; NOTE and SAR; CAP, D2L and DiCaP; Oliver 2018; MixPatch and COLD-CI; IDK cascades and NoScope; the ECCV 2020 temporal-consistency distillation; KITTI-STEP sizes; the Grounding DINO 8.5 GB figure; the PP-LiteSeg variants; the Selective Multi-Branch full text (403).
- **Recalled from memory (not opened in this research), and must be checked before citing:** GFL/QFL, Unbiased Teacher, FixMatch, structured KD, CWD, TENT, BranchyNet, RoI pooling papers.
- **Inference:** every statement that a baseline "likely" or "probably" matches an idea is the reviewers' or my own judgement, not a measured result. This covers EMA vs GRU, crop vs warp, perspective grid vs search, tuned thresholds vs ASL, and segmenter vs YOLIC. So are the FLOP scalings and run-time and run-count estimates.

**Open checks before committing:**

1. Read the ICVES 2026 quantization×corruption paper (it decides R3).
2. Download the outdoor set and test whether filenames (`YYMMDD_HHMMSS…` pattern guessed) give session IDs and frame order. Check whether Kaggle ships split lists. Survivor #1's leakage test and #3's correlation test depend on this.
3. Read ACDC's licence, registration terms and resolution (needed for #3).
4. Check the indoor dataset's licence and size.
5. Get the full texts of Cost-Sensitive YOLIC (are the weights in the loss or only the metric?) and Selective Multi-Branch (does it contain a head ablation that pre-empts part of #1?).
6. Do a dedicated literature pass on weak supervision and MIL pooling for P1, and on grid-based obstacle classification for wheelchairs and visually impaired users.
7. Measure LR-ASPP accuracy at a 224-equivalent input before framing #1. No published number exists.
8. Decide whether to buy a Pi 4B. Without one, publish no FPS claims.
9. Email the authors about D1–D3 and how YOLO outputs were mapped to cells.
