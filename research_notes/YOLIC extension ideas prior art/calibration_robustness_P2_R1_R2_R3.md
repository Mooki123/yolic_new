# Prior art and reviewer baselines for YOLIC calibration and robustness ideas (P2, R1, R2, R3)

Researched 2026-10-01. Idea text: `research/02_ideas.md`. Context: `research/01_brief.md`.

**Tags.** **[V]** means VERIFIED: I opened the source (the PDF text, the arXiv abstract or HTML, or the paper page) and read the claim there. **[S]** means SNIPPET ONLY: the claim comes from a search-result snippet or an aggregator, and I did not open the primary source. **[G]** means GUESSED or my own inference.

---

## P2: Conformal risk control (CRC) for recall-guaranteed per-class thresholds

### Takeaway
Conformal Risk Control already includes, as worked examples, false-negative-rate (FNR) control for **multi-label classification (MS-COCO)** and for **pixel-wise segmentation (polyps)**. Thresholding YOLIC's cell sigmoids to control the miss rate is therefore a direct application, not a new method. The obvious reviewer baseline is to pick on a validation set the threshold whose empirical miss rate is at most α. CRC differs from that baseline only by a B/(n+1) slack, which is about 0.002 for n ≈ 500, so in practice the two are almost the same. The idea only has value if it becomes a careful study of where the guarantee **breaks**: cross-city and adverse-weather shift, video correlation, and per-cell-group conditional coverage.

### Cited Findings
- **[V]** CRC (Angelopoulos, Bates, Fisch, Lei, Schuster) was published at ICLR 2024. It controls the expected value of any monotone loss, and the abstract names FNR control as an example — [arXiv 2208.02814](https://arxiv.org/abs/2208.02814); [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
- **[V]** The introduction sets up exactly the YOLIC situation. A multi-label classifier f: X → [0,1]^K gives the set C_λ(x) = {k : f(x)_k ≥ 1−λ}, and "conformal risk control finds a threshold value λ̂ that controls the fraction of missed classes" — [arXiv PDF 2208.02814](https://arxiv.org/pdf/2208.02814)
- **[V]** §3.1 covers "FNR control in tumor segmentation", with loss L_FNR = 1 − |Y ∩ C_λ(X)|/|Y| over pixels. The data pools polyp datasets (Kvasir, Hyper-Kvasir, CVC-ColonDB, CVC-ClinicDB, ETIS-Larib) — [arXiv PDF 2208.02814](https://arxiv.org/pdf/2208.02814)
- **[V]** §3.2 covers "FNR control in multilabel classification" on MS-COCO (80 classes) with a TResNet, n = 4000 calibration points and α = 0.1. Over 1000 trials the risk had mean 0.0996 and standard deviation 0.0052 — [arXiv PDF 2208.02814](https://arxiv.org/pdf/2208.02814)
- **[V]** The selection rule is λ̂ = inf{λ : (1/(n+1)) Σ ℓ_i(λ) ≤ α − B/(n+1)}. It needs exchangeability and a monotone loss bounded by B. The guarantee holds **in expectation**: E[L_{n+1}(λ̂)] ≤ α. A lower bound shows the procedure is tight to within 2B/(n+1) — [arXiv PDF 2208.02814](https://arxiv.org/pdf/2208.02814)
- **[V]** Under covariate shift the paper gives a weighted (Tibshirani-style) version with known likelihood ratio. Under arbitrary shift it gives a total-variation bound on how far plain CRC degrades (Proposition 3) — [arXiv PDF 2208.02814](https://arxiv.org/pdf/2208.02814)
- **[V]** Learn then Test (Angelopoulos, Bates, Candès, Jordan, Lei) treats risk control as multiple hypothesis testing. Its examples include multi-label classification (FDR control) and instance segmentation (IoU control) — [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)
- **[G]** LTT gives a high-probability guarantee, P(R(λ̂) ≤ α) ≥ 1−δ, over the draw of the calibration set. CRC's guarantee is marginal in expectation. The abstract I opened did not state this explicitly; it is from my background knowledge — [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)
- **[V]** Conformal semantic segmentation (Mossina, Dalmau, Andéol, arXiv April 2024) is a post-hoc conformal method that builds prediction sets for segmentation masks — [arXiv 2405.05145](https://arxiv.org/abs/2405.05145)
- **[S]** CRC has been used for pixel-wise segmentation with a threshold on pixel probabilities to bound the FNR of critical classes, e.g. industrial surface-defect detection (2025) — [arXiv 2504.17721](https://arxiv.org/pdf/2504.17721)
- **[S]** "Conditional Conformal Risk Adaptation" (2025) says plain CRC gives inadequate **conditional** risk control for segmentation and proposes an adaptive score — [arXiv 2504.07611](https://arxiv.org/abs/2504.07611)
- **[V]** Conformal Object Detection by Sequential Risk Control (Andéol, Mossina, Mazoyer, Gerchinovitz; arXiv May 2025, v2 Oct 2025) introduces SeqCRC, which extends CRC to two sequential parameters for detection, and provides a toolkit — [arXiv 2505.24038](https://arxiv.org/abs/2505.24038)
- **[S]** Further 2025–2026 work exists: conformal prediction sets for instance segmentation ([arXiv 2602.10045](https://arxiv.org/pdf/2602.10045)), MultiRisk, i.e. multiple-risk control by iterative score thresholding ([arXiv 2512.24587](https://arxiv.org/pdf/2512.24587)), and conformal coverage for occupancy-map estimation ([arXiv 2607.14906](https://arxiv.org/pdf/2607.14906))
- **[V]** Oliveira, Orenstein, Ramos and Romano (JMLR 25(225), 2024) show that split conformal stays valid for many non-exchangeable processes, such as stationary β-mixing ones, at the cost of a small coverage penalty — [JMLR](https://jmlr.org/papers/v25/23-1553.html)
- **[S]** Conformal prediction under Markovian data (2024) — [arXiv 2407.15277](https://arxiv.org/pdf/2407.15277)
- **[V]** Cityscapes fine annotations come from the 20th frame of 30-frame snippets. Splits are made **at city level**: "a city is completely within a single split". Calibration and test images drawn from val are therefore mostly separate snippets — [Cityscapes paper, arXiv 1604.01685](https://arxiv.org/pdf/1604.01685)

### Inferences
- **[G] This is an application, not a method contribution.** CRC's §3.2 FNR-for-multi-label example matches "per-cell multi-hot sigmoids + threshold". A reviewer at an IEEE journal would read "CRC on YOLIC cells" as a re-use of a published worked example.
- **[G] Metric mismatch to watch.** CRC's FNR is the *per-image* fraction of missed positives, averaged over images; images with no positives get loss 0. YOLIC reports *pooled per-class recall* over all cells, which is a ratio of sums. A guarantee on mean per-image FNR is **not** a guarantee on pooled recall. The note must define the loss as, for example, "fraction of positive People-cells missed in an image" and say which quantity is guaranteed.
- **[G] Reviewer baseline: pick λ on validation so empirical recall ≥ 1−α.** With B = 1 and n ≈ 250 calibration images (half of Cityscapes val), CRC's extra slack is 1/251 ≈ 0.004 in α. The chosen thresholds will be practically identical to the naive empirical choice. The real differences are in the claims:
  - CRC gives a finite-sample guarantee in expectation.
  - LTT gives a high-probability guarantee, which is noticeably more conservative at small n.
  - The naive rule gives no guarantee, but in practice it is only about O(1/n) optimistic.
  The useful experiment is the **distribution of realized test FNR over many random calibration/test splits** for naive, CRC and LTT, as in CRC's 1000-trial protocol.
- **[G] Where the idea can earn novelty, and these are the parts to emphasise:**
  - (a) Calibrate on Cityscapes val, test on ACDC conditions. The total-variation bound predicts violation, and measuring by how much is informative.
  - (b) Group-conditional guarantees per cell region, near versus far. Each group has fewer positives, so variance is larger. CRA (2504.07611) is the closest prior work here.
  - (c) Video correlation on the YOLIC outdoor set. Neighbouring frames shrink the effective n. Calibration has to be split by video or sequence, which the dataset may not support (brief §6 Q5). The β-mixing results of Oliveira et al. are the theoretical reference.
- **[G] Exchangeability on Cityscapes.** The val cities (I believe Frankfurt, Lindau and Münster, from the dataset folder layout; not opened here) are already held out from train. So "calibrate on val cities, test on other val cities" is a mild shift. Using the Cityscapes *test* split is impossible because its labels are withheld.
- **[G] Failure modes:**
  - Rare classes, such as People in far cells, give few positive images. The guaranteed threshold then collapses toward 0, so everything is flagged and precision is trivial.
  - Temperature scaling does not change the ranking, so it does not change CRC's threshold for a single class. It is pointless as a step before per-class CRC.
  - Controlling per-class FNR for M classes separately needs a multiplicity argument (LTT or Bonferroni) if a joint statement is wanted.

### Gaps
- I did not open the full Mossina et al. 2024 paper, so I cannot confirm its datasets (e.g. whether Cityscapes is used). Same for the industrial CRC segmentation paper, CRA and the instance-segmentation paper. These come from abstracts or snippets only.
- I found no paper applying CRC specifically to fixed-grid or cell-wise presence classifiers. A search for "conformal occupancy grid" returned only 3D occupancy and occupancy-map work.
- I did not open the LTT body to quote its exact high-probability statement.

---

## R1: Cityscapes → ACDC adverse-weather transfer of cell labels

### Takeaway
ACDC is a very mature Cityscapes→adverse benchmark, with source-only, unsupervised-domain-adaptation (UDA), test-time-adaptation (TTA) and continual-TTA numbers from 2021 to 2025. Re-running it with a weaker 224-px cell classifier is what a reviewer will see. "First robustness characterisation of cell-wise detection" is a weak contribution unless the cell framing yields a result the segmentation literature cannot, such as per-region (near/far) degradation. Two features of ACDC limit the study:
- The usable labels are **val only**: 406 images, about 100 per condition. Test labels are withheld, and the server scores segmentation, not cells.
- The **camera differs** from Cityscapes (GoPro, 1080p, mounted differently), so weather shift is confounded with geometry shift. ACDC's paired normal-condition images are the control that can separate the two.

### Cited Findings
- **[V]** ACDC has 4,006 adverse images, split evenly across fog, night, rain and snow. Each comes with a normal-condition image of the same scene (8,012 images in total). The adverse images have pixel-level panoptic annotations, plus 1,503 annotated normal counterparts (5,509 annotated in total). Binary masks mark uncertain regions. It was published at ICCV 2021 and extended in T-PAMI 2025 — [arXiv 2104.13395](https://arxiv.org/abs/2104.13395)
- **[V]** Recording used a "1080p GoPro Hero 5 camera, mounted in front of the windshield at nighttime and in normal conditions and behind the windshield in fog, rain, and snow", at 30 Hz, in Switzerland, mostly urban but also highway and rural roads — [ar5iv 2104.13395](https://ar5iv.labs.arxiv.org/html/2104.13395)
- **[V]** Each condition has 400 train, 100 val and 500 test images; night has 106 val. In total there are 1600 train and 406 val images with public annotations, and 2000 test images with withheld annotations. ACDC uses the 19 Cityscapes evaluation classes — [ar5iv 2104.13395](https://ar5iv.labs.arxiv.org/html/2104.13395)
- **[V]** Source-only DeepLabv2 trained on Cityscapes scores, in mIoU: fog 33.5, night 30.1, rain 44.5, snow 40.2 — [ar5iv 2104.13395](https://ar5iv.labs.arxiv.org/html/2104.13395)
- **[V]** Cityscapes, by contrast, was recorded with a 2 MP stereo rig (1/3-in OnSemi AR0331, rolling shutter, 17 Hz) mounted behind the windshield. The authors "deliberately did not record in adverse weather conditions" — [arXiv 1604.01685](https://arxiv.org/pdf/1604.01685)
- **[V]** CoTTA (Wang, Fink, Van Gool, CVPR 2022) evaluates Cityscapes→ACDC continual TTA with SegFormer-B5. Results in mIoU, first round:

  | Method | Fog | Night | Rain | Snow |
  |---|---|---|---|---|
  | Source | 69.1 | 40.3 | 59.7 | 57.8 |
  | BN-stats adapt | 62.3 | 38.0 | 54.6 | 53.0 |
  | TENT-continual | 69.0 | 40.2 | 60.1 | 57.3 |
  | CoTTA | 70.9 | 41.2 | 62.4 | 59.7 |

  The reported means are 56.7 / 52.0 / 52.3 / 58.6. TENT's 52.3 is not the mean of its four first-round numbers (56.65), so the mean is presumably taken over repeated continual rounds. The authors say BN-based methods underperform because SegFormer is mostly LayerNorm — [ar5iv 2203.13591](https://ar5iv.labs.arxiv.org/html/2203.13591). A search aggregator gave different CoTTA numbers (59.3 mean) — [GitHub cotta](https://github.com/qinenergy/cotta). Treat the exact means as version-dependent.
- **[S]** On the Cityscapes→ACDC UDA leaderboard (aggregator): Refign-HRDA 72.1, HALO 71.9, MIC 70.4, HRDA 68.0 mIoU — [hyper.ai SOTA page](https://hyper.ai/en/sota/tasks/domain-adaptation/benchmark/domain-adaptation-on-cityscapes-to-acdc)
- **[V]** Refign (WACV 2023) adapts segmentation to adverse conditions using the paired normal-condition images — [CVF WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Bruggemann_Refign_Align_and_Refine_for_Adaptation_of_Semantic_Segmentation_to_WACV_2023_paper.pdf). Only the title and venue were seen; the PDF was not opened.
- **[S]** Further Cityscapes→ACDC work includes MIC (CVPR 2023), HRDA (ECCV 2022), condition-invariant semantic segmentation (CISS, 2023), and CoDA (2024) — [MIC](https://arxiv.org/pdf/2212.01322), [HRDA GitHub](https://github.com/lhoyer/HRDA), [CISS](https://arxiv.org/html/2305.17349), [CoDA](https://ar5iv.labs.arxiv.org/html/2403.17369)
- **[S]** Licence: search snippets say the ETH data is "only for research purposes, unless stated differently". The ACDC site renders with JavaScript, and I could not read its terms or registration requirement — [ETH CVL datasets](https://vision.ee.ethz.ch/datsets.html), [ACDC site](https://acdc.vision.ee.ethz.ch/)

### Inferences
- **[G] The reviewer's view.** Source-only, BN-adapt, TENT, style augmentation and few-shot fine-tuning on ACDC are all standard. Doing them with MobileNetV2 + GAP + FC at 224 px will give lower absolute numbers and the same qualitative story: night is worst and fog the mildest of the shifts. "First robustness characterisation of cell-wise detection" will read as filling a gap in the YOLIC paper, not as new knowledge. That might pass at a mid-tier venue only if packaged with something the cell framing adds:
  - degradation as a function of cell distance or size;
  - a safety-class miss rate (links to P2);
  - INT8 versus FP32 under real weather (links to R3).
- **[G] The simplest baseline a reviewer will demand.** A lightweight Cityscapes segmentation model (LR-ASPP or Fast-SCNN), pooled into cells and evaluated on ACDC the same way. A public strong model (e.g. SegFormer) pooled into cells serves as an upper reference. If segmentation-pooled-to-cells degrades less than YOLIC, the main finding becomes "use a segmenter".
- **[G] Geometry confound and its control.** YOLIC's Cityscapes cells are absolute pixel boxes in 2048×1024 (brief §2.2). ACDC is 16:9 1080p from a different mount, with an in-front versus behind-windshield difference across conditions. A "re-fitted layout" therefore changes both camera and weather at once. ACDC's annotated **normal-condition reference images** (same camera, same route) allow clean-vs-adverse comparison on the *same* camera. That is the correct control and should be in the design.
- **[G] Data feasibility.** Only 406 labelled val images exist, about 100 per condition. Per-condition, per-region cell statistics will have wide confidence intervals; bootstrap them. Few-shot fine-tuning can use the 1600 labelled train images, then evaluate on val. Holding out part of train for model selection is necessary.
- **[G] Failure modes:**
  - The ego-hood and void handling in YOLIC's labelling (magic index 216, brief D12) will mis-label ACDC's bottom rows.
  - The "≥1 pixel present" rule (D13) interacts badly with ACDC's uncertain-region masks; ignored pixels need a rule.
  - BN adaptation with batch size 1 is unstable (see R2).

### Gaps
- I could not confirm ACDC's exact pixel resolution (1920×1080 is implied by "1080p" but not quoted), its licence text, or its registration requirement.
- I did not find any prior paper applying ACDC to grid-cell or cell-classification models.
- The hyper.ai leaderboard numbers come from an aggregator and were not checked against the papers.

---

## R2: Leave-cities-out generalisation on Cityscapes + TENT/BN test-time adaptation

### Takeaway
The premise is partly false for Cityscapes. **Cityscapes' official splits are already city-disjoint**, so YOLIC's own Cityscapes evaluation on val is already a held-out-city test. The Cityscapes authors also report that performance across geographic and city-size subsets is homogeneous, within 1.5 points. Leave-k-cities-out within Cityscapes will therefore probably show a *small* gap, and TTA has little to recover. Large cross-city gaps exist only across continents. The NTHU Cross-City work found 64.6 mIoU on Frankfurt dropping to 35–39 on Rome, Rio, Tokyo and Taipei. TENT and BN adaptation are known to fail at batch size 1 and on temporally correlated streams, which is exactly the edge-video regime. The leakage concern the idea cites does apply to the **outdoor in-house** set (random per-frame split), not to Cityscapes.

### Cited Findings
- **[V]** Cityscapes split protocol: "We chose not to split the data randomly"; splits are balanced on city size, geography and time of year, and "the data is split at the city level, i.e. a city is completely within a single split". The splits are 2975 train, 500 val and 1525 test images, with test labels withheld — [arXiv 1604.01685](https://arxiv.org/pdf/1604.01685)
- **[V]** The Cityscapes authors trained an FCN on val and tested on test subsets at the extremes of each split characteristic. "With the exception of the time of year, the performance is very homogeneous, varying less than 1.5 % points (often much less)". The end-of-year subset scored 3.8 points better than the whole test set; low-temperature (cloudy) images +4.5, warm (sunny) −0.9 — [arXiv 1604.01685](https://arxiv.org/pdf/1604.01685)
- **[V]** Recording covered 50 cities, "primarily in Germany but also in neighboring countries", in spring, summer and fall. Fine annotations come from 27 cities. All recordings used one stereo rig behind the windshield, recalibrated before each session — [arXiv 1604.01685](https://arxiv.org/pdf/1604.01685)
- **[V]** Chen et al., "No More Discrimination: Cross City Adaptation of Road Scene Segmenters" (ICCV 2017):
  - Dataset: Rome, Rio, Tokyo and Taipei, 1600 image pairs (3200 images) per city, with 100 annotated images per city for evaluation.
  - A dilated-FCN pre-trained on Cityscapes scores 64.6% mIoU on "Cityscapes (Frankfurt)" against Rome 38.2%, Tokyo 39.2% and Taipei 35.1% (Rio's row was garbled in my extraction).
  - The authors note the degradation grows with geographic distance.
  - Adaptation with global plus class-wise alignment brings Rome to 42.1 and Taipei to 38.8.

  — [arXiv 1704.08509](https://arxiv.org/pdf/1704.08509)
- **[V]** BN-statistics adaptation *hurt* on Cityscapes→ACDC with SegFormer-B5 (mean 56.7 → 52.0), and TENT-continual did not help. The authors attribute this to LayerNorm-based architectures — [ar5iv 2203.13591 (CoTTA)](https://ar5iv.labs.arxiv.org/html/2203.13591)
- **[S]** NOTE (NeurIPS 2022): "most existing TTA methods fail dramatically when test samples are temporally correlated (non-i.i.d.)", and for TENT the failure stems from re-computing BN statistics on the test batch — [arXiv 2208.05117](https://arxiv.org/pdf/2208.05117), [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/ae6c7dbd9429b3a75c41b5fb47e57c9e-Paper-Conference.pdf)
- **[S]** SAR (ICLR 2023, "Towards stable test-time adaptation in dynamic wild world") targets mixed shifts, batch size 1 and label shift. A snippet says SAR does best at batch size 1. OpenReview blocked my fetch — [OpenReview PDF](https://openreview.net/pdf?id=g2YraF75Tj)
- **[S]** MemBN (ECCV 2024) is BN with a statistics memory for small-batch, non-i.i.d. TTA — [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04143.pdf)
- **[S]** Test-time adaptation *for quantized networks* (2025) is relevant to edge deployment and intersects with R3 — [arXiv 2508.02180](https://arxiv.org/pdf/2508.02180)
- **[S]** A 2026 survey of continual TTA covers methods and benchmarks — [arXiv 2607.08164](https://arxiv.org/pdf/2607.08164)
- **[G]** Not opened, cited from background knowledge only: TENT (Wang et al., ICLR 2021, [arXiv 2006.10726](https://arxiv.org/abs/2006.10726)) and BN adaptation for corruption robustness (Schneider et al., NeurIPS 2020, [arXiv 2006.16971](https://arxiv.org/abs/2006.16971))

### Inferences
- **[G] The core claim is already answered for Cityscapes.** YOLIC's Cityscapes result (Table 6) was measured on val, which is city-disjoint from train. A leave-k-cities-out protocol mostly reproduces that with less training data. Given the Cityscapes paper's ≤1.5-point homogeneity result, the expected iid-vs-city-held-out gap is small. My guess is a few F1 points at most, possibly within seed noise. A reviewer will ask "what is the gap?", and a null result is likely.
- **[G] Where a real gap exists:**
  - (a) Cross-dataset or cross-continent transfer: Cityscapes→ACDC normal-condition images, Mapillary, or BDD100K. But camera geometry then changes too, which is fatal for fixed pixel cells (see R1).
  - (b) The **outdoor in-house set**, where the random per-frame split almost certainly leaks (brief §3.3). Perceptual-hash pseudo-scene splits are the honest version, and this is the part of R2 worth keeping. It overlaps with idea N2.
- **[G] TTA on the edge.** YOLIC's MobileNetV2 and ShuffleNetV2 use BN, so BN-adapt and TENT apply, unlike with SegFormer. A Pi stream, however, gives batch size 1 with strongly correlated frames, which is exactly NOTE/SAR's failure regime. A reviewer will demand:
  - source-only;
  - BN-adapt with a running average;
  - TENT;
  - at least one robust method (SAR, NOTE or RoTTA);
  - evaluation on a correlated (sequence-ordered) stream, not shuffled batches.
- **[G] Failure mode.** TENT minimises entropy over independent sigmoids, which is a binary-entropy sum. With heavy background imbalance this can drive rare-class logits toward "absent" and lower recall. That is a plausible but untested risk specific to multi-label cell heads.

### Gaps
- I did not find any published leave-k-cities-out experiment *within* Cityscapes with numbers. The only evidence on intra-Cityscapes homogeneity is the Cityscapes paper's own subset analysis.
- I did not confirm the val city names in a primary source.
- I could not open the SAR paper or the TENT and BN-adapt papers in this pass.

---

## R3: Corruption robustness × INT8 quantization, and corruption-aware QAT

### Takeaway
A Cityscapes corruption benchmark already exists (Kamann & Rother, CVPR 2020 / IJCV 2021). Quantized-model robustness benchmarks exist too:
- Xiao et al., CVPR-W 2023, extended as RobustMQ.
- A 2025 YOLO12 PTQ study.
- A 2026 IEEE ICVES paper titled "Quantization and Corruption Robustness in Deployed Road-Scene Perception" (snippet only, but it is very close to R3).

The evidence says **8-bit quantization barely amplifies corruption sensitivity**. On ImageNet-C with ResNet18, 8-bit QAT loses about 1 point of mean corrupted accuracy against FP32 (31.7 vs 32.78) while matching clean accuracy. Large losses appear only at 2–4 bits. The one consistent exception is **noise**: impulse noise for QAT, and Gaussian noise for static INT8 PTQ. The claim "INT8 amplifies YOLIC's corruption sensitivity" is therefore likely false or marginal for blur, JPEG and low light. The "corruption-aware QAT fixes it" part was already tried as degradation-aware calibration, which gave no consistent gain.

### Cited Findings
- **[V]** Xiao, Zhang, Liu and Qin, "Benchmarking the Robustness of Quantized Models" (CVPR 2023 workshop). They evaluate DoReFa, PACT and LSQ QAT at 2, 4, 6 and 8 bits on ResNet18, ResNet50, RegNetX600M and MobileNetV2, with 15 ImageNet-C corruptions. Their conclusion: "lower-bit quantized models … are more susceptible to natural corruptions and systematic noises" — [arXiv 2304.03968](https://arxiv.org/abs/2304.03968)
- **[V]** ResNet18 numbers from the same paper's Tables 1 and 3:

  | Model | Clean accuracy | Natural robustness (mean ImageNet-C accuracy) |
  |---|---|---|
  | FP32 | 71.06 | 32.78 |
  | 8-bit DoReFa | 71.51 | 31.70 |
  | 8-bit PACT | 71.43 | 31.69 |
  | 8-bit LSQ | 71.31 | 31.70 |
  | 4-bit | — | 30.36–30.95 |
  | 2-bit | — | 23.30–26.42 |

  Impulse noise is the most harmful corruption (about 50% average decrease) and brightness the least (about 10%) — [arXiv PDF 2304.03968](https://arxiv.org/pdf/2304.03968)
- **[S]** RobustMQ (the extended version, arXiv 2308.02350): "Increasing the quantization bit-width generally leads to … an increase in natural robustness" — [arXiv 2308.02350](https://arxiv.org/pdf/2308.02350)
- **[V]** Karimov, Imani and Kazakov, "Quantization Robustness to Input Degradations for Object Detection" (arXiv 2508.19600, 2025):
  - Setup: YOLO12 n–x on COCO, comparing FP32, FP16 TensorRT, dynamic UINT8 in ONNX Runtime, and static INT8 TensorRT.
  - Degradations: noise, blur, low contrast, JPEG.
  - Static INT8 was "consistently" more noise-sensitive. Blur and contrast drops were comparable.
  - Degradation-aware calibration (50/50 clean/degraded) gave "robustness performance remarkably similar to standard clean data calibration" for most models.

  — [arXiv 2508.19600v3](https://arxiv.org/html/2508.19600v3)
- **[S]** A GitHub repo describes itself as "Official code for the IEEE ICVES 2026 paper *Quantization and Corruption Robustness in Deployed Road-Scene Perception*". The fetch returned 404, so I could not read the content — [GitHub](https://github.com/AymenBOUGUERRA/Quantization-and-Corruption-Robustness-in-Deployed-Road-Scene-Perception)
- **[S]** A small 2025–2026 study (GTSRB plus Mapillary) asks "Does INT8 quantization hurt a traffic-sign classifier's robustness to corruptions?". It finds PTQ INT8 keeps clean accuracy within 0.6 pp — [GitHub](https://github.com/Charith-Reddy-Pareddy/Robustness-and-Quantized-Inference-for-Autonomous-Traffic-Sign-Analytics)
- **[V]** Kamann & Rother, "Benchmarking the Robustness of Semantic Segmentation Models" (CVPR 2020), extended in IJCV 2021 (129(2):462–483). It covers Cityscapes, PASCAL VOC 2012 and ADE20K with ImageNet-C-style corruptions, nearly 400,000 images, and DeepLabv3+ variants. Models generalise well to noise and blur but not to digital or weather corruptions — [CVF CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kamann_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models_CVPR_2020_paper.pdf); [IJCV](https://link.springer.com/article/10.1007/s11263-020-01383-2). Opened via search-result summary only, not the full PDF.
- **[V]** ImageNet-C defines 15 corruption types (noise, blur, weather, digital) at 5 severities — [arXiv 1903.12261](https://arxiv.org/pdf/1903.12261)
- **[S]** A 2025 paper asks whether synthetic corruptions are a reliable proxy for real-world ones — [arXiv 2505.04835](https://arxiv.org/pdf/2505.04835)

### Inferences
- **[G] The headline claim is likely false or marginal at INT8.** QAT at 8 bits gives about a 1-point absolute (about 3% relative) drop in mean corrupted accuracy for ResNet18, with no clean drop. YOLIC's own QAT costs 1–2 F1 points on clean data (brief §3.4). An "amplification" of that size will sit inside single-seed noise unless several seeds are run. The honest framing is "test whether it holds". The expected answer is "no, except for noise corruptions", which is publishable only as a negative or confirmatory result.
- **[G] A Cityscapes-C already exists** (Kamann & Rother), so building one is not a contribution. Re-use their corruption set and protocol for comparability.
- **[G] Simplest reviewer baseline for the remedy:**
  - (i) FP32 trained with corruption augmentation (AugMix or ImageNet-C-style) followed by standard QAT or PTQ;
  - (ii) PTQ with degradation-aware calibration.

  "Corruption-aware QAT" will probably equal baseline (i). Karimov et al. found (ii) gives no consistent gain. A reviewer will ask whether the gain comes from augmentation alone, which helps FP32 and INT8 equally.
- **[G] Technical failure modes:**
  - PyTorch x86 (fbgemm) or ONNX Runtime INT8 on a laptop is not the ARM/ncnn INT8 YOLIC deploys. Activation-range calibration and kernel rounding differ, so laptop results may not transfer.
  - Noise corruptions inflate activation ranges and saturate INT8 clipping. That is the plausible mechanism for the noise-specific effect, and per-tensor versus per-channel and calibration choice will dominate.
  - The YOLIC QAT code is not released (brief D14), so the QAT baseline is a re-implementation.
- **[G] Most defensible version.** Measure the FP32-vs-INT8 gap per corruption type and severity on cell F1, with seeds and CIs, on real adverse data (ACDC val, see R1) as well as synthetic corruptions. Then report whether noise is the only amplifier. This is small but checkable, and it overlaps with the ICVES 2026 paper, which must be read first.

### Gaps
- I could not open the ICVES 2026 paper or repo, so its datasets, findings and overlap are unknown. It is the closest potential prior work and must be checked before committing.
- I did not open RobustMQ's MobileNetV2 numbers. The MobileNetV2 result, i.e. YOLIC's backbone, may differ from ResNet18, since depthwise convolutions are known to be more quantization-sensitive (background knowledge, unverified).
- I found no study of INT8 corruption robustness for multi-label or cell-wise heads specifically.
