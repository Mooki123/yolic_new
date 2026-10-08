# Supervision ideas S1, S2, S3, C2 for YOLIC-style cell-wise detection: prior art and reviewer baselines

Tag legend:
- **[VERIFIED]** I opened the page or PDF and read the claim there.
- **[SNIPPET]** The claim comes only from a search-engine summary of the linked page. I did not open the page, so treat the numbers as provisional.
- **[GUESSED]** The claim comes from my background knowledge. I did not open the link in this session (I recalled the arXiv ID), so check it before citing.

Context for all four ideas (from `research/01_brief.md` and `02_ideas.md`):
- YOLIC is a classifier: backbone, then GAP, then FC, with N×(M+1) sigmoids.
- The outdoor set has cell-level labels only (104 cells × 11 hazards + road).
- The Cityscapes version builds cell labels from pixel masks with a "≥1 pixel" rule (`cityscapes.py:116-117`).
- The outdoor split is a random per-frame split of continuous video.

Search coverage: 24 tool calls (searches, page fetches and local text extraction). I found no follow-up paper that extends YOLIC itself with auto-labelling, soft labels, semi-supervision or distillation. A search for "YOLIC" 2025 follow-ups returned only the original paper ([arXiv 2307.06689](https://arxiv.org/html/2307.06689v3)). That search was shallow (one query), so it is not proof of absence.

---

## S1: Can foundation-model auto-labels (Grounding-DINO + SAM etc.) replace human cell labels for YOLIC, and is it already done?

### Takeaway
The generic pipeline is already productised and benchmarked: foundation-model pseudo-labels train a small model. Examples are Roboflow Autodistill (GroundedSAM to YOLOv8) and Griffin et al. 2025. The only new element would be the cell-level target and a per-class "what auto-labelling can and cannot do" map on unusual hazard classes.

The strongest published evidence is Griffin et al.: the gap between auto-labels and human labels is small on VOC/COCO-type classes but large on driving data (BDD) and rare classes. That points to the outdoor hazards (bump, dent, weed, column) being exactly where S1 will fail. The main scientific risk is the unknown YOLIC labelling policy.

### Cited Findings
- **[VERIFIED]** Autodistill is an open-source framework that uses a "Base Model" (e.g. GroundedSAM, i.e. Grounding DINO + SAM) with an "Ontology" (text prompt → class) to label unlabelled images. It then trains a "Target Model" such as YOLOv8, advertised as "Images to inference with no labeling" — [autodistill GitHub](https://github.com/autodistill/autodistill); [autodistill-grounded-sam](https://github.com/autodistill/autodistill-grounded-sam); [autodistill-yolov8](https://github.com/autodistill/autodistill-yolov8). A Grounded-SAM-2 plugin (Florence-2 grounding + SAM 2) also exists — [autodistill-grounded-sam-2](https://github.com/autodistill/autodistill-grounded-sam-2). (Verified from search-result repo descriptions; I did not read the code.)
- **[VERIFIED]** Griffin, Gangwar, Sela, Corso, "Auto-Labeling Data for Object Detection", arXiv June 2025 — [arXiv 2506.02359](https://arxiv.org/html/2506.02359):
  - **Labelers and labelling time:** YOLO-World (72.9M params, 197.2 s to label VOC), YOLOE (35.2M, 204.9 s) and Grounding DINO (172.2M, 2,290.3 s to label VOC).
  - **Students:** YOLO11n/s/m/l/x and RT-DETR.
  - **Datasets:** VOC, COCO, LVIS, and BDD (driving).
  - **YOLO11n mAP50, human labels vs YOLO-World auto-labels:**

    | Dataset | Human labels | Auto-labels |
    |---|---|---|
    | VOC | 0.756 | 0.715 |
    | COCO | 0.496 | 0.460 |
    | BDD (driving) | 0.434 | 0.271 |
    | LVIS | 0.087 | 0.059 |

  - **Best configuration:** "YOLOW with a confidence threshold of 0.2 is the single most reliable auto-labeling configuration".
  - **Rare classes:** on LVIS, almost all of the five least-frequent classes get AP50 = 0 (about a 97% drop from frequent to rare classes).
  - **Cost:** auto-labelling all train sets "takes 1.27 hours and costs $1.18 while human labeling takes 6,703 hours".
- **[SNIPPET]** "Robust Object Detection with Pseudo Labels from VLMs using Per-Object Co-teaching" (arXiv Nov 2025) uses OWLv2 as the pseudo-labeller ("current state-of-the-art … widely used for the pseudo-labelling task") and YOLOv5 as the student — [arXiv 2511.09955](https://arxiv.org/pdf/2511.09955). This shows that *noise-robust training on VLM pseudo-labels* is itself an active topic, so a plain pipeline is not novel.
- **[SNIPPET]** Further auto-annotation papers:
  - SAM2Auto (2025): [arXiv 2506.07850](https://arxiv.org/pdf/2506.07850).
  - Empirical study of SAM with Grounding-DINO text prompts for automated annotation: [ResearchGate 381771169](https://www.researchgate.net/publication/381771169_Segment_Anything_Model_for_automated_image_data_annotation_empirical_studies_using_text_prompts_from_Grounding_DINO).
  - Roboflow Grounded-SAM-2 auto-label notebook: [Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/grounded-sam-2-auto-label.ipynb).
- **[SNIPPET]** Grounding DINO: 52.5 AP zero-shot on COCO and a 26.1 mean AP on ODinW zero-shot, ECCV 2024 — [arXiv 2303.05499](https://arxiv.org/html/2303.05499v5); [GitHub](https://github.com/idea-research/groundingdino). The low ODinW number, a benchmark of "in-the-wild" niche domains, is the relevant signal for unusual classes.
- **[SNIPPET]** Road-damage VLM work: RoadBench / RoadCLIP (2025) reports 78.6% *zero-shot classification accuracy* on its benchmark after domain-specific training. It is a CLIP-style recognition model, not a detector — [arXiv 2507.17353](https://arxiv.org/abs/2507.17353). An open-vocabulary crack detector paper also exists — [Applied Sciences 2025](https://doi.org/10.3390/app151910350) — as does pavement-condition assessment with VLMs — [arXiv 2604.08212](https://arxiv.org/html/2604.08212). **I found no published zero-shot *detection* benchmark for "speed bump", "pothole" or "dent" with Grounding DINO / OWLv2 / YOLO-World / Florence-2.** The fact that domain-specific training was needed in RoadBench suggests generic VLMs are weak here (inference).
- **[SNIPPET]** Grounding DINO-T speed and memory, as reported by search summaries:
  - 9.4 FPS in PyTorch and 42.6 FPS with TensorRT on an A100.
  - 1.1 FPS on Orin NX.
  - "~8.5 GB" GPU memory.

  Sources: [Grounding DINO 1.5 paper, arXiv 2405.10300](https://arxiv.org/pdf/2405.10300); [mmdetection GroundingDINO README](https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/README.md); [GitHub issue #50 on speed](https://github.com/IDEA-Research/GroundingDINO/issues/50). I could not tell which source the 8.5 GB figure comes from or under what batch/resolution. Treat it as unreliable.
- **[GUESSED]** Other relevant foundation detectors, not opened:
  - OWLv2: "Scaling Open-Vocabulary Object Detection", [arXiv 2306.09683](https://arxiv.org/html/2306.09683v2.pdf). This link did appear in my search results.
  - YOLO-World: arXiv 2401.17270.
  - Grounded-SAM tech report: arXiv 2401.14159.
  - Florence-2: arXiv 2311.06242.

### Inferences
- **Already done?** Yes for the generic pipeline (foundation model → pseudo-labels → small detector): Autodistill, Griffin et al. 2025, and co-teaching on VLM labels. S1's novelty would be limited to:
  - (a) the cell-level output (rasterising masks to cells is trivial);
  - (b) a per-class auditability study on non-COCO hazard classes, judged against an existing human-labelled cell dataset.

  (b) is a reasonable small-venue or workshop contribution, not a strong IEEE journal contribution on its own.
- **Reviewer's simplest baselines:**
  1. Run the auto-labeller itself at test time and rasterise to cells (the teacher ceiling: "why distil at all?"). The answer is speed, which is already well understood.
  2. YOLIC trained on human labels (the upper bound).
  3. YOLIC trained on a *small* human-labelled subset, e.g. 1–5% of frames. With near-duplicate video frames, a few hundred human-labelled frames may beat 20k auto-labelled ones. That would sink the "zero-label" claim.
  4. YOLO-World or OWLv2 as a cheaper labeller. Griffin et al. found YOLO-World at a 0.2 confidence threshold the most reliable.
- **Expected per-class behaviour (unverified hypotheses):**
  - Likely to work: "vehicle", "traffic cone", "traffic sign", "person/creature", possibly "zebra crossing" and "fence".
  - Likely to fail or need humans: "bump", "dent", "column", "weed", "wall".
    - "Dent" and "bump" are ambiguous referents in natural language.
    - "Wall" and "weed" are stuff-like, and box detectors are weak on them.

  The BDD gap (0.271 vs 0.434) and the LVIS rare-class collapse in Griffin et al. support this pessimism.
- **The biggest validity risk (agree with the brief):** the YOLIC human labels follow an unknown policy, e.g. what extent of a dent or weed patch counts, and whether a far-away vehicle counts. A low F1 then mixes three things: detector failure, label-policy mismatch, and "≥k pixel" threshold mismatch. One mitigation is to hand-audit about 200 cells per class to estimate the policy-disagreement rate.
- **Compute on a 6 GB RTX 4050 (estimate, not measured):**
  - Grounding DINO-T in PyTorch at about 2–5 FPS, scaled down from the A100's 9.4 FPS, gives roughly 1–3 h for 20k frames at 848×480.
  - SAM ViT-B adds an encoder pass per frame, roughly another 1–2 h.
  - With fp16 and batch size 1 it plausibly fits in 6 GB. The "8.5 GB" snippet conflicts with this, so test it.
  - Text prompts with 11 classes need one forward pass per prompt chunk (Grounding DINO has a 256-token text limit), so cost may double.
  - YOLO-World is about 10× faster than Grounding DINO for labelling (197 s vs 2,290 s on VOC in Griffin et al.).
- **Failure modes:**
  - Prompt sensitivity and synonym choice ("speed bump" vs "bump" vs "speed hump").
  - Mask bleed from SAM making stuff classes over-cover cells, which interacts with the "≥1 pixel" rule. Use a k-pixel or coverage threshold here (links to S2).
  - Hallucinated detections of absent classes. Open-vocabulary detectors tend to fire on the most similar object; the co-teaching paper above targets exactly this noise.
  - Domain: campus scooter footage at 848×480 is low-resolution for small cones and signs.
- **Data feasibility:** high. The outdoor frames are public (Kaggle, DbCL licence per the brief), and no new labels are needed. Evaluation, however, inherits the leaky random split.

### Gaps
- No published numbers found for zero-shot Grounding DINO / OWLv2 / YOLO-World / Florence-2 *detection* of "speed bump", "pothole", "dent", "weed" or "column". I could not verify the claim either way; it needs a pilot run.
- I did not open Griffin et al.'s per-class BDD table, so I don't know which driving classes failed (e.g. traffic sign vs car).
- I found no specific "Segment Any Road" paper or a SAM-pseudo-label-for-Cityscapes paper in my searches. That literature (e.g. SAM-assisted pseudo-labels for domain-adaptive segmentation) likely exists but is unconfirmed here.
- Actual RTX 4050 throughput and memory for Grounding DINO-T + SAM-B: not found and not measured.

---

## S2: Coverage-fraction soft cell labels vs the "≥1 pixel" binary rule: novel or just an ablation?

### Takeaway
Soft or proportion-based labels for mixed patches are established practice in histopathology patch classification. Soft IoU/quality targets are standard in detection classification heads (GFL/QFL, background knowledge only). "Use the mean of the mask in the cell as the target" is a one-line change.

A reviewer would almost certainly call it an ablation or design choice, not a contribution. The threshold-k ablation is useful content inside a larger paper but is not publishable on its own. Its value would be as part of a paper that also fixes YOLIC's evaluation, or the per-cell "coverage" output as a capability, and even that is thin.

### Cited Findings
- **[SNIPPET]** Histopathology: "proportion-based soft labeling methods are used to define ground-truth labels for mixed-patches" — [MixPatch, Diagnostics 2022 / PMC9221905](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9221905/); [MDPI version](https://www.mdpi.com/2075-4418/12/6/1493).
- **[SNIPPET]** BreastPathQ patches are labelled with a *percentage cellularity* (a continuous target), not a binary one — via search summary of [arXiv 2409.13720](https://arxiv.org/html/2409.13720) and related. Some pipelines instead *discard* patches with tumour fraction strictly between 0% and 10% as ambiguous. This is the "drop boundary cells" alternative a reviewer would also ask about.
- **[SNIPPET]** Remote-sensing grid labelling uses area thresholds, e.g. keeping grid cells with "at least 10% of labelled area" — [COLD-CI, arXiv 2606.20767](https://arxiv.org/pdf/2606.20767).
- **[SNIPPET]** A patent describes soft labels that encode "how much area the object occupies in each candidate bounding box" for detection classifiers — [USPTO 12468940](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12468940). This shows the area-fraction soft-label idea even appears in patents.
- **[GUESSED]** Closest ML analogues, not opened:
  - Generalized Focal Loss / Quality Focal Loss (Li et al., NeurIPS 2020, arXiv 2006.04388): uses soft IoU-valued classification targets with a BCE-style loss.
  - VarifocalNet (arXiv 2008.13367).
  - Label smoothing (Szegedy et al. 2016; Müller et al., "When does label smoothing help?", arXiv 1906.02629).
  - Learning from label proportions (LLP).
  - Fractional-cover regression in remote sensing (sub-pixel unmixing).
  - Soft or probabilistic occupancy grids in robotics.

  All of these make "soft area target" a known technique.
- **[VERIFIED, from repo]** YOLIC's Cityscapes labels are "≥1 pixel present" with no area threshold (`cityscapes.py:116-117,137-138`, brief §2.1 D13). The outdoor set has no masks, so S2 can only be *trained* on Cityscapes-like data.

### Inferences
- **Is it trivial?** Yes, mechanically: target = mean(mask_k within cell) and loss = BCE with soft targets. A reviewer would see it as an ablation unless it comes with something non-obvious, for example:
  - (a) a demonstrated mismatch between the training rule and the evaluation rule (training with soft targets, evaluating presence at threshold k);
  - (b) calibration analysis;
  - (c) a coverage output used downstream (e.g. "fraction of cell blocked" for path planning);
  - (d) evidence that the GAP+FC head cannot learn small-coverage cells. This connects to the localisation bottleneck in the brief §4.2.
- **Is the "≥k pixel threshold" ablation publishable on its own?** Unlikely. It is a dataset-construction parameter. It does matter for *fair comparison*, because YOLO-vs-YOLIC numbers change with k. So it belongs in a benchmark or re-evaluation paper (idea N2) as one table.
- **Evaluation caveat that can create a fake win:** if training uses soft coverage but evaluation keeps the ≥1-pixel binary GT, soft training will *lower* recall on 1-pixel cells by design. If evaluation switches to ≥k, the comparison is no longer like-for-like with the paper. Report both, plus threshold-free metrics (AP per class).
- **Reviewer baselines:**
  1. Binary labels at threshold k for several k values (the obvious "you could have just thresholded" baseline).
  2. Label smoothing at a constant ε.
  3. Ignoring or masking boundary cells (0 < coverage < τ) in the loss.
  4. Soft targets with an identical evaluation protocol.

  Baseline 3 is very likely to match soft labels in practice (inference, not verified).
- **Failure modes:**
  - The soft-target BCE minimum sits at the coverage value, so outputs become small for thin objects (poles, people far away). A 0.5 threshold then misses them, and recalibration or per-class thresholds are needed (idea P2).
  - Stuff classes ("Other", road) dominate coverage statistics.
- **Data feasibility:** Cityscapes only; trivial compute.

### Gaps
- I found no paper specifically on soft-coverage targets for *fixed grid / cell* presence classification in driving. The closest are histopathology patches, remote-sensing grids and detection soft-IoU targets. Absence is not proven: my search was one query.
- I did not open the GFL/QFL or label-smoothing papers in this session.

---

## S3: Semi-supervised YOLIC (FixMatch / mean teacher with correct flip-equivariant cell permutation): prior art, baselines, and the near-duplicate-frame problem

### Takeaway
Semi-supervised *multi-label* classification is an active 2023–2025 area with class-aware thresholds: CAP (NeurIPS 2023), D2L/MAT (ECCV 2024), CBSA (2024), DiCaP (2025). YOLIC's output is exactly a multi-label vector, so S3 is "apply SSML to a multi-label head", plus flip equivariance, which is standard in semi-supervised detection and segmentation.

Reviewers will demand:
- a properly tuned supervised baseline (Oliver et al. 2018);
- a self-supervised-pretraining baseline;
- leakage-free splits.

On continuous-video frames with a random split, the labelled 5–20% subset already covers almost every scene. Two consequences follow: SSL gains will be small or illusory, and the label-efficiency curve will mostly measure leakage.

### Cited Findings
- **[SNIPPET]** CAP, "Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning" (NeurIPS 2023): uses class-aware thresholds so the pseudo-label distribution matches the true class distribution — [NeurIPS poster](https://neurips.cc/virtual/2023/poster/71510); [paper PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/5195825ee60d7efc1e42b7f3f3137040-Paper-Conference.pdf).
- **[SNIPPET]** "Dual-Decoupling Learning and Metric-Adaptive Thresholding for Semi-Supervised Multi-Label Learning" (ECCV 2024): metric-adaptive thresholds estimated on labelled data — [arXiv 2407.18624](https://arxiv.org/pdf/2407.18624); [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72943-0_25).
- **[SNIPPET]** "Context-Based Semantic-Aware Alignment for Semi-Supervised Multi-Label Learning" (Dec 2024) — [arXiv 2412.18842](https://arxiv.org/pdf/2412.18842).
- **[SNIPPET]** "DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning" (Nov 2025) — [arXiv 2511.20225](https://arxiv.org/pdf/2511.20225).
- **[SNIPPET]** Oliver, Odena, Raffel, Cubuk, Goodfellow, "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms" (NeurIPS 2018). Findings:
  - "the performance of simple baselines which do not use unlabeled data is often underreported";
  - SSL methods' sensitivity to the amount of labelled data varies;
  - performance "can degrade substantially when the unlabeled dataset contains out-of-distribution examples";
  - unrealistically large validation sets are a problem.

  Sources: [arXiv 1804.09170](https://arxiv.org/pdf/1804.09170); [dblp](https://dblp.org/rec/conf/nips/OliverORCG18.html).
- **[VERIFIED]** "Systematic comparison of semi-supervised and self-supervised learning for medical image classification" (arXiv 2307.08919): with realistic tuning, "MixMatch delivers the most reliable gains across 4 datasets". It compares against self-supervised pretraining + fine-tuning and supervised baselines — [arXiv 2307.08919](https://arxiv.org/abs/2307.08919). This is a reference for the comparison protocol reviewers expect.
- **[SNIPPET]** Su et al., "A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification" (CVPR 2021): realistic SSL evaluation starting from pretrained models — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Su_A_Realistic_Evaluation_of_Semi-Supervised_Learning_for_Fine-Grained_Classification_CVPR_2021_paper.pdf). The fetch returned 403, so I could not verify its conclusions. From memory, SSL gains shrink when starting from ImageNet or self-supervised initialisation, and out-of-domain unlabelled data can hurt. This is GUESSED until checked.
- **[GUESSED]** Foundational and dense-prediction SSL, not opened:
  - FixMatch (arXiv 2001.07685).
  - Mean Teacher (arXiv 1703.01780).
  - UniMatch, "Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation" (CVPR 2023, arXiv 2208.09910).
  - Unbiased Teacher for semi-supervised detection (arXiv 2102.09480). It applies flip augmentation to images and pseudo-boxes consistently, which is the detection analogue of "flip with cell permutation".
  - SimCLRv2, "Big Self-Supervised Models are Strong Semi-Supervised Learners" (arXiv 2006.10029).
- **[VERIFIED, from repo]** The outdoor split is a random per-frame split of continuous video (`outdoor_yolic.py:133-135`; brief §3.3).

### Inferences
- **Already done?** The algorithmic core is fully covered: SSML with class-aware thresholds, FixMatch, and equivariant pseudo-labels under flips. Cell labels add a large, highly imbalanced, spatially structured multi-label vector (1,248 bits). The only arguably new part is exploiting the cell structure, e.g. thresholds per cell group or spatial consistency between neighbouring cells. Fixing the flip permutation is a bug fix, not a contribution.
- **Reviewer-demanded baselines:**
  1. Supervised-only on the labelled subset, ImageNet-pretrained, with strong augmentation (RandAugment, a correct horizontal flip, mixup), tuned with the same budget as the SSL method (Oliver et al.).
  2. Self-supervised pretraining (SimCLR/MoCo/DINO-style, or simply a stronger off-the-shelf pretrained backbone) on the unlabelled frames, then fine-tuning.
  3. An off-the-shelf SSML method (CAP or D2L) applied unchanged to the 1,248-dim vector.
  4. Plain pseudo-labelling / self-training (teacher → student).

  Given the redundancy argument below, baseline 1 plausibly comes close to the SSL method on the in-house data.
- **The near-duplicate-frame problem:** consecutive frames of a scooter video are near-identical. If the labelled subset is sampled per frame at random, then at 5% (≈700 of 14k train frames) almost every scene has a labelled near-twin.
  - The labelled set is effectively "complete", so unlabelled frames add little new information.
  - Supervised accuracy at 5% will already be close to 100%-label accuracy, so the gap SSL can close is small.
  - Test frames also leak.

  To make S3 meaningful, sample the labelled fraction by *scene or sequence* (e.g. perceptual-hash clusters, or filename order if recoverable). Evaluate on held-out scenes, or use Cityscapes, where images are already about 1 per 20-frame snippet and cities can be held out. The Cityscapes extra unlabelled data (the `leftImg8bit_trainextra` coarse set) would be the natural unlabelled pool. Its size and access were not verified here.
- **Failure modes:**
  - Confirmation bias on rare classes. The extreme cell × class imbalance means confidence thresholds produce almost all-negative pseudo-labels, which is exactly why CAP/D2L use class-aware thresholds.
  - Strong colour augmentation interacts with colour-defined classes (hue=0.5 in the YOLIC code, brief D8).
  - Any non-flip geometric augmentation (crop, rotate, translate) breaks the fixed pixel-to-cell mapping, so the "strong view" must be photometric-only or use exactly re-derived cell labels. This limits FixMatch's power.
- **Data feasibility:** high. Labels are hidden from existing sets, and the GPU cost is about 2× supervised training.

### Gaps
- I could not open Su et al. CVPR 2021 (403), so its quantitative conclusions remain unverified.
- I found no paper on SSL specifically for grid or cell presence classification on driving video, and no paper quantifying SSL gains vs frame redundancy in video-derived image datasets. Not found in my one search; it may exist.
- Whether the outdoor Kaggle release preserves frame order or sequence IDs is unknown (brief §6 Q5).

---

## C2: Distilling a Cityscapes segmentation teacher into YOLIC: prior art, and the "just pool a light segmenter" baseline

### Takeaway
Distillation from dense teachers is mature, so pooled soft-cell distillation is a small variant: Structured KD (CVPR 2019), CWD (ICCV 2021), and plain Hinton soft targets (background knowledge, not opened).

The decisive baseline is a light real-time segmenter pooled into cells:
- LR-ASPP MobileNetV3-Small: 68.38 mIoU on Cityscapes val at 2.90B MAdds and 327 ms on one Pixel 3 core for a half-resolution input.
- LR-ASPP MobileNetV3-Large: 72.36 mIoU.
- PIDNet-S: 78.6 mIoU at 93.2 FPS on GPU.

This baseline needs no new method, and on the paper's own Cityscapes numbers YOLIC is already behind YOLO on object classes. Whether YOLIC + distillation can beat a resolution-matched segmenter at equal FLOPs is the open empirical question and the only thing that would make C2 publishable.

Teachers are freely available: SegFormer-B0…B5 in mmseg (76.5–82.3 mIoU), plus HF SegFormer-B0 and Mask2Former/OneFormer Cityscapes checkpoints. However, the teacher's 19 Cityscapes classes don't cover most outdoor hazard classes.

### Cited Findings
- **[VERIFIED]** Table 7 of "Searching for MobileNetV3" (Howard et al., ICCV 2019), Cityscapes val, as I extracted it from the PDF ([arXiv 1905.02244](https://arxiv.org/pdf/1905.02244)):

  | Model | mIoU | Params | MAdds | CPU, full res 1024×2048 | CPU, half res 512×1024 |
  |---|---|---|---|---|---|
  | MobileNetV3-Large + LR-ASPP (128 filters) | 72.36 | 1.51M | 9.74B | 2.47 s | 657 ms |
  | MobileNetV3-Small + LR-ASPP | 68.38 | 0.47M | 2.90B | 1.21 s | 327 ms |

  CPU time is measured on a single large core of a Pixel 3 (floating point).
- **[VERIFIED]** torchvision's `lraspp_mobilenet_v3_large` weights are trained on a COCO subset with the 20 VOC classes (57.9 mIoU on COCO-val2017-VOC-labels). **There is no Cityscapes checkpoint in torchvision**, so this baseline must be trained (a cheap job) — [torchvision docs](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html).
- **[VERIFIED]** PIDNet abstract: PIDNet-S reaches "78.6% mIOU with inference speed of 93.2 FPS" on Cityscapes — [arXiv 2206.02066](https://arxiv.org/abs/2206.02066). Conflict: a search summary of the same paper gave 78.8% for PIDNet-S, plus 80.1% / 39.8 FPS for M and 80.9% / 31.1 FPS for L — [arXiv PDF](https://arxiv.org/pdf/2206.02066). The difference is probably val vs test or different versions; check the table before citing.
- **[SNIPPET]** PP-LiteSeg: 73.6% mIoU / 123.7 FPS, with variants at 72.0% / 273.6 FPS and 77.5% / 102.6 FPS on a GTX 1080Ti (the summary does not say which variant or input size each pair belongs to) — [ResearchGate](https://www.researchgate.net/publication/359786746_PP-LiteSeg_A_Superior_Real-Time_Semantic_Segmentation_Model). The primary source is arXiv 2204.02681 (GUESSED ID).
- **[SNIPPET]** BiSeNetV2: 72.6% test mIoU at 156 FPS on a 1080Ti — [arXiv 2004.02147](https://arxiv.org/pdf/2004.02147).
- **[SNIPPET]** SeaFormer (ICLR 2023 / IJCV 2025) and TopFormer report latency on a single ARM core of a Snapdragon 865, with ADE20K and Cityscapes results. Example: SeaFormer-Base reaches 41.0 vs 33.1 mIoU (ADE20K) against MobileNetV3, at 106 vs 126 ms — [arXiv 2301.13156](https://arxiv.org/pdf/2301.13156); [GitHub fudan-zvg/SeaFormer](https://github.com/fudan-zvg/seaformer); [TopFormer, ResearchGate](https://www.researchgate.net/publication/359920409_TopFormer_Token_Pyramid_Transformer_for_Mobile_Semantic_Segmentation). I did **not** extract their Cityscapes mIoU or latency numbers.
- **[VERIFIED]** mmsegmentation SegFormer Cityscapes (1024×1024 crop, 160k iterations), all with checkpoint download links — [mmseg SegFormer README](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segformer/README.md):

  | Model | mIoU | mIoU (ms+flip) | Memory | Inference FPS (sliding window) |
  |---|---|---|---|---|
  | MiT-B0 | 76.54 | 78.22 | 3.64 GB | 4.74 |
  | MiT-B1 | 78.56 | — | — | — |
  | MiT-B2 | 81.08 | — | 7.42 GB | — |
  | MiT-B3 | 81.94 | — | — | — |
  | MiT-B4 | 81.89 | — | — | — |
  | MiT-B5 | 82.25 | 83.48 | 18.00 GB | 1.39 |

  The memory column is training memory as listed; the README attributes the low FPS to the 1024×1024 sliding-window inference.
- **[VERIFIED]** HuggingFace Cityscapes checkpoints from a search listing — [HF model search](https://huggingface.co/models?search=cityscapes):
  - `nvidia/segformer-b0-finetuned-cityscapes-{1024-1024, 512-1024, 640-1280, 768-768}`
  - `facebook/mask2former-swin-{tiny, small, large, base-IN21k}-cityscapes-semantic`
  - `shi-labs/oneformer_cityscapes_swin_large` and `oneformer_cityscapes_dinat_large`
  - `facebook/maskformer-resnet101-cityscapes`

  The listing I saw did not show an official `nvidia/segformer-b5-finetuned-cityscapes` entry; only a community `leadawon/segformer-b5-finetuned-cityscapes` appeared.
- **[GUESSED]** Dense-KD foundations, not opened:
  - Structured Knowledge Distillation for Semantic Segmentation (Liu et al., CVPR 2019, arXiv 1903.04197).
  - Channel-wise Knowledge Distillation for Dense Prediction (CWD, Shu et al., ICCV 2021, arXiv 2011.13256).
  - Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv 1503.02531).

  Pooling a teacher's probabilities into regions and distilling with BCE/KL is a direct application of soft-target KD with region aggregation.
- **[VERIFIED, from brief]** On the paper's own Cityscapes Table 6, YOLIC-M2 People+Vehicle macro-F1 is about 0.768, vs 0.831 for YOLOv8-N. The brief also lists a pooled light segmenter as a missing baseline (brief §3.1(a), §3.4).

### Inferences
- **Already done?** The generic recipes are mature: dense teacher → compact student, and teacher soft targets on unlabelled data (Noisy Student-style). Applying them to cell targets is incremental. C2's real research value is *answering the baseline question*: at YOLIC's compute (~0.3 GFLOPs at 224×224 for S2, per the brief), does a GAP+FC cell classifier distilled from SegFormer beat a tiny segmenter at the same FLOPs whose output is pooled into cells? A careful negative or positive answer is publishable as part of a benchmark paper (N2). C2 alone as a "method" is weak.
- **Rough compute matching (inference):**
  - MobileNetV3-Small LR-ASPP costs 2.90B MAdds at 512×1024.
  - Scaled linearly by pixel count to a 224×224-equivalent input (×0.096), that is about 0.28B MAdds. This is the same order as YOLIC-S2's reported 0.30 GFLOPs, though the paper's "FLOPs" may mean MAdds.
  - So a resolution-matched segmenter baseline is computationally comparable. Its accuracy at about 224 px input is unknown and will drop well below 68 mIoU. Coarse cell presence, however, needs far less than pixel accuracy.
  - I expect this baseline to be competitive with or better than YOLIC on small cells. Reason: it keeps spatial maps instead of GAP, which is the brief's localisation-bottleneck hypothesis. This is untested.
- **Reviewer-demanded baselines:**
  1. A light segmenter (LR-ASPP MBV3-S/L, PIDNet-S, BiSeNetV2, PP-LiteSeg-T, SeaFormer-T/TopFormer-T) trained on Cityscapes at YOLIC's input size and pooled with the same "≥k pixel" rule, compared at matched FLOPs and CPU latency.
  2. The teacher itself pooled into cells (the upper bound).
  3. YOLIC on hard labels with the D1/D2 bugs fixed.
  4. YOLIC with a spatial head (idea E1).

  Whether YOLIC + KD matches baseline 1 at similar cost is likely "no" for People/Vehicle small cells. That is a guess based on the GAP bottleneck, and it needs the experiment.
- **On Cityscapes itself, the teacher adds little new label information.** It was trained on the same 2,975 train images, so its pooled outputs on train ≈ GT. Gains can come only from soft "dark knowledge" or from extra unlabelled images. Candidates are Cityscapes `train_extra` (≈20k images, coarse set; size and access unverified here), BDD100K, or Mapillary.
- **Domain/class mismatch with the outdoor set:** the 19 Cityscapes classes have no bump, dent, weed (vegetation is the closest), zebra crossing, or traffic cone. Mapping is partial: person/rider → creature, car/truck/bus/bicycle → vehicle, pole → column?, wall, fence, traffic sign. So C2 cannot supervise the full outdoor label space. This limits C2 to Cityscapes-protocol experiments or a partial-class study.
- **Teacher feasibility on 6 GB:**
  - SegFormer-B0/B1 inference is easily feasible.
  - B5 and Mask2Former Swin-L inference at 1024×2048 in fp16 with sliding windows is plausibly feasible but slow. The mmseg B5 sliding-window speed is 1.39 FPS on their GPU, so about 0.5–2 h per 3k images on a 4050 (estimate).
  - mmseg memory figures are training memory, not inference memory.
  - Precompute and cache the pooled soft cell targets once.
- **Failure modes:**
  - The teacher's pooled probability (mean of pixel probabilities) is a *coverage* signal, not presence. It needs a max/noisy-OR aggregation to match the "≥1 pixel" rule; otherwise S2's calibration issue appears again.
  - Teacher errors on thin or far objects become student labels.
  - Domain shift on unlabelled external datasets (different camera framing breaks the fixed cell layout unless images are re-cropped to Cityscapes geometry).

### Gaps
- I did not extract Cityscapes mIoU or ARM latency for SeaFormer-T, TopFormer-T or PP-LiteSeg-T, or Raspberry Pi-class latency for any light segmenter. I found no Pi 4 Cityscapes segmentation benchmark.
- PIDNet-S value conflict (78.6 in the abstract vs 78.8 in the search summary) is unresolved.
- No published work found that distils segmentation teachers specifically into grid or cell presence classifiers; one search, absence not proven.
- I did not verify the official HF SegFormer-B5 Cityscapes checkpoint or the licences of the HF/mmseg weights (Cityscapes-trained weights inherit Cityscapes non-commercial terms; not checked).
