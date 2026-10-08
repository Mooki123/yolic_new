# YOLIC prior art: base-author follow-ups, citing papers, related grid/cell literature, and baselines a reviewer will demand

Research date: 2026-10-01. Tags used throughout:
- **[V]** VERIFIED: I opened the page, API record, README or code, or read the abstract myself.
- **[G]** GUESS / from memory / inferred: not confirmed in this pass.

Primary lookup tools:
- Semantic Scholar Graph API: YOLIC citations (11 records) and author pages for Kai Su (S2 id 2065702109) and Yoichi Tomioka (2269322586).
- OpenAlex: citation cross-check (8 records), abstracts and affiliations.
- ORCID public API for Kai Su (0000-0003-2044-0671).
- arXiv API for abstracts; GitHub API for READMEs and code.

---

## Q1. What have Kai Su and co-authors (Tomioka, Zhao, Liu; Univ. of Aizu) published on cell-wise / CoI / risk / obstacle detection, before and after YOLIC?

### Takeaway
The YOLIC line has 1 prequel conference paper, 2 related 2020 precursors, 2 multi-branch "RoI-expert" papers, and 3 post-YOLIC extensions. The extensions are cost-sensitive region weighting, ternary-weight fault-tolerant ensembles, and a SAM-assisted labelling tool. I found **no** temporal, layout-agnostic / geometry-aware, semi-supervised, segmentation-distilled, FPGA/MCU, or "YOLIC v2" paper by the authors. Kai Su now appears to be at Jiangxi Agricultural University. Of the 24 ideas, the direct author prior art falls on E3 (gating/branches), P2 (safety weighting), S1 (auto-labelling), C2 (distillation) and R3 (quantization/robustness).

### Cited Findings

**Before YOLIC (2019–2022)**
- **[V] Su, Wang, Chowdhury, Zhao, Tomioka, "You Only Look at Interested Cells: Real-Time Object Detection Based on Cell-Wise Segmentation", iCAST 2020 (= YOLIC ref [33]).** Introduces YOLIC as multi-label classification of predefined interested cells with one network ("YOLIC can use existing classification models without any structural change. The main point is to define a proper loss function"). Test case: on-road risk detection. Claims faster and more accurate than YOLOv3 in FPS and F1. — [DOI 10.1109/iCAST51195.2020.9319469](https://doi.org/10.1109/iCAST51195.2020.9319469); abstract via [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/iCAST51195.2020.9319469)
- **[V] Su, Intisar, Zhao, Tomioka, "Knowledge Distillation for Real-time On-Road Risk Detection", IEEE DASC/PiCom/CBDCom/CyberSciTech 2020, pp. 110–117 (= YOLIC ref [23]).** A tiny network improved by knowledge distillation detects "the position and type of road obstacles" in real time on a Raspberry Pi with an Intel Neural Compute Stick 2. — [DOI 10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00032](https://doi.org/10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00032)
  - [G] Whether this is a per-cell crop classifier or a YOLIC-style multi-label head can't be told from the abstract. I could not open the full text.
- **[V] Wang, Su, Chowdhury, Zhao, Tomioka, "Comparison Between Block-Wise Detection and A Modular Selective Approach", iCAST 2020.** "Block-Wise Detection" with VGG19 / VGG19-BN / ResNet backbones reaches 90.51% accuracy (ResNet-50), against 89.40% for MS-Net (ResNet-44 router + ResNet-101 expert). The dataset is the group's "on-road risk detection dataset" recorded with an RGB-D sensor on a senior car. — [DOI 10.1109/iCAST51195.2020.9319484](https://doi.org/10.1109/iCAST51195.2020.9319484)
  - [G] This "block-wise detection" is the most likely concrete form of the "earlier per-cell crop classifier" baseline (c) in the brief.
- **[V] Modular-network precursors by Kai Su (not cell-wise, same router/expert idea later reused):**
  - MS-NET: modular selective network, Int. J. Mach. Learn. & Cybern. 2020/21 — [DOI 10.1007/s13042-020-01201-8](https://doi.org/10.1007/s13042-020-01201-8)
  - CMNN: Coupled Modular Neural Network, IEEE Access 2021 — [DOI 10.1109/ACCESS.2021.3093541](https://doi.org/10.1109/ACCESS.2021.3093541)
  - Stabilization of the Modular Selective Neural Network Model Based on Inter-Class Correlation, CYBCONF 2021 — [DOI 10.1109/CYBCONF51991.2021.9464136](https://doi.org/10.1109/CYBCONF51991.2021.9464136)
  - Mapping DCNN to a Three Layer Modular Architecture, ICSIP 2020 — [DOI 10.1109/ICSIP49896.2020.9339332](https://doi.org/10.1109/ICSIP49896.2020.9339332)
  - Product Surface Defect Detection Based on CNN Ensemble with Rejection, DASC 2019 — [DOI 10.1109/DASC/PiCom/CBDCom/CyberSciTech.2019.00067](https://doi.org/10.1109/DASC/PiCom/CBDCom/CyberSciTech.2019.00067)
  - Full list: [Kai Su ORCID works](https://pub.orcid.org/v3.0/0000-0003-2044-0671/works); [S2 author 2065702109](https://api.semanticscholar.org/graph/v1/author/2065702109/papers?fields=title,year,venue)
- **[V] Su, Tomioka, Zhao, "A Multi-branch Network with Internal Feature Fusion for Road Risk Detection", ICCE-Asia 2022.** One common block extracts global features, then several branches detect road risks in particular pre-specified regions of interest. It "can reduce the inference cost significantly while preserving accuracy." — [DOI 10.1109/ICCE-Asia57006.2022.9954875](https://doi.org/10.1109/ICCE-Asia57006.2022.9954875)

**The main paper**
- **[V] YOLIC journal version:** Image and Vision Computing 147 (2024) 105095; arXiv 2307.06689 (v1 July 2023). Abstract: >30 FPS on a Raspberry Pi 4B CPU; all resources on the project page. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0262885624001999); [arXiv](https://arxiv.org/abs/2307.06689)
- **[V] Citation counts disagree:** Semantic Scholar shows 11 citations, OpenAlex shows 8. — [S2](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.imavis.2024.105095?fields=citationCount); [OpenAlex](https://api.openalex.org/works/doi:10.1016/j.imavis.2024.105095)
- **[V] The project page lists no follow-up work.** It links only the paper, the two Kaggle datasets (outdoor hazard, indoor obstacle), YOLIC_code, YOLIC-Labeling and Cell-designer. — [project page](https://kai3316.github.io/yolic.github.io/)

**After YOLIC (2024–2026)**
- **[V] Su, Tomioka, Zhao, Liu, "A Selective Multi-Branch Network for Edge-Oriented Object Localization and Classification", Electronics 13(8):1472, 2024.**
  - Builds on YOLIC. CoIs are grouped into RoIs "based on their locations and urgency", with an expert branch per RoI.
  - A "selective attention unit" locates RoIs likely to contain objects and triggers only the corresponding branches. "Only part of the feature map is used to make decisions."
  - Claims lower inference time with competitive accuracy.
  - — [MDPI](https://www.mdpi.com/2079-9292/13/8/1472) (403 to my fetcher; abstract via [S2](https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/electronics13081472)); [ResearchGate](https://www.researchgate.net/publication/379793009_A_Selective_Multi-Branch_Network_for_Edge-Oriented_Object_Localization_and_Classification)
  - [G] Could not read the full text. The datasets, Pi FPS and head design are unknown to me.
  - Date conflict: the MDPI PDF version stamp corresponds to April 2024, while a search snippet said "December 2024". Treat it as April 2024 (vol. 13, issue 8).
- **[V] Su, Zhao, Tomioka, Liu, "Cost-Sensitive Road Obstacle Detection for Low-Cost Autonomous Vehicles", IEEE MobileCloud 2024.**
  - Assigns higher weights to cells close to the vehicle. Weights vary with "detection distance, direction, and relation to driving".
  - Reduces cost in the most dangerous areas compared with baseline YOLIC, on two road-obstacle datasets.
  - Fastest model is real-time on a Raspberry Pi 4B. Targets scooters and delivery robots.
  - — [DOI 10.1109/MobileCloud62079.2024.00011](https://doi.org/10.1109/MobileCloud62079.2024.00011)
- **[V] Ishii, Su, Tomioka, Saito, "Efficient and Fault-tolerant Object Localization and Classification Based on an Ensemble of Dual Ternary YOLIC Models", IEEE MCSoC 2024.**
  - Uses dual-modular redundancy of **ternary-weight** YOLIC models for tolerance to hardware faults.
  - Three ternary models give comparable accuracy and cut compute by 89.5% (79.7%) versus triple modular redundancy of the FP model.
  - Evaluated on two road-surface risk datasets.
  - — [DOI 10.1109/MCSoC64144.2024.00061](https://doi.org/10.1109/MCSoC64144.2024.00061)
- **[V] Su, Zhao (Aihua), Hua, Chen, "YOLIC labeling: A semi-automated image annotation tool with segment anything model for cell-wise labeling", SoftwareX 34 (June 2026), 102577.**
  - SAM-assisted labelling, manual polygon annotation and semi-automatic labelling on customizable cell configurations.
  - OpenAlex lists Kai Su at **Jiangxi Agricultural University**; the co-authors are not from Aizu.
  - — [DOI 10.1016/j.softx.2026.102577](https://doi.org/10.1016/j.softx.2026.102577); metadata via [Crossref](https://api.crossref.org/works/10.1016/j.softx.2026.102577) and [OpenAlex](https://api.openalex.org/works/doi:10.1016/j.softx.2026.102577)
  - The tool's README lists "semi-automatic labeling", "RGB and RGB-D support" and "customizable granularity". — [GitHub YOLIC-Labeling](https://github.com/kai3316/YOLIC-Labeling)
  - Semantic Scholar mis-resolves the authors as "Kai-Wei Su, Ai-Hua Zhao …". — [S2](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.softx.2026.102577?fields=authors)
- **[V] Kai Su's recent GitHub activity is outside YOLIC.**
  - `CGSDLKDModel` is code for a KD-based lightweight YOLO11 navel-orange detector, under review at Computers and Electronics in Agriculture.
  - `YOLIC_code` was last updated 2025-10 and still holds only the 10 MobileNetV2 scripts.
  - — [GitHub kai3316 repos](https://api.github.com/users/kai3316/repos); [CGSDLKDModel](https://github.com/kai3316/CGSDLKDModel)

**Tomioka-group work adjacent to YOLIC (Aizu; not YOLIC itself, per abstracts)**
- **[V] Hayafuji, Tomioka et al., "Risk- and Quantization-Aware Training for CNNs", MCSoC 2025.** Temperature-scaled risk-penalty loss combined with QAT, on a self-driving RC-car dataset and CIFAR-10. — [DOI 10.1109/MCSoC67473.2025.00037](https://doi.org/10.1109/MCSoC67473.2025.00037)
- **[V] Suzuki, Saito, Semba, Tomioka, Hanyu, "Random Forest-Based Approximation for Quantized CNNs on Edge FPGAs", MCSoC 2025.** — [DOI 10.1109/MCSoC67473.2025.00107](https://doi.org/10.1109/MCSoC67473.2025.00107)
- **[V] Other Tomioka MCSoC papers in S2:**
  - "Autonomous Driving Robot Using FPGA and BNN with Random Forest" (2023)
  - "Fault-Tolerant Ensemble CNNs Increasing Diversity Based on Knowledge Distillation" (2023)
  - "Highly Efficient MetaFormer-Based End-to-End Autonomous Driving Model With Token-Pruning" (2024)
  - — [S2 author 2269322586](https://api.semanticscholar.org/graph/v1/author/2269322586/papers?fields=title,year,venue)

**What I specifically looked for and did not find**
- **[V] No "YOLIC v2", temporal/video YOLIC, multi-task YOLIC, YOLIC-on-FPGA/MCU, or semi-supervised YOLIC paper** appears in:
  - Kai Su's ORCID, his S2 author page, the Tomioka S2 page, the YOLIC citation lists (S2 and OpenAlex), or the project page.
  - — [ORCID](https://pub.orcid.org/v3.0/0000-0003-2044-0671/works); [S2 citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.imavis.2024.105095/citations?fields=title,year,venue&limit=100)
  - The closest hardware-oriented item is the ternary YOLIC paper (MCSoC 2024).

### Inferences
- **[G] Overlap of the 24 ideas (02_ideas.md) with the authors' own prior art.** A reviewer will expect each of these to be cited:
  - **E3 (uncertainty-gated cascade)** is closest to the *Selective Multi-Branch Network* (Electronics 2024) and the ICCE-Asia 2022 multi-branch net. Both gate computation by region, so E3 must be framed as *uncertainty/temporal* gating versus their *learned RoI-trigger* gating, and should compare against it.
  - **P2 (recall-guaranteed thresholds)** sits next to *Cost-Sensitive YOLIC* (MobileCloud 2024), which already handles "safety-critical regions matter more" via loss weighting by distance and direction. P2's novelty must be the *statistical guarantee* (conformal), not the region priority. Cost-sensitive YOLIC is a natural baseline.
  - **S1 (zero-label auto-labelling)** overlaps with *YOLIC labeling + SAM* (SoftwareX 2026). That tool is human-in-the-loop, semi-automatic. S1 differs if it is text-prompted, fully automatic, and measures downstream accuracy.
  - **C2 (segmentation-teacher distillation)** has a precursor in the KD on-road risk paper (DASC 2020, ref [23]).
  - **R3 (quantization × corruption)** is adjacent to ternary fault-tolerant YOLIC (hardware faults, not input corruptions) and to Risk- and QAT (MCSoC 2025).
  - **A1, A2, A3, P1, P3, S2, S3, E1, E2, R1, R2, N1, N2, N3, C1, C3** have no author prior art that I could find.
- [G] With Kai Su apparently moved to Jiangxi Agricultural University and working on fruit detection, an active YOLIC follow-up from the original group looks less likely in the short term. This is a guess from affiliation and repo activity.

### Gaps
- I could not read the full texts of: Selective Multi-Branch (MDPI returned 403), Cost-Sensitive, Ternary YOLIC, KD-2020, iCAST-2020 YOLIC, Block-wise comparison. So I lack their datasets, FPS, head designs, and whether the "two road obstacle datasets" are the public outdoor set plus Cityscapes.
- Google Scholar and IEEE Xplore were not queried directly (no API; scraping blocked). Coverage relies on S2, OpenAlex, ORCID and Crossref, so very recent Japanese domestic workshop papers (e.g. IEICE) may be missing.
- I could not confirm whether the 2020 KD paper's student is a per-cell crop classifier or a YOLIC head.

---

## Q2. Which papers cite YOLIC, and do any extend it?

### Takeaway
11 citing records (S2) / 8 (OpenAlex). The only papers that **extend** YOLIC are the authors' own three (cost-sensitive, ternary ensemble, labelling tool). Every external citer, as far as abstracts show, cites YOLIC as related "efficient edge detection" or "region/block-based" work and builds its own detector, dataset or survey. **No independent group has extended YOLIC.**

### Cited Findings
Self-citations (extensions):
- **[V] Cost-Sensitive Road Obstacle Detection (MobileCloud 2024):** region-weighted, cost-sensitive YOLIC — [DOI](https://doi.org/10.1109/MobileCloud62079.2024.00011)
- **[V] Dual Ternary YOLIC ensemble (MCSoC 2024):** ternary weights plus DMR fault tolerance — [DOI](https://doi.org/10.1109/MCSoC64144.2024.00061)
- **[V] YOLIC labeling (SoftwareX 2026):** SAM-assisted cell-wise annotation tool — [DOI](https://doi.org/10.1016/j.softx.2026.102577)

External citers (one line each; none extends YOLIC per abstract):
- **[V] Kaundanya et al., "MLRDv2: A Dataset for Improving Micromobility Safety via Attention-Integrated Compact CNN Models", SN Computer Science 2026.** A micromobility lane-recognition dataset, framed as image *classification* instead of segmentation, with MobileNetV2/V3 plus channel and spatial attention. This is the same "classify, don't segment, on cheap hardware for scooters" motivation, but not cell-wise. — [DOI 10.1007/s42979-026-04745-8](https://doi.org/10.1007/s42979-026-04745-8)
- **[V] Yan, Kaundanya, O'Connor, Little, Liu, "Machine Learning in Micromobility: A Systematic Review of Datasets, Techniques, and Applications", arXiv 2508.16135 (2025).** A survey. — [arXiv](https://arxiv.org/abs/2508.16135)
- **[V] Lu et al., "Neural-symbolic framework for dynamic hazard detection through compositional visual reasoning", J. Ambient Intell. Humaniz. Comput. 2026.** Industrial hazard detection with symbolic safety rules and Meta Module Networks. — [DOI 10.1007/s12652-026-05088-1](https://doi.org/10.1007/s12652-026-05088-1)
- **[V] Liu et al. (SCUT), "Fast highway abandoned object detection via block-based multi-group foreground extraction" (UBMG), Scientific Reports 2025.** Block-based frame selection and foreground extraction, plus a new HAO video dataset. — [DOI 10.1038/s41598-025-20331-z](https://doi.org/10.1038/s41598-025-20331-z)
- **[V] Khater et al., "TinyEcoWeedNet: Edge Efficient Real-Time Aerial Agricultural Weed Detection", arXiv 2509.18193 (2025).** Pruning, QAT and TensorRT on a Jetson Orin Nano; 184 FPS at FP16. — [arXiv](https://arxiv.org/abs/2509.18193)
- **[V] Xie et al. (Wuhan Univ. of Technology), "A knowledge distillation-based object detection model focused on road scene perception and localization", Neurocomputing 2025.** No abstract was available in S2 or OpenAlex. — [DOI 10.1016/j.neucom.2025.131501](https://doi.org/10.1016/j.neucom.2025.131501)
- **[V] Choi, Baek, Lee (ETRI), "Design of specific situation estimation function using multi-robot system in military operations", ETRI Journal 2025.** Clustered edge devices on robots for object location estimation and "risk zone identification" indoors. — [DOI 10.4218/etrij.2024-0477](https://doi.org/10.4218/etrij.2024-0477)
- **[V] Fan et al., "SS-YOLOv8: A Lightweight Algorithm for Surface Litter Detection", Applied Sciences 2024.** Lightweight YOLOv8 variant. — [DOI 10.3390/app14209283](https://doi.org/10.3390/app14209283)

Second-order citations (citing the YOLIC precursors or extensions):
- **[V] Cost-Sensitive YOLIC is cited by:**
  - a 2025 pothole-detection framework (Mechatronics & Intelligent Transportation Systems) — [DOI 10.56578/mits040204](https://doi.org/10.56578/mits040204)
  - a 2026 ADSSSC paper, "Real-Time Obstacle Detection and Risk Assessment for Autonomous Vehicles: An Ensemble with Adaptive Low-Light Enhancement" — [DOI 10.1109/adsssc67751.2026.11582681](https://doi.org/10.1109/adsssc67751.2026.11582681)
  - — via [OpenAlex](https://api.openalex.org/works?filter=cites:W4398614775)
- **[V] The 2020 KD on-road-risk paper is cited by:**
  - the TPAMI 2023 survey "When Object Detection Meets Knowledge Distillation" — [DOI 10.1109/tpami.2023.3257546](https://doi.org/10.1109/tpami.2023.3257546)
  - ACM Computing Surveys 2025, "Designing Object Detection Models for TinyML" — [DOI 10.1145/3744339](https://doi.org/10.1145/3744339)
  - SIViP 2024, "On-road obstacle detection in real time environment using an ensemble deep learning model" — [DOI 10.1007/s11760-024-03241-x](https://doi.org/10.1007/s11760-024-03241-x)
- **[V] The Selective Multi-Branch paper has 0 citations in OpenAlex.** — [OpenAlex query](https://api.openalex.org/works/doi:10.3390/electronics13081472)

### Inferences
- [G] Low uptake (about 8–11 citations in roughly two years, none extending the method) means two things for an extension paper:
  - **Upside:** novelty against third-party YOLIC follow-ups is easy to claim.
  - **Downside:** reviewers may question relevance. The paper should anchor to the broader "grid/cell classification" and "lightweight segmentation" literatures (Q3/Q4), not only to YOLIC.
- [G] MLRD/MLRDv2 (DCU, micromobility) is the most aligned external community (scooter safety on compact CNNs). Their dataset may be a candidate second domain, but I have not checked licence or labels.

### Gaps
- Google Scholar "cited by" could not be queried programmatically. It typically lists more citers (theses, preprints) than S2 or OpenAlex.
- No abstract is available for the Neurocomputing 2025 KD paper or the SIViP 2024 ensemble paper, so whether they use YOLIC's cell formulation is unknown.

---

## Q3. Closely related "grid/cell classification instead of detection" literature

### Takeaway
The formulation "global features → FC → per-cell (or per-row-anchor) classification" already exists and is widely accepted in lane detection: Ultra-Fast Lane Detection v1/v2. UFLD keeps spatial layout by **flattening a 1×1-conv-reduced map**, not by GAP. Column-wise obstacle classification (StixelNet family), monocular semantic occupancy grids, and image-level traversability or collision classifiers are the other close neighbours. YOLIC cites none of these. A reviewer will expect at least UFLD and StixelNet to be discussed.

### Cited Findings
- **[V] Ultra Fast Structure-aware Deep Lane Detection (UFLD), Qin, Wang, Li, ECCV 2020.** Treats lane detection "as a row-based selecting problem using global features". A light version reaches 300+ FPS. — [arXiv 2004.11757](https://arxiv.org/abs/2004.11757)
- **[V] UFLD head design (from the official code):** `self.pool = Conv2d(512, 8, 1)`, then `.view(-1, 1800)` (8×9×25 flatten), then `Linear(1800, 2048)`, then `Linear(2048, total_dim)` with `cls_dim = (gridding cells, row anchors, lanes)`. So spatial layout is preserved through a flatten of a channel-reduced map, **not** global average pooling. — [UFLD model.py](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/model/model.py)
- **[V] UFLDv2, TPAMI 2022 (arXiv 2206.07389).** Hybrid row and column anchors with ordinal classification over grid cells; 300+ FPS lightweight version. — [arXiv 2206.07389](https://arxiv.org/abs/2206.07389)
- **[V] StixelNet, Levi, Garnett, Fetaya, BMVC 2015.** Monocular obstacle detection and road segmentation reduced to a **column-wise** problem solved by a CNN, on KITTI. — [BMVC 2015 paper109](https://www.bmva.org/bmvc/2015/papers/paper109/)
- **[V] StixelNExT (2024):** monocular, lightweight, directly predicts a multi-layer Stixel world, trained from LiDAR-generated ground truth. — [arXiv 2407.08277](https://arxiv.org/abs/2407.08277)
- **[V] StixelNExT++ (2025):** "Lightweight Monocular Scene Segmentation and Representation for Collective Perception". Title only seen in search results; abstract not read. — [arXiv 2507.06687](https://arxiv.org/abs/2507.06687)
- **[V] Lu, van de Molengraft, Dubbelman, "Monocular Semantic Occupancy Grid Mapping with Convolutional Variational Encoder-Decoder Networks", RA-L 2019.**
  - Predicts a 64×64 top-view semantic grid from a monocular image, evaluated on Cityscapes.
  - More than 12% mIoU better than flat-plane deterministic mapping.
  - About 35 Hz on a Titan V at 256×512.
  - Relevant to idea A2 (ground-plane cells).
  - — [arXiv 1804.02176](https://arxiv.org/abs/1804.02176)
- **[V] Roddick & Cipolla, "Pyramid Occupancy Networks", CVPR 2020.** Monocular image to BEV semantic Bayesian occupancy grid; accumulates over cameras and timesteps, which is relevant to A2 and A3. — [arXiv 2003.13402](https://arxiv.org/abs/2003.13402)
- **[V] Hirose et al., GONet (2018).** Semi-supervised (GAN) image-level **traversability classification** from fisheye images on a mobile robot; releases about 24 h of indoor video data. Relevant to S3 and the indoor setting. — [arXiv 1803.03254](https://arxiv.org/abs/1803.03254)
- **[V] Gandhi, Pinto, Gupta, "Learning to Fly by Crashing", IROS 2017.** Self-supervised image classification for UAV obstacle avoidance from 11,500 crashes. — [arXiv 1704.05588](https://arxiv.org/abs/1704.05588)
- **[V] "A Novel Obstacle Detection Method based on Deformable Grid for the Visually Impaired", IEEE Trans. Consumer Electronics 2015.** Grid-based obstacle detection for assistive navigation. Seen in search results only. — [ACM DL / IEEE TCE](https://dl.acm.org/doi/abs/10.1109/TCE.2015.7298298)
- **[V] YOLO-OD (2024):** YOLOv8-based obstacle detection for visually impaired navigation. A detector-based counterpart; seen in search results only. — [PMC11645096](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11645096/)
- **[V] SqueezeDet (2016):** a fully-convolutional, small, low-power detector for autonomous driving. Its conv head predicts per grid location, the conv-head analogue of YOLIC's FC head. — [arXiv 1612.01051](https://arxiv.org/abs/1612.01051)
- **[V] Position-encoding literature that explains how a GAP+FC head can localise at all** (brief §1.2 [I]; idea N1):
  - Islam, Jia, Bruce, "How Much Position Information Do CNNs Encode?" (ICLR 2020) shows a "surprising degree of absolute position information" in standard CNNs. — [arXiv 2001.08248](https://arxiv.org/abs/2001.08248)
  - Kayhan & van Gemert, "On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location" (CVPR 2020) shows that CNNs exploit absolute location "by exploiting image boundary effects", and proposes a fix. — [arXiv 2003.07064](https://arxiv.org/abs/2003.07064)
- **[V] Survey anchor for traversability:** "A Survey of Traversability Estimation for Mobile Robots" (2022). — [arXiv 2204.10883](https://arxiv.org/abs/2204.10883)

### Inferences
- [G] **UFLD is the single most important missing citation for YOLIC extensions.** It is the same "classification over predefined spatial bins from a global feature" idea, published at ECCV and TPAMI, with a different spatial-retention mechanism (flatten, not GAP). Idea E1 (spatial head) and N1 (how GAP localises) can be framed as "YOLIC's GAP vs UFLD's flatten vs a 1×1-conv + cell-pooling head".
- [G] Idea N1 is partly pre-empted at the general level by Islam 2020 and Kayhan 2020. N1's novelty must be the *cell-presence-specific* analysis: cell size, border distance, padding swaps on a deployed edge task.
- [G] Idea A2 (ground-plane cells) must be positioned against BEV occupancy-grid work (Lu 2019, Roddick 2020). Its differentiator is edge cost and fixed cell semantics, not BEV prediction per se.

### Gaps
- I did not verify a specific paper on **patch- or cell-level obstacle classification for wheelchairs or visually-impaired users on a fixed grid** that closely matches YOLIC. The search surfaced only the 2015 deformable-grid paper and detector-based systems. A dedicated search (IEEE Xplore "grid-based obstacle classification wheelchair CNN") is still needed.
- I did not open the original Garnett et al. follow-up ("Real-time category-based and general obstacle detection", ICCVW 2017); the arXiv id I tried (1707.01183) was wrong.
- I recall Giusti et al., "A Machine Learning Approach to Visual Perception of Forest Trails" (RA-L 2016: left/centre/right image classification), but did not verify it in this pass. [G]

---

## Q4. Strongest simple baselines for cell-presence on edge CPUs (segmentation pooled into cells; ultra-light detectors), with published Pi 4 / ARM numbers

### Takeaway
- **On published numbers, no ultra-light *detector* at its native resolution reaches 30 FPS on a Raspberry Pi 4 CPU.**
  - Best measured: Yolo-FastestV2 at 352×352, **18.8 FPS** (ncnn, Pi 4 overclocked to 1950 MHz).
  - NanoDet-m 320: 13.0 FPS. PP-PicoDet 320: 7.5. YOLOv8n 640: 3.1.
  - Yolo-Fastest-1.1 under MNN at 320: about 35 ms (≈28.6 FPS).
- Torchvision's quantized **MobileNetV2 classifier itself runs at 33.7 FPS on a Pi 4 at 224**, and ShuffleNetV2 x0.5 at 46.7 FPS (PyTorch tutorial). So YOLIC's speed is essentially "the backbone's speed".
- [G] At YOLIC's 224 input, Yolo-FastestV2 / FastestDet would plausibly exceed 30 FPS (FLOPs scale about (224/352)² ≈ 0.40). This is the comparison a reviewer will demand, but it is **not published**.
- For segmentation, Cityscapes mIoU numbers are abundant (LR-ASPP-MNV3-L 72.3/72.6, Fast-SCNN 68.0, BiSeNetV2 72.6, PP-LiteSeg-T 72.0, PIDNet-S 78.6, SeaFormer-S 71.1–76.4). But **I found no published Raspberry Pi 4 FPS for any of them**. All reported speeds are on desktop GPUs or Snapdragon ARM cores.

### Cited Findings

**Ultra-light detectors: Raspberry Pi 4 and ARM numbers**
- **[V] Qengineering Pi 4 benchmark table** (ncnn; "RPi 4 1950" column = Pi 4 overclocked to 1950 MHz; FPS covers inference only, excluding grabbing, post-processing and drawing):

  | Model | Input | Pi 4 FPS | COCO mAP |
  |---|---|---|---|
  | Yolo-FastestV2 | 352² | **18.8** | 24.1 mAP@0.5 |
  | NanoDet | 320² | 13.0 | 20.6 |
  | YOLOX-nano | 416² | 7.0 | — |
  | PP-PicoDet | 320² | 7.5 | 27.0 |
  | NanoDet-Plus | 416² | 5.0 | — |
  | YOLOv4-tiny | 416² | 3.4 | — |
  | YOLOv8n | 640² | 3.1 | — |
  | YOLOv6n | 640² | 2.7 | — |
  | YOLOv5n | 640² | 1.6 | — |

  — [Qengineering YoloFastestV2-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloFastestV2-ncnn-Raspberry-Pi-4) (same table in [YoloV8-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4))
  - Caveat [V]: the table's YOLOv5 rows repeat identical mAP and FPS for nano and small (22.5 / 1.6), which looks like a copy error. Treat the YOLOv5 rows as unreliable.
- **[V] Yolo-Fastest README.**
  - Yolo-Fastest-1.1 (320², 0.252 BFLOPs, 0.35 M params, COCO mAP@0.5 24.4%): 62.31 ms on a Raspberry Pi 3B (ncnn, bf16s).
  - Third-party MNN port: "raspberry pi 4B 2G, input 320×320, average inference time 0.035 s".
  - The README also claims "Raspberry Pi 4b … full real-time 30fps+".
  - — [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
- **[V] Yolo-FastestV2 README:** 352², COCO mAP@0.5 24.10%, 0.212 GFLOPs, 0.25 M params, ShuffleNetV2 backbone. 3.29 ms (4 cores) / 5.37 ms (1 core) on a Kirin 990 (Mate 30) with ncnn. — [dog-qiuqiu/Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)
- **[V] FastestDet README** (RK3568 Cortex-A55 at 2.0 GHz, ncnn):

  | Model | Input | COCO mAP | Latency (4 cores / 1 core) |
  |---|---|---|---|
  | FastestDet (0.24 M params) | 352² | mAP@0.5 25.3%, mAP@0.5:0.95 13.0% | 23.51 / 70.62 ms |
  | Yolo-FastestV2 | 352² | — | 23.8 / 68.9 ms |
  | nanodet_m | 320² | — | 49.24 ms |
  | yolox-nano | 416² | — | 76.31 ms |
  | yolov6n | 416² | — | 109.24 ms |
  | yolov5s | 640² | — | 395.31 ms |

  Also 16.24 ms (4 cores) on a Snapdragon 835. No Pi 4 number. — [dog-qiuqiu/FastestDet](https://github.com/dog-qiuqiu/FastestDet)
  - [G] The Cortex-A55 at 2.0 GHz is likely slower per core than the Pi 4's Cortex-A72 at 1.5–1.8 GHz, so FastestDet at 352 on a Pi 4 is plausibly at least as fast as about 40 FPS. Unverified.
- **[V] NanoDet README** (ARM latency on Kirin 980 4×A76, ncnn):

  | Model | Input | COCO mAP | ARM latency | GFLOPs | Params |
  |---|---|---|---|---|---|
  | NanoDet-m | 320² | 20.6 | 10.23 ms | 0.72 | 0.95 M |
  | NanoDet-Plus-m | 320² | 27.0 | 11.97 ms | 0.9 | 1.17 M |
  | NanoDet-Plus-m | 416² | 30.4 | 19.77 ms | — | — |
  | YOLOv5-n | 640² | — | 44.39 ms | — | — |
  | YOLOX-Nano | 416² | — | 23.08 ms | — | — |

  ShuffleNetV2 1.0× backbone. — [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)
- **[V] PP-PicoDet (arXiv 2111.00902):** PicoDet-S has 0.99 M params and 30.6% mAP. "123 FPS (150 FPS using Paddle Lite) on mobile ARM CPU when the input size is 320." — [arXiv 2111.00902](https://arxiv.org/abs/2111.00902)
- **[V] Ultralytics Raspberry Pi guide** (as of the fetch) now benchmarks only **YOLO26** on a **Raspberry Pi 5**. YOLO26n at 640 under ONNX gives 7.79 FPS, against 6.79 FPS for YOLO11n. No Pi 4 table. — [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- **[V] EfficientDet (arXiv 1911.09070):** scalable family. I found no Pi 4 number for EfficientDet-Lite in this pass. — [arXiv 1911.09070](https://arxiv.org/abs/1911.09070)

**Backbone-only speed on a Pi 4 (decisive for "is YOLIC's speed the method or the backbone?")**
- **[V] PyTorch official tutorial "Real Time Inference on Raspberry Pi 4 and 5":** quantized (qnnpack) torchvision classifiers at 224×224 on a Pi 4.

  | Model | FPS | Model time (ms) |
  |---|---|---|
  | mobilenet_v2 | 33.7 | 26.4 |
  | mobilenet_v3_large | 29.3 | 30.7 |
  | shufflenet_v2_x0_5 | 46.7 | 18.2 |
  | shufflenet_v2_x1_0 | 24.4 | 37.7 |
  | resnet18 | 9.2 | 100.3 |

  JIT raises about 20 FPS to about 30 FPS. Pi 5 reaches about 41 FPS. — [PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html)
  - [G] Compare YOLIC Table 9's 40.06 FPS for INT8 ShuffleNetV2 under ncnn: framework choice (ncnn vs PyTorch) matters as much as the head.

**Lightweight semantic segmentation (to pool into cells): Cityscapes accuracy and speed**
- **[V] LR-ASPP / MobileNetV3 (Howard et al., ICCV 2019).** "MobileNetV3-Large LR-ASPP is 30% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes" (arXiv abstract; the WebSearch summary said 34%). — [arXiv 1905.02244](https://arxiv.org/abs/1905.02244)
- **[V] fastseg (third-party LR-ASPP implementation)** on Cityscapes at 1024×2048 (V100):

  | Model | Cityscapes mIoU | GPU FPS | TensorRT FPS |
  |---|---|---|---|
  | MNV3-Large LR-ASPP F=128 | 72.3 | 25.7 | 37.3 |
  | MNV3-Small LR-ASPP F=128 | 67.4 | 38.2 | 52.4 |

  No ARM numbers. — [ekzhang/fastseg](https://github.com/ekzhang/fastseg)
- **[V] Fast-SCNN (arXiv 1902.04502):** 68.0% mIoU at 123.5 FPS on Cityscapes at 1024×2048 (GPU); "suited to … embedded devices with low memory". — [arXiv 1902.04502](https://arxiv.org/abs/1902.04502)
- **[V] BiSeNet V2 (arXiv 2004.02147):** 72.6% mIoU on Cityscapes test at 156 FPS on a GTX 1080 Ti (2048×1024 input per abstract). — [arXiv 2004.02147](https://arxiv.org/abs/2004.02147)
- **[V] PIDNet (arXiv 2206.02066):** PIDNet-S 78.6% mIoU at 93.2 FPS on Cityscapes (GPU). — [arXiv 2206.02066](https://arxiv.org/abs/2206.02066)
- **[V] PP-LiteSeg (arXiv 2204.02681):** 72.0% mIoU / 273.6 FPS and 77.5% / 102.6 FPS on Cityscapes test, on a GTX 1080 Ti. — [arXiv 2204.02681](https://arxiv.org/abs/2204.02681)
- **[V] TopFormer (CVPR 2022).**
  - The README measures latency on a *single Snapdragon 865 ARM CPU core* at 512².
  - ADE20K mIoU: TopFormer-T 32.5–34.6 (1.4 M params, 0.5–0.6 GFLOPs), -S 36.5–37.0, -B 38.3–39.2.
  - Per the abstract, the tiny version is real-time on an ARM mobile device.
  - The README has no Cityscapes table.
  - — [arXiv 2204.05525](https://arxiv.org/abs/2204.05525); [hustvl/TopFormer](https://github.com/hustvl/TopFormer)
- **[V] SeaFormer / SeaFormer++ (ICLR 2023; arXiv 2301.13156).** Best accuracy-latency trade-off "on the ARM-based mobile devices". Cityscapes (README):

  | Model | Cityscapes mIoU | GFLOPs |
  |---|---|---|
  | SeaFormer-Small, half-res head | 71.1 | 2.0 |
  | SeaFormer-Small, full-res head | 76.4 | 8.0 |
  | SeaFormer-Base, half-res head | 72.2 | 3.4 |
  | SeaFormer-Base, full-res head | 77.7 | 13.7 |

  — [arXiv 2301.13156](https://arxiv.org/abs/2301.13156); [fudan-zvg/SeaFormer](https://github.com/fudan-zvg/SeaFormer)
- **[V] Segmentation on a Pi 4, the only number found:** Qengineering's TFLite U-Net runs at about 4 FPS on a Pi 4 (overclocked to 1900 MHz, 32-bit). This illustrates generic dense segmentation cost on a Pi 4, but it is not a lightweight real-time model. — [Qengineering TensorFlow_Lite_Segmentation_RPi_32-bit](https://github.com/Qengineering/TensorFlow_Lite_Segmentation_RPi_32-bit)

### Inferences
- [G] **Answer to "which run >30 FPS on a Pi 4"** (on published evidence):
  - No detector at its native resolution.
  - Possibly Yolo-Fastest-1.1 via MNN at 320 (≈28.6 FPS, just under).
  - Quantized MobileNetV2 or ShuffleNetV2-x0.5 *classifiers* at 224.
  - YOLIC's claimed QS2 (40 FPS).
  - So the paper's "first >30 FPS at 224" claim is not contradicted by published numbers, but those numbers don't test detectors *at 224*.
- [G] **The fair detector baselines are Yolo-FastestV2 / FastestDet / NanoDet-m (all ShuffleNetV2-based) re-run at 224×224 on the same Pi 4 with ncnn.** They share YOLIC-S2's backbone, which addresses baseline (e) in the brief. Expected FLOPs at 224 (scaled by area from 352):
  - Yolo-FastestV2: about 0.086 GFLOPs.
  - FastestDet: in the same range as Yolo-FastestV2.
  - YOLIC-S2: 0.30 G per the brief.
  - So these detectors would likely be **faster** than YOLIC-S2 at 224. Accuracy at 224 on cell-rasterised boxes is the open question.
- [G] **The fair segmentation baseline is LR-ASPP-MobileNetV3-Small/Large or Fast-SCNN at a low input (e.g. 224×448 or 256×512), pooled into cells.** Torchvision ships LR-ASPP-MNV3-L, which lowers the implementation burden. Expected Pi 4 speed at about 0.1–0.3 MP input is plausibly 5–20 FPS: unverified, and must be measured. On the laptop CPU it can be reported as relative latency.
- [G] PIDNet-S, PP-LiteSeg and SeaFormer give the accuracy ceiling for "segmentation then pool" on Cityscapes. Their GPU-only speed reporting means a Pi-4 claim against them requires your own measurements.

### Gaps
- **No published Raspberry Pi 4 FPS found for:** LR-ASPP-MobileNetV3, Fast-SCNN, BiSeNetV2, PIDNet-S, PP-LiteSeg, TopFormer, SeaFormer, EfficientDet-Lite0, or YOLOv8n/YOLO11n under ncnn at 224–320.
- Ultralytics' Pi guide now covers only the Pi 5 and YOLO26. Older Ultralytics Pi 4 tables (if they existed) were not retrievable in this pass.
- Exact LR-ASPP Cityscapes mIoU from the MobileNetV3 paper's table (72.6 for MNV3-L at full res, from memory [G]) was not re-read; fastseg's 72.3 is the verified number.
- Qengineering numbers are a single hobbyist source. They are internally inconsistent for YOLOv5 and should be cross-checked if used as citations in a paper.

---

## Q5. Has a "1×1 conv on the spatial feature map + per-cell pooling" head (instead of GAP+FC) been shown to work for cell-presence tasks? (brief baseline (d), idea E1)

### Takeaway
I found **no paper that runs exactly this head on a YOLIC-style cell-presence task**. But its components are standard and well established:
- FC layers can be recast as convolutions (FCN).
- YOLOv2 removed YOLOv1's FC grid head in favour of a conv head.
- GAP+FC is mathematically the spatial mean of a 1×1-conv class map (CAM), so "1×1 conv then pool per cell" is the cell-restricted version of CAM.
- UFLD shows a third option: 1×1 conv, then flatten, then FC.

A reviewer will see E1 as an obvious, low-novelty but necessary baseline/ablation, not as a contribution on its own.

### Cited Findings
- **[V] YOLOv1 to YOLOv2:** YOLOv1 predicted a per-grid-cell tensor through fully connected layers; YOLOv2 states "We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes" (text extracted from the paper PDF). This is direct precedent that a conv head over the spatial grid replaces an FC per-cell head. — [arXiv 1612.08242](https://arxiv.org/abs/1612.08242)
- **[V] FCN (Long, Shelhamer, Darrell, CVPR 2015):** fully connected layers can be viewed as convolutions with kernels covering the whole input, and a 1×1 conv produces class scores at every coarse location. — [arXiv 1411.4038](https://arxiv.org/abs/1411.4038); [CVPR open access](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
- **[V] CAM (Zhou et al., CVPR 2016):** the GAP + linear classifier structure "enables the CNN to have remarkable localization ability despite being trained on image-level labels". [G] The equivalence GAP→FC ≡ 1×1 conv→GAP follows from linearity and is the CAM construction. — [arXiv 1512.04150](https://arxiv.org/abs/1512.04150)
- **[V] UFLD's head** (1×1 conv 512→8, flatten 1800, FC) is a working per-cell (row-anchor × gridding-cell) classifier that keeps spatial layout without GAP. — [UFLD model.py](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/model/model.py); [arXiv 2004.11757](https://arxiv.org/abs/2004.11757)
- **[V] SqueezeDet:** a fully convolutional detection head (ConvDet) predicts per grid location for driving; a small, low-power model. — [arXiv 1612.01051](https://arxiv.org/abs/1612.01051)
- **[V] Fully convolutional per-cell classification of occupancy grid maps** (LiDAR grid maps, not camera cells) replaces sliding-window patch classifiers. — [arXiv 1709.03139](https://arxiv.org/abs/1709.03139)
- **[V] The YOLIC paper itself names head growth with N as a limitation and runs no head ablation** (brief §3.3, §4.1, from `yolic_paper_text.txt`). — [YOLIC arXiv](https://arxiv.org/abs/2307.06689)

### Inferences
- [G] **Minimum head ablation for any YOLIC extension:**
  - (i) GAP+FC (YOLIC).
  - (ii) Flatten+FC (UFLD-style, with 1×1 channel reduction).
  - (iii) 1×1 conv + fixed cell-pooling matrix (E1 / baseline (d)), with mean and max pooling variants.
  - (iv) Optionally a tiny segmentation decoder (LR-ASPP) + cell pooling.
  - Report params, latency and per-cell-size F1.
- [G] Because the components are textbook (FCN, CAM, YOLOv2), E1's novelty claim must rest on *empirical findings specific to cell presence*: head size independent of N, small-cell accuracy, transfer across layouts (links to A1 and P1). It cannot rest on the architecture.
- [G] CAM-style "1×1 conv → per-cell max/noisy-OR pooling" is also exactly the supervision operator idea P1 needs. Prior art for P1 is the weakly-supervised segmentation / multiple-instance-learning literature (not searched here; see Gaps).

### Gaps
- I did not search the weakly-supervised segmentation / MIL pooling literature (e.g. WILDCAT, noisy-OR MIL pooling) that would be prior art for P1 and for "cell-pooled 1×1 conv" heads trained from region-level labels. This needs a separate pass.
- I found no published head ablation (GAP+FC vs spatial head) on any cell-presence or CoI dataset, including the YOLIC follow-ups. The Selective Multi-Branch paper (full text unread) may contain a partial one, since it "uses only part of the feature map".
