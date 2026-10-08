# 02 — Ideas extending YOLIC

Input: `research/01_brief.md`. Section references (§) point to that brief.
This pass favours breadth. Feasibility, novelty and priority are left for the separate review.

**Resource envelope assumed:**
- One RTX 4050 Laptop GPU (6 GB) and 7.6 GB RAM.
- Three months.
- Public data: Cityscapes (pixel masks), the YOLIC outdoor and indoor sets (cell labels only), and other public driving or segmentation sets.
- No Raspberry Pi unless one is bought. Ideas whose evidence depends on Pi latency say so.

**Discarded as plainly impossible with these resources:**
- Collecting new sensor data (LiDAR, IMU, stereo rigs).
- Pre-training or fine-tuning large foundation models.
- Anything that needs the full Cityscapes video sequences (~324 GB).
- Multi-device fleet or on-vehicle field trials.
- Re-labelling the outdoor set at pixel level by hand.

---

## Lens 1 — Assumption-breaking

### A1. Layout-agnostic YOLIC: one model, any cell configuration at test time
- **Claim:** A YOLIC head that takes the cell geometry as an *input* can serve layouts it never saw during training, without retraining or re-annotation.
- **Mechanism:**
  - Keep the backbone's spatial feature map (e.g. stride 16, 14×14) instead of global-average-pooling it.
  - For each cell, rasterise its polygon onto the feature grid and mask-average-pool the features. Add a small embedding of the cell's position and size.
  - A shared MLP maps each pooled cell feature to M+1 sigmoids.
  - Train on Cityscapes with *randomly sampled layouts every batch*: rectangles and polygons of varied sizes, with labels recomputed on the fly from the masks.
- **New capability / result:** A deployer can redraw cells in the field, e.g. after moving the camera or changing the task. Head size no longer grows with N (the paper's admitted limitation, §4.1). Accuracy can be measured on held-out layouts.
- **Data:** Cityscapes fine masks for training and evaluating unseen layouts. The outdoor/indoor sets serve as fixed-layout sanity checks only.

### A2. Ground-plane cells: from fixed pixel boxes to physically defined cells
- **Claim:** Defining cells in metric ground-plane coordinates and projecting them with the current camera pose keeps their physical meaning ("0–1 m ahead, left") when camera height, pitch or roll changes.
- **Mechanism:**
  - Specify cells as polygons on the ground plane (bird's-eye view).
  - Project them into the image with the per-image intrinsics and extrinsics, so labels and pooling regions follow the pose.
  - Condition the model on the pose: either feed pitch, roll and height as an input embedding, or pool features through the projected cells as in A1.
  - Simulate pose changes during training with homography warps of the images and masks.
- **New capability / result:** One model and one cell definition that survive remounting and a vehicle pitching over bumps. This directly removes the "fixed camera geometry" assumption (§1.3 #1). Accuracy can be reported as a function of pose perturbation.
- **Data:** Cityscapes masks plus its released per-image camera calibration (intrinsics and extrinsics), and synthetic homography perturbations.

### A3. Temporal YOLIC: from independent frames to a recurrent cell state
- **Claim:** Carrying each cell's state across frames with a tiny recurrent update improves recall and stability at almost no added compute. The paper names this only as future work (txt:876-877).
- **Mechanism:**
  - Keep a hidden vector per cell, or the previous logits.
  - Fuse it with the current frame's cell features through a gated update (a GRU cell shared across cells).
  - Train on short clips with labels on the annotated frames.
  - Optionally warp the previous state by ego-motion before fusing.
- **New capability / result:** Fewer flickering cell decisions and fewer single-frame misses on safety classes. A measurable temporal-consistency metric, alongside per-frame F1.
- **Data:** A video dataset with sparse labels.
  - CamVid: 30 Hz video, labelled at 1 Hz, small download.
  - Or the outdoor/indoor sets, if frame order can be recovered from filenames (open question 5 in §6).
  - Cityscapes sequences are excluded for size.

---

## Lens 2 — Practitioner pain

### P1. Change the layout without re-annotating: dense occupancy learned from cell labels
- **Claim:** Coarse cell labels contain enough signal to train a dense (fine-grid) occupancy map. That map can be re-aggregated into any new layout, so old annotations transfer to a new configuration.
- **Mechanism:**
  - Predict a fine grid, e.g. 28×28 × M classes.
  - Supervise it only through a differentiable cell-aggregation operator that matches the label rule: max- or noisy-OR-pooling inside each cell, which mimics "≥1 pixel present".
  - At deployment, aggregate the fine map through the new layout's cells.
  - Validate on Cityscapes: train with layout L1 labels, then test on layout L2, whose ground truth comes from the masks.
- **New capability / result:** The costly part of deploying YOLIC, re-labelling for each new cell design (§4.2), goes away. As a by-product, the model gives coarse segmentation maps learned from cell-level supervision alone.
- **Data:** Cityscapes, where the L1/L2 ground truth is derivable. The outdoor cell labels serve as a qualitative demonstration.

### P2. Recall-guaranteed thresholds for safety classes
- **Claim:** Replacing the fixed 0.5 threshold with per-class (or per-cell-group) thresholds chosen by conformal risk control gives a user-specified bound on the missed-hazard rate.
- **Mechanism:**
  - Optionally apply temperature scaling to the sigmoids first.
  - On a held-out calibration split, use conformal risk control to pick, for each class (or for near/far cell groups), the largest threshold whose expected miss rate stays at or below α.
  - Report the precision that each recall guarantee costs.
  - Check how sensitive the guarantee is to a calibration split that isn't exchangeable with the test data, e.g. a different city.
- **New capability / result:** Operating points a practitioner can actually sign off on, e.g. "≤5% missed people in the 0–2 m cells", instead of uncalibrated 0.5 outputs. This addresses the unaddressed recall issue in §4.2.
- **Data:** Existing outdoor or Cityscapes predictions plus a calibration split. Cityscapes cities can be used for exchangeability tests.

### P3. Automatic cell-layout design under a cell budget
- **Claim:** Optimising the layout itself, i.e. where to place cells and how big to make them for a given count N, beats hand-designed layouts at equal N.
- **Mechanism:**
  - Start from a fine grid over the region of interest.
  - Greedily merge or split cells to maximise an objective computed from the masks, e.g. expected per-cell label purity, object-footprint coverage, or the F1 of a quick proxy model, subject to N cells.
  - Optionally add distance weighting from camera calibration.
  - Retrain YOLIC on the found layouts and compare with the paper-style hand layout at the same N.
- **New capability / result:** A replacement for the manual "Cell Designer" step (txt:347-350), giving a principled layout for a new camera or task, and an accuracy-vs-N curve for designed vs hand layouts.
- **Data:** Cityscapes masks, which let any layout be relabelled for free.

---

## Lens 3 — Supervision

### S1. Zero-manual-label YOLIC via open-vocabulary auto-labelling
- **Claim:** Cell labels produced automatically by off-the-shelf open-vocabulary detectors or segmenters train a YOLIC model that comes close to one trained on human cell labels.
- **Mechanism:**
  - Run a text-prompted detector plus a promptable segmenter (e.g. small Grounding-DINO + SAM ViT-B checkpoints) offline on the frames, prompting with the class names ("traffic cone", "weed", "bump", …).
  - Rasterise the masks into cell labels with the same "≥ k pixels" rule.
  - Train YOLIC on the pseudo-labels, optionally filtered by confidence.
  - Compare against the human labels on the outdoor test split.
- **New capability / result:** Deploying to a new site or class set without manual annotation. A per-class map of which hazards auto-labelling handles and which still need humans (e.g. "dent").
- **Data:** Unlabelled outdoor frames, with the outdoor human cell labels used only for evaluation. Public model checkpoints that fit offline inference on 6 GB.

### S2. Coverage-weighted soft labels instead of "any pixel"
- **Claim:** Training on the *fraction* of each cell covered by a class, instead of the brittle ≥1-pixel binary label (§2.1 D13), reduces boundary label noise and improves small-object cells.
- **Mechanism:**
  - Compute the per-cell, per-class area fraction from the masks.
  - Use it as a soft BCE target, or as a two-part target: presence plus regressed coverage.
  - Evaluate presence with a minimum-area rule, and evaluate coverage separately (MAE).
  - Ablate the area threshold used to define "present".
- **New capability / result:** Cleaner supervision, and a new per-cell output ("how much of the cell is blocked"), which gives a coarse shape and extent estimate the binary head cannot.
- **Data:** Cityscapes masks. The outdoor set has no masks, so it only provides binary evaluation.

### S3. Semi-supervised YOLIC with equivariant cell consistency
- **Claim:** With only 5–20% of frames labelled, consistency training on unlabelled frames, using *correct* geometric equivariance of cell labels, recovers most of the fully supervised accuracy.
- **Mechanism:**
  - Mean-teacher or FixMatch-style training.
  - Weak and strong views of each unlabelled frame. The horizontal-flip view is paired with the mirrored cell permutation (fixing the vertical-flip bug, D1).
  - Confidence-thresholded per-cell pseudo-labels from the teacher supervise the student.
  - Sweep the labelled fraction.
- **New capability / result:** A label-efficiency curve for YOLIC, showing how many labelled frames a new deployment actually needs. It also directly tests whether cell labels make good targets for semi-supervised learning.
- **Data:** Outdoor and indoor sets with part of the labels hidden. Cityscapes in the same way.

---

## Lens 4 — Efficiency / edge systems

### E1. N-independent head: a fixed cell-pooling matrix plus a shared classifier
- **Claim:** Replacing the N×(M+1)-wide FC layer with a fixed sparse "cell pooling" matrix over the backbone feature map plus a shared 1×1 classifier cuts head parameters by orders of magnitude, with no loss in accuracy.
- **Mechanism:**
  - Precompute P ∈ ℝ^{N×HW}, where each row is a cell's normalised area-overlap with the feature-grid locations.
  - Head = shared 1×1 conv (C→M+1) on the H×W map, followed by one matmul with P, plus a per-cell bias.
  - The deployed result is a single small GEMM.
  - Compare parameters, memory, latency on the laptop CPU (and on a Pi if one is available), and accuracy against GAP+FC.
- **New capability / result:** Head cost that does not grow with the number of cells, which removes the admitted limitation (§4.1). It also allows many more or finer cells on the same device.
- **Data:** Outdoor, indoor and Cityscapes. Absolute Pi latency claims need a Pi 4B; otherwise report relative latency on the laptop CPU.

### E2. Look *only* at the interested cells: crop and foveated resampling of the input
- **Claim:** Spending the fixed 224×224 pixel budget only on the region the cells cover, with more resolution for small distant cells, raises accuracy at the same FLOPs.
- **Mechanism:**
  - Instead of squashing the whole frame, apply a fixed, layout-derived warp before the backbone. It drops uncovered areas (e.g. the top 320 px and the sides in Cityscapes) and magnifies far cells while shrinking near cells, e.g. a piecewise-linear or thin-plate warp computed once from the cell layout.
  - Cell labels are unchanged by the warp because they are defined per cell.
  - Compare with the original squash at equal input size, and at smaller input sizes (160, 128).
- **New capability / result:** Better small-object cells (e.g. Cityscapes People, F1 0.60–0.67, §3.4), or the same accuracy at a lower input size and higher FPS. The name "only look at interested cells" would then also describe the input.
- **Data:** Cityscapes and the outdoor set. Neither needs new labels.

### E3. Uncertainty-gated cascade (and frame skipping) for edge budgets
- **Claim:** A tiny first-stage model handles most frames, and a larger model runs only when cell-level uncertainty is high, giving close to large-model accuracy at close to small-model average latency.
- **Mechanism:**
  - Stage 1 is a very small backbone or a low input resolution (e.g. MobileNetV3-Small at 128–160 px).
  - Compute an uncertainty score over safety-relevant cells (entropy, or margin to threshold).
  - Above a threshold, run stage 2 (e.g. YOLIC-M2).
  - Optionally skip frames whose stage-1 features barely change from the previous frame.
  - Sweep the threshold to trace an accuracy-vs-average-compute curve.
- **New capability / result:** An adjustable accuracy/latency knob for a fixed device, and a measure of how often the "hard" path is needed in each dataset.
- **Data:** Outdoor, indoor and Cityscapes. Frame skipping needs ordered video (CamVid, or the outdoor set if order is recoverable). Pi timings are optional; otherwise use laptop-CPU proxy timings.

---

## Lens 5 — Robustness and generalisation

### R1. Adverse-condition transfer: Cityscapes → ACDC (fog, night, rain, snow)
- **Claim:** YOLIC trained on clear-weather Cityscapes degrades in specific, measurable ways under adverse conditions, and cell-level adaptation (e.g. style augmentation or BN re-estimation) recovers part of the loss.
- **Mechanism:**
  - Derive cell labels for ACDC from its Cityscapes-compatible masks, using a layout re-fitted to ACDC's camera framing.
  - Evaluate the Cityscapes-trained model per condition.
  - Compare remedies: photometric and style augmentation, test-time BN statistics updates, and fine-tuning on a few labelled adverse images.
  - Report per-condition and per-cell-region drops.
- **New capability / result:** The first robustness characterisation of cell-wise detection under adverse weather, which the paper does not test (§3.3), plus a practical recipe for adverse conditions.
- **Data:** Cityscapes plus ACDC (public after registration, Cityscapes label protocol).

### R2. City-held-out generalisation and test-time adaptation
- **Claim:** Accuracy on random frame splits overstates deployment accuracy. Leave-cities-out evaluation shows the real generalisation gap, and lightweight test-time adaptation narrows it.
- **Mechanism:**
  - Use Cityscapes' city metadata to build leave-k-cities-out splits.
  - Measure the drop relative to an iid split with the same number of training images.
  - Apply test-time adaptation methods that suit edge devices: BN-statistics re-estimation, and entropy minimisation on the head only (TENT-style).
  - On the outdoor set, approximate held-out "scenes" by clustering frames with perceptual hashes.
- **New capability / result:** An honest estimate of how YOLIC transfers to a new site (§3.3 leakage concern), and a cheap adaptation step for new deployments.
- **Data:** Cityscapes (city IDs are in the filenames) and the outdoor set with clustered pseudo-scenes.

### R3. Corruption robustness × quantization
- **Claim:** INT8 quantization amplifies YOLIC's sensitivity to common camera corruptions (blur, noise, JPEG, low light), and corruption-aware QAT removes this extra drop.
- **Mechanism:**
  - Build a "Cityscapes-C / Outdoor-C" benchmark with standard corruption functions at 5 severities.
  - Evaluate the FP32 and INT8 (QAT or PTQ, PyTorch x86 or ONNX Runtime) models.
  - Compare the relative drops.
  - Retrain QAT with corruption augmentation.
  - Analyse per-class and per-cell-size effects.
- **New capability / result:** Evidence on whether the paper's "quantization keeps accuracy" result (§3.4) holds under realistic edge-camera degradation, and a fix if it does not.
- **Data:** Cityscapes and outdoor test sets with synthetic corruptions. No new labels needed.

---

## Lens 6 — Analysis

### N1. How does a GAP+FC head localise anything?
- **Claim:** YOLIC's cell localisation relies on absolute-position cues that leak through zero-padding and image borders. Removing these cues breaks it in predictable ways.
- **Mechanism:**
  - Probe trained models with:
    - (a) translated or cropped inputs, to check whether predictions follow the content or stay pinned to the cells;
    - (b) padding swaps (zero vs reflect vs circular) at test time and at train time;
    - (c) linear probes for cell coordinates on the pooled features;
    - (d) per-cell accuracy vs cell size, position and distance from the border.
  - Compare against a spatial head (E1) under the same probes.
- **New capability / result:** A mechanistic account of an unexplained behaviour (§1.2, §1.3 #3), and design rules for when GAP heads are acceptable and when they fail, e.g. small central cells.
- **Data:** Cityscapes, where any layout and perturbation can be relabelled. The outdoor set provides a second domain.

### N2. What is the real gap? Re-evaluating YOLIC vs detectors under a common protocol
- **Claim:** Once the outdoor results are rerun with the augmentation bug fixed, leakage-free splits, several seeds and a shared per-cell metric for every method, they look different from the paper's tables.
- **Mechanism:**
  - Fix D1 and D2.
  - Build near-duplicate-aware splits with perceptual-hash clustering.
  - Run 3–5 seeds.
  - Train YOLO-N/S properly. On Cityscapes, also train YOLO on *real object boxes* from the instance masks and rasterise its detections into cells.
  - Score every method with the same per-cell presence metric, threshold sweeps and AP.
- **New capability / result:** A trustworthy benchmark and confidence intervals for this class of methods, settling the overstated claims flagged in §3.4.
- **Data:** Outdoor set and Cityscapes (instance masks give real boxes).

### N3. Granularity scaling: accuracy as a function of cell size and count
- **Claim:** Per-cell accuracy follows a predictable curve in cell size relative to backbone stride and input resolution, with a collapse point that tells designers the smallest usable cell.
- **Mechanism:**
  - Generate a family of regular and irregular layouts on Cityscapes, varying N from about 16 to 1024 and the cell size in feature-map units.
  - Train with a fixed recipe across 2–3 input resolutions and two backbones.
  - Fit accuracy against effective cell size (in pixels at the input and in feature cells).
  - Locate where the head or the feature resolution becomes the bottleneck.
- **New capability / result:** Design guidance missing from the paper, which has no ablation over cells (§3.3): given a camera and a backbone, how fine a layout can be trusted.
- **Data:** Cityscapes masks.

---

## Lens 7 — Cross-pollination

### C1. Losses from multi-label image classification for extreme cell imbalance
- **Claim:** Losses designed for imbalanced multi-label classification (asymmetric loss, focal-style down-weighting of easy negatives, logit adjustment by class prior) improve rare-class and small-cell recall over YOLIC's plain BCE.
- **Mechanism:**
  - Replace BCE with ASL, focal loss, or prior-corrected BCE, using per-class (and optionally per-cell) positive rates computed from the training labels.
  - Keep everything else fixed.
  - Report macro-F1 and AP on rare classes (traffic sign, traffic cone, people) and on small cells.
  - Add label-correlation modelling as an optional variant: a tiny class-graph or transformer decoder over the M+1 outputs of each cell.
- **New capability / result:** Better performance on the hard classes the paper reports as weakest (traffic sign recall ≈ 0.6–0.7, Cityscapes People), with zero inference cost.
- **Data:** Outdoor, indoor and Cityscapes.

### C2. Distilling a dense segmentation teacher into a cell student
- **Claim:** A Cityscapes-trained semantic segmentation network, pooled into cells, is a strong teacher. Distilling it into YOLIC, including on unlabelled images from other datasets, beats training on hard labels alone.
- **Mechanism:**
  - Run a public Cityscapes segmentation checkpoint (e.g. a SegFormer or DeepLab variant) offline.
  - Convert its per-pixel class probabilities into soft per-cell targets with the same aggregation rule.
  - Train YOLIC on a mix of ground-truth cell labels and teacher targets (KL or BCE on the soft targets).
  - Extend training to unlabelled driving images from other datasets for domain breadth.
- **New capability / result:** A path to more accurate cell models from cheap unlabelled data. This is also the dense-segmentation baseline the paper is missing (§3.1).
- **Data:** Cityscapes, public segmentation checkpoints, and optional unlabelled driving images (e.g. from BDD100K or Mapillary after registration).

### C3. Structured output over the cell graph (CRF / message passing)
- **Claim:** Treating cells as nodes in a spatial graph and adding a few rounds of learned message passing (or a mean-field CRF) makes neighbouring cells consistent and improves object-extent recovery.
- **Mechanism:**
  - Build an adjacency graph from the layout (shared edges or distance), which works for irregular polygons too.
  - After the cell features or logits, run 1–3 rounds of a light GNN or a mean-field CRF with learned pairwise compatibilities between classes.
  - Train end to end.
  - Compare boundary-cell accuracy and the fragmentation of predicted object regions.
- **New capability / result:** Spatially coherent outputs, i.e. fewer isolated false-positive cells and fewer holes inside objects, borrowed from structured prediction in segmentation and grid-based occupancy mapping. It applies to any layout, including the irregular indoor one.
- **Data:** Outdoor, indoor and Cityscapes.
