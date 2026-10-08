# Prior art and reviewer baselines for the layout and geometry ideas (A1, A2, P1, P3, E1)

Tag legend used throughout:
- **[V-full]**: I opened the primary source and read the relevant passage.
- **[V-abs]**: I opened the abstract or landing page only.
- **[V-snip]**: Seen only in a search-result snippet. I did not open the source, so treat it as likely but unconfirmed.
- **[G]**: Guessed or recalled from background knowledge and not opened in this session. The link is given so it can be checked. Do not cite it as verified.
- **[I]**: My own inference or arithmetic from the cited facts.

Context, from `research/01_brief.md`: YOLIC = backbone + GAP + one FC layer giving N×(M+1) sigmoids. Cells are fixed pixel boxes. Each cell's label is "≥1 pixel of the class present". On Cityscapes there are 256 cells and 4 bits (People/Vehicle/Other/Road), and the upper cells are about 7×3.5 px at a 224 input.

---

## Q1. A1 / E1: is "pool features inside arbitrary regions or masks, then apply a shared classifier" already standard? Does "light segmentation net, then average per cell" give layout-agnosticism for free? Is E1 just RoI pooling? Does E1 differ from A1?

### Takeaway
Yes. Region or mask pooling plus a shared classifier is textbook and current practice:
- RoI pooling in detection (Fast R-CNN / Mask R-CNN lineage);
- mask pooling in open-vocabulary segmentation (OpenSeg, FC-CLIP);
- SAM+DINOv2 region features (2024);
- anchor-based feature pooling in lane detection (LaneATT).

Because a 1×1 classifier and average pooling are both linear, E1 is mathematically "a dense FCN logit map averaged inside fixed cells". That is the CAM identity, and it is the same as A1 with a linear head and no position embedding. A segmentation network trained on pixel masks and then aggregated per cell is layout-agnostic by construction. It is the obvious, and probably dominant, baseline for A1, and the paper already lacks it (brief §3.1(a)).

### Cited Findings
- **Mask pooling in open-vocabulary segmentation [V-full].**
  - FC-CLIP's in-vocabulary classifier gets class embeddings "by mask pooling over the final pixel features from pixel decoder".
  - Its out-of-vocabulary classifier "applies mask pooling to the frozen CLIP backbone features".
  - FC-CLIP credits mask pooling to Ghiasi et al., "Scaling open-vocabulary image segmentation with image-level labels" (OpenSeg, ECCV 2022).
  - Source: [FC-CLIP, arXiv 2308.02487 (HTML)](https://arxiv.org/html/2308.02487)
- **Region features from masks plus a linear decoder (2024) [V-abs].** Pairing class-agnostic segmenters (SAM) with DINOv2 features gives region representations that work "even with linear decoders" for semantic segmentation, retrieval and video. A search snippet says features are pooled within each mask and that average pooling beat max pooling [V-snip]. — [Region-Based Representations Revisited, arXiv 2402.02352](https://arxiv.org/abs/2402.02352)
- **Mask-based attention pooling (2025) [V-snip].** TextRegion generates SAM2 soft masks and performs "mask-based attention pooling" over downsampled masks on frozen image-text features. — [TextRegion, arXiv 2505.23769](https://arxiv.org/pdf/2505.23769)
- **Anchor-based feature pooling with a light backbone (lane detection) [V-snip].** LaneATT is "an anchor-based deep lane detection model which uses anchors for the feature pooling step" and runs at up to 250 FPS. — [LaneATT, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.pdf)
- **UFLD is the lane-detection twin of YOLIC's GAP+FC design [V-abs].** UFLD treats lane detection "as a row-based selecting problem using global features", reaching 300+ FPS. — [UFLD, arXiv 2004.11757](https://arxiv.org/abs/2004.11757)
  - UFLDv2 (TPAMI 2022) uses hybrid row and column anchors with ordinal classification [V-snip]. — [UFLDv2, arXiv 2206.07389](https://arxiv.org/pdf/2206.07389)
  - Relevance: a fixed set of image-space anchors, each classified from global features, is already an established fast design that predates YOLIC.
- **Column-wise obstacle classification from a monocular camera [V-snip].**
  - StixelNet (BMVC 2015) reduces "closest obstacle in each direction" to a column-wise problem solved by a CNN, with labels generated automatically from LiDAR. — [BMVC 2015](https://www.bmva.org/bmvc/2015/papers/paper109/)
  - 2024 successor StixelNExT [V-abs]: predicts a multi-layer Stixel world from monocular images, trained from LiDAR, and described as "low-weight". — [arXiv 2407.08277](https://arxiv.org/abs/2407.08277)
- **"Segment, then aggregate over a grid of cells" already exists for robot obstacle avoidance [V-snip].** A 2023 MDPI paper runs binary semantic segmentation (FCN-VGG16), divides the image into grid cells, and plans paths over cell centres. — [Electronics 12(8):1932](https://www.mdpi.com/2079-9292/12/8/1932)
- **How cheap a lightweight Cityscapes segmentation baseline is [V-full]** (MobileNetV3 paper, Table 7/8):
  - LR-ASPP + MobileNetV3-Small: **68.38 mIoU** on Cityscapes val, 0.47M params, **0.74B MAdds at 512×1024**, 327 ms on one Pixel-3 core (Table 7 row 11; Table 8 gives 69.4 on test at OS16).
  - LR-ASPP + MobileNetV3-Large: 72.37 mIoU val, 10.33B MAdds at full resolution.
  - Source: [MobileNetV3, arXiv 1905.02244](https://arxiv.org/pdf/1905.02244)
- **RoI pooling / RoIAlign, CAM, deformable RoI pooling [G]: foundational, not opened here.**
  - Fast R-CNN RoI pooling: [arXiv 1504.08083](https://arxiv.org/abs/1504.08083)
  - Mask R-CNN RoIAlign: [arXiv 1703.06870](https://arxiv.org/abs/1703.06870)
  - CAM, which shows that GAP followed by a linear layer equals averaging a per-location class map: [arXiv 1512.04150](https://arxiv.org/abs/1512.04150)
  - Deformable ConvNets / deformable RoI pooling: [arXiv 1703.06211](https://arxiv.org/abs/1703.06211)

### Inferences
- **[I] E1 is textbook. It reduces to a known identity.**
  - Let F be the H×W×C feature map, W the shared 1×1 weights (C→M+1), and P the fixed cell-pooling matrix.
  - Then P·(F·W) = (P·F)·W. "Average features in the cell, then apply a linear classifier" is identical to "compute a dense per-location logit map, then average it in each cell".
  - That is a fully convolutional segmentation head with average-pooling multiple-instance aggregation per cell. In the CAM sense, it is YOLIC's GAP+FC with the GAP replaced by N cell-wise average pools.
  - It is also exactly a fixed-RoI average-pooling layer (RoIPool with one bin per RoI and average instead of max).
  - A reviewer will call it "a 1×1-conv segmentation head with RoI average pooling". The novelty would have to come from a measured edge result, not from the head itself.
- **[I] E1 vs A1.** E1 is a special case of A1:
  - linear head instead of MLP;
  - no position or size embedding;
  - a fixed layout instead of random layouts at training time.

  The two differ only in (i) whether the classifier can be non-linear and position-aware, and (ii) the training-layout distribution. If both are proposed, merge them into one idea: E1 is the zero-embedding ablation of A1.
- **[I] E1's parameter claim is true but trivial.**
  - Head weights: YOLIC-M2 Cityscapes = 1280×1024 ≈ 1.31M (outdoor 1280×1248 ≈ 1.60M). E1 = 1280×4 = 5,120 + N biases.
  - So yes, it is orders of magnitude smaller. But the backbone (about 2.2M for MobileNetV2) then dominates, so total model size drops by only about 35–40%. FPS will barely change, because the FC GEMM is a small share of 0.3 GFLOPs.
  - Claim "N-independent head", not "faster".
- **[I] The key technical failure mode for A1 and E1 is feature resolution versus cell size.**
  - At 224 input with a stride-32 backbone (7×7 map), each feature location covers 32×32 input px.
  - Cityscapes upper cells are about 7×3.5 px at input, so many adjacent cells fall inside the *same* feature location. They get identical pooled features.
  - With a shared classifier, they then get identical outputs except for the per-cell bias.
  - GAP+FC sidesteps this with *per-cell* weights over a global vector, which can exploit implicit absolute position.
  - E1/A1 therefore probably **lose** on small cells unless they use stride-8/16 features (e.g. MobileNetV2 at the 14×14 or 28×28 stage) or a higher input resolution. That costs FLOPs and erodes the edge argument.
  - Predicted pattern: better on large near cells, worse on small far cells, where People already scores worst (F1 0.60–0.67).
- **[I] Shared classifiers lose cell-specific priors.** YOLIC's per-cell weights encode "people never appear in the top-left cell" and similar priors. A shared head only gets these back through a position embedding (A1) or a per-cell bias (E1). This argues for keeping A1's embedding, but the embedding also re-introduces layout dependence, which is a tension A1 has to report.
- **[I] Layout-agnosticism for free.**
  - Option 1: train any light segmentation net (e.g. LR-ASPP-MobileNetV3-Small) on Cityscapes pixel masks and apply YOLIC's rule (max over pixels, or a pixel-count threshold, of the predicted class in the cell). That serves any layout at test time with no retraining.
  - Option 2: do the same with a stride-8 logit map, which A1's training objective essentially reproduces.
  - Rough cost at 224×224: 0.74B MAdds × (224²/(512·1024)) ≈ 0.07B MAdds, below YOLIC-S2's reported 0.30 GFLOPs. Accuracy at that resolution is unknown.
  - **A1 is only interesting if it beats this baseline** at equal compute on held-out layouts, *or* if it works where pixel masks don't exist (only cell labels, e.g. the outdoor set). In that second case it collapses into P1.
- **[I] Verdict sketch.**
  - A1 is engineering-incremental unless reframed as "train from cell labels only, serve any layout". That framing is P1.
  - E1 is a clean, small ablation (and a good N1 analysis tool), but not a standalone novelty.

### Gaps
- I did not open the Fast R-CNN, Mask R-CNN, CAM, MaskCLIP or ODISE papers in this session. They are listed as [G] foundational.
- I did not verify LaneATT's anchor-selection details (whether anchors are filtered by training-set frequency).
- No published accuracy exists for LR-ASPP or other light segmentation nets at a 224×224 input on Cityscapes. It must be measured.
- I found no paper that trains a cell-level detector with *randomly sampled layouts* as data augmentation. Absence of evidence, after limited searching.

---

## Q2. A2: prior art on BEV / ground-plane cells, pose conditioning and homography augmentation. Does Cityscapes ship per-image camera files? Is pitch constant?

### Takeaway
- Monocular BEV occupancy is a mature field: PON 2020, Translating Images into Maps 2022, LSS 2020. It is trained on nuScenes/Argoverse with 3D/LiDAR ground truth, not on Cityscapes.
- Camera-parameter-conditioned networks (CAM-Convs 2019) and pose-perturbation robustness (Klinghoffer ICCV 2023; MonoGAE; 3DRot 2025) exist.
- Cityscapes does ship one `*_camera.json` per image (intrinsics plus extrinsic pitch, roll, yaw and height). But the values are calibration constants per recording session. Pitch spans only about 0.038–0.05 rad (about 2.2°–2.9°) across the test set, so **real pose variation can't be tested on Cityscapes. Only synthetic warps can.**

### Cited Findings
- **Cityscapes ships per-image camera files [V-full].**
  - File list of the camera package: e.g. `camera/train/aachen/aachen_000000_000019_camera.json` — [sakaridis/fog_simulation-SFSU_synthetic file list (GitHub)](https://github.com/sakaridis/fog_simulation-SFSU_synthetic) (found via `gh search code`).
  - Contents of one such file, copied verbatim into another repo: `extrinsic: {baseline 0.209313, pitch 0.038, roll 0.0, x 1.7, y 0.1, yaw -0.0195, z 1.22}`, `intrinsic: {fx 2262.52, fy 2265.30, u0 1096.98, v0 513.137}` — [Citynthesizer data/camera.json, "originally: aachen_000000_000019_camera.json"](https://github.com/pilkonimo/Citynthesizer)
- **Coordinate conventions [V-full].** The official calibration doc defines vehicle (ISO 8855, origin on the ground below the rear-axle centre), camera and image frames. It builds R from yaw/pitch/roll "extrinsic" and t from x/y/z. — [csCalibration.pdf](https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf)
  - The README only says the package holds "internal and external camera calibration. For details, please refer to csCalibration.pdf". — [cityscapesScripts README](https://github.com/mcordts/cityscapesScripts)
- **Pitch barely varies in Cityscapes [V-full].** Butt & Taj tabulate Cityscapes test-set (1,525 images) camera parameters:
  - θp (pitch) min 0.038 / mean 0.041 / max 0.05;
  - tz (height) 1.18 / 1.230 / 1.3;
  - tx 1.7 / 1.699 / 1.7.
  - Source: [Multi-task learning for camera calibration, arXiv 2211.12432](https://arxiv.org/pdf/2211.12432)
- **Calibration per recording session [V-snip].** The Cityscapes paper says the rig was "recalibrated on-site before each recording session". — [Cordts et al. CVPR 2016](https://arxiv.org/pdf/1604.01685)
- **Monocular BEV semantic occupancy [V-snip].**
  - PON (Roddick & Cipolla, CVPR 2020) predicts BEV semantic occupancy grids from monocular images with a semantic Bayesian occupancy-grid framework. Evaluated on nuScenes and Argoverse. — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Roddick_Predicting_Semantic_Map_Representations_From_Images_Using_Pyramid_Occupancy_Networks_CVPR_2020_paper.html)
  - Saha et al. (ICRA 2022) frame image→BEV as a translation problem, assuming a 1-1 correspondence between image vertical scanlines and BEV rays. — [arXiv 2110.00966](https://arxiv.org/abs/2110.00966)
  - [G] Lift-Splat-Shoot, [arXiv 2008.05711](https://arxiv.org/abs/2008.05711), and MonoScene, [arXiv 2112.00726](https://arxiv.org/abs/2112.00726): not opened.
- **Camera-aware networks [V-snip].** CAM-Convs (Facil et al., CVPR 2019) feed camera intrinsics into convolutions so single-view depth generalises across cameras. — [arXiv 1904.02028](https://arxiv.org/abs/1904.02028)
- **Sensitivity to viewpoint change [V-snip].**
  - Klinghoffer et al. (ICCV 2023, NVIDIA) find BEV segmentation models "surprisingly sensitive" to small pitch, yaw, depth and height changes. A 10° pitch reduction caused a 17% IoU drop.
  - Their fix is novel-view synthesis to re-render source data to the target rig, tested on CARLA with 36 viewpoints.
  - Sources: [CVF paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Klinghoffer_Towards_Viewpoint_Robustness_in_Birds_Eye_View_Segmentation_ICCV_2023_paper.pdf); [project page](https://nvlabs.github.io/viewpoint-robustness/)
- **Roll/pitch perturbation augmentation [V-snip].** MonoGAE (roadside mono-3D) adds random roll and pitch offsets during training. It reports that ground-plane-equation maps beat ground-depth maps by 8.09% under roll/pitch disturbance. — [arXiv 2310.00400](https://arxiv.org/pdf/2310.00400)
- **Rotation about the optical centre needs no depth [V-abs].** 3DRot (2025) "rotates and mirrors images about the camera's optical center while synchronously updating RGB images, camera intrinsics, object poses, and 3D annotations to preserve projective geometry". Validated on SUN RGB-D, NYUv2 and KITTI. — [arXiv 2508.01423](https://arxiv.org/abs/2508.01423)
- **Camera-height robustness is an active 2025 topic [V-snip, title only].** — [CHARM3R, arXiv 2508.11185](https://arxiv.org/pdf/2508.11185)

### Inferences
- **[I] Pure pitch/roll changes are an exact homography** (rotation about the optical centre: H = K R K⁻¹). So synthetic pitch/roll augmentation and evaluation are geometrically exact for the whole image, as 3DRot relies on.
- **[I] Height changes and translations are *not* a single homography for non-planar scenes.**
  - The ground-plane homography warps the road correctly but distorts people and cars.
  - Klinghoffer et al. needed novel-view synthesis for height and depth shifts.
  - A2's "homography augmentation" is therefore only valid for pitch/roll. Height robustness can't be tested honestly on Cityscapes.
- **[I] Projecting a ground polygon into the image is not ground-plane occupancy.**
  - A person 10 m ahead covers image pixels far *above* their ground footprint. Pixels inside the projected footprint polygon contain their feet *plus* any nearer, taller occluder.
  - So "≥1 pixel inside the projected cell" mislabels occupancy, both for occlusion and for tall objects spilling into "farther" cells.
  - Correct ground-plane labels need footprint or contact points, e.g. the bottom of the instance mask, or Cityscapes 3D boxes. True BEV labels need 3D data, which is why PON and LSS train on nuScenes/Argoverse.
- **[I] Flat-ground assumption.** Projected cells drift on slopes and crowns. That is the same failure IPM-based free-space detection has.
- **[I] On Cityscapes the camera is effectively fixed.**
  - Pitch range ≈ 0.7°. Height range ≈ 12 cm across sessions, and probably within a session the values are constant.
  - Pose conditioning would get almost no real signal. Every pose result would come from synthetic warps of the same images.
  - A reviewer will ask for real-pose evaluation, i.e. a dataset with per-frame IMU pitch.
    - [G] KITTI raw ships per-frame OXTS IMU roll/pitch.
    - [G] nuScenes ships per-frame ego pose.
    - Neither has been opened or verified here.
- **[I] Obvious strongest baselines:**
  - (a) YOLIC trained with the *same* rotation-homography augmentation but without pose input. This isolates "conditioning" from "augmentation".
  - (b) "Undo the pose": warp the test image back to the canonical pose with H = K R⁻¹ K⁻¹ (given the pose A2 assumes it has) and run the unchanged fixed-layout model. This test-time rectification baseline is trivial and probably very strong for pitch/roll.
  - (c) A segmentation net plus projected-cell aggregation.

  If (b) works, conditioning the network on pose adds little. The novelty is then just "IPM-defined cells", which is classical.
- **[I] Verdict sketch.** The idea is sound as an application, but its novel parts are thin. Pose conditioning has CAM-Convs. Viewpoint robustness has Klinghoffer 2023. Rotation augmentation has 3DRot/MonoGAE. BEV cells have PON and IPM. And the dataset can't exercise real pose change.

### Gaps
- I did not open every Cityscapes camera JSON. "Constant within a session" is inferred from the narrow test-set range plus the per-session calibration statement, not checked file by file.
- I did not verify whether Cityscapes `vehicle/*.json` contains per-frame IMU pitch. The README says "vehicle odometry, GPS coordinates, and outside temperature" ([cityscapesScripts](https://github.com/mcordts/cityscapesScripts)); IMU pitch is not mentioned.
- I did not open a specific IPM free-space detection paper (classical; no citation verified).
- Not verified: whether KITTI or nuScenes per-frame pose would let real pitch-variation evaluation reuse Cityscapes-style semantic masks. KITTI has only about 200 semantic training images [G].

---

## Q3. P1: MIL / weakly supervised dense maps from cell labels. Closest prior work, known successes and failures, and whether full supervision trivially wins.

### Takeaway
P1 is patch- or tile-level weakly supervised segmentation, i.e. multiple-instance learning with each cell as a bag. Max, log-sum-exp and noisy-OR pooling for this date to 2015 (Oquab; Pinheiro & Collobert), with WILDCAT in 2017. Patch-level → pixel segmentation is an established subfield in histopathology (WSSS4LUAD challenge; Han et al. MedIA 2022) and remote sensing (tile-level labels; coarse-to-fine label resolution).

Known failure: max-pooling and CAM-style MIL localise discriminative parts, not full extents. On Cityscapes, a fully supervised segmentation net is the upper bound and will win. P1 is only meaningful where masks don't exist (the outdoor set), and there only coarsening layouts (unions of existing cells) can be scored quantitatively.

### Cited Findings
- **MIL framing, 2015 [V-snip].**
  - Pinheiro & Collobert (CVPR 2015) frame segmentation as MIL ("every training image is known to have (or not) at least one pixel" of the class). They replace global max pooling with Log-Sum-Exp pooling. — [CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pinheiro_From_Image-Level_to_2015_CVPR_paper.pdf)
  - P1's "≥1 pixel present" cell rule is exactly this MIL assumption, applied per cell instead of per image [I].
- **Global max pooling for weak localisation [V-snip].** Oquab et al. (CVPR 2015) use global max pooling, which "searches for the best-scoring candidate object position". — [CVF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)
- **WILDCAT [V-snip].** Durand et al. (CVPR 2017) introduce pooling that aligns regions and learns localised features for classification, pointwise localisation and segmentation. — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.html)
- **Histopathology, patch-level labels → tissue segmentation [V-snip].**
  - Han et al., "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels" (Medical Image Analysis 80, 2022).
  - Method: CAM pseudo-masks from patch labels, then multi-layer pseudo-supervision with progressive dropout attention and a classification gate that reduces false positives.
  - Sources: [arXiv 2110.08048](https://arxiv.org/abs/2110.08048); [code](https://github.com/ChuHan89/WSSS-Tissue)
  - WSSS4LUAD challenge (10k+ lung adenocarcinoma patches, image-level labels). — [arXiv 2204.06455](https://arxiv.org/pdf/2204.06455)
- **Max-pooling CNNs lose fine spatial information [V-snip].** — [HistoSegCap, arXiv 2402.10851](https://arxiv.org/html/2402.10851v1)
- **Remote sensing, tile-level labels [V-snip].** Wang et al. (Remote Sensing 12(2):207, 2020) use 50×50-px tiles labelled ">50% / <50%" of the target class, because such labels are "quick for humans to assess". — [MDPI](https://www.mdpi.com/2072-4292/12/2/207) (403 when fetched; snippet only)
- **Coarse-resolution labels → fine predictions (remote sensing) [V-abs].**
  - "Handling Image and Label Resolution Mismatch in Remote Sensing" trains on low-resolution labels without upsampling them. It uses "region aggregation, adversarial learning, and self-supervised pretraining", plus a small exemplar set of high-resolution labels. — [arXiv 2211.15790](https://arxiv.org/abs/2211.15790)
  - CS-SUNet (IJCAI 2022) learns 30 m predictions from 3 km labels (100×) and says smoothness regularisation is "crucial for preventing overfitting". — [arXiv 2207.08022](https://arxiv.org/abs/2207.08022)
- **Full supervision upper bound.** LR-ASPP-MobileNetV3-Small reaches 68.38 mIoU on Cityscapes val with pixel masks [V-full]. — [arXiv 1905.02244](https://arxiv.org/pdf/1905.02244)

### Inferences
- **[I] P1 is easier than classic image-level weak segmentation.** Cityscapes has 256 cells per image, each a bag of about 64×32 or 128×64 px at full resolution. That is far stronger supervision than one label per image. Closer analogues are patch-level histopathology and tile-level remote sensing, where the approach is known to work reasonably well.
- **[I] Expected failure modes:**
  1. **Max / noisy-OR pooling gives each positive cell only one or a few "responsible" locations.** Expect peaky maps that under-segment objects spanning several fine-grid positions. This is the known CAM/MIL partial-coverage problem.
  2. **Noisy-OR over many locations saturates.** With about 50–100 fine positions per cell, small per-location probabilities compound towards 1. The model learns to suppress everything except one location, and calibration on new layouts with different cell areas then breaks: a bigger cell has more terms, so a higher OR.
  3. **Layout shift equals bag-size shift.** Re-aggregating to finer L2 cells asks the map to localise *within* L1 cells, which the training loss never constrained.
     - Coarser or union layouts are exact (OR of children).
     - Finer or shifted layouts are where it fails.
     - Expect precision to drop sharply as L2 cells shrink below L1 cells.
  4. **Very small cells at 224 input** (upper Cityscapes cells about 7×3.5 px) are below one stride-8 location, so a 28×28 fine grid can't resolve them.
- **[I] Baselines a reviewer will demand:**
  - (a) **Overlap transfer.** An L2 cell is positive if any overlapping L1 cell is predicted positive. This needs no dense map: it is recall-optimal and precision-poor, but trivial. P1 must beat it on precision at equal recall.
  - (b) **Average-pooling MIL**, which is E1. P1's max/noisy-OR choice should be ablated against mean and LSE.
  - (c) **A fully supervised segmentation net + aggregation** as the upper bound on Cityscapes. It will win.
  - (d) Train directly on L2 labels (the oracle).
- **[I] Can it be evaluated where it matters?**
  - On the outdoor set, without masks, the only exact quantitative test is L2 = unions of the original cells, which is trivially derivable and weak.
  - Finer or shifted outdoor layouts can only be shown qualitatively, or with a small hand-labelled L2 test subset, which brings back some annotation.
  - Honest framing: "simulate on Cityscapes, demonstrate on outdoor".
- **[I] Verdict sketch.** The method is not novel (MIL plus patch-level weak supervision). The novel angle is the *application*: layout transfer for cell detectors, with bag-size and calibration analysis. It overlaps heavily with A1 and E1. Together they make one paper: "dense head + cell aggregation; supervision = pixel masks (upper bound) vs cell labels (P1); test on unseen layouts (A1)".

### Gaps
- I did not open the WILDCAT, Pinheiro or Oquab full texts to quote their reported failure analyses. The partial-coverage claim rests on widely known CAM literature, not on a verified quote.
- I did not verify the Cityscapes "coarse" annotation set (about 20k coarsely labelled images [G]), which would be a natural extra weak-label source.
- I found no prior work that trains a cell-level detector and then *re-aggregates to a different layout*. Absence after limited searching.

---

## Q4. P3: prior art on learned or adaptive spatial partitioning, and the baseline a reviewer would demand

### Takeaway
Prior art on learning *where to pool* exists:
- learned pooling regions (Jia, Huang & Darrell, CVPR 2012);
- deformable RoI pooling [G];
- saliency-based sampling (Learning to Zoom, ECCV 2018);
- non-uniform or adaptive occupancy grids in robotics.

No work optimises a *YOLIC-style labelled cell layout* under a budget, but the field is adjacent and the baselines are obvious: a uniform grid at equal N and a perspective (equal-ground-area) grid.

The fatal issue is evaluation. Changing the layout changes the task and the metric, so "designed layout beats hand layout in F1 at equal N" can be gamed by putting cells where labels are easy (sky, road), unless the metric is defined independently of the layout.

### Cited Findings
- **Learning pooling regions [V-snip].** Jia, Huang & Darrell (CVPR 2012) go "beyond the window sampling of fixed spatial pyramids" to densely sampled windows over location, size and aspect ratio, and select a concise set. — [ICSI page](https://www.icsi.berkeley.edu/icsi/node/4872)
- **Saliency-based sampling layer [V-snip].** Recasens et al. (ECCV 2018) learn input resampling end to end. — [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.html)
- **Non-uniform occupancy grids [V-snip].** Non-uniform cell sizes reduce the number of cells and the compute "without compromising on the quality of the results". — [IEEE 9304571](https://ieeexplore.ieee.org/document/9304571/)
- **Adaptive-resolution maps (OctoMap variants) [V-snip].** — [A-OctoMap, arXiv 2406.13910](https://arxiv.org/html/2406.13910v2)
- **Deformable RoI pooling [G], not opened.** — [arXiv 1703.06211](https://arxiv.org/abs/1703.06211)
- **[G]** YOLOv2's k-means "dimension clusters" for anchor design is the closest "data-driven layout of prediction slots" analogue in detection. Not opened.

### Inferences
- **[I] The metric problem.**
  - Per-cell F1 depends on the cell definition. Small cells over always-empty regions add easy true negatives. Cells aligned with class boundaries change positive rates.
  - A fair comparison needs a **layout-independent target**. Options:
    - pixel-level hazard recall within the region of interest, measured by back-projecting cell predictions to pixels and scoring against masks;
    - or a fixed downstream decision (e.g. "is any person within X m in lane") evaluated identically for every layout.
  - Without one, a reviewer will reject the comparison.
- **[I] Strong baselines at equal N:**
  - (1) uniform grid over the same ROI;
  - (2) perspective-scaled grid with equal ground area per cell (needs only pitch/height, e.g. Cityscapes `camera.json`);
  - (3) YOLIC's hand layout;
  - (4) a random layout.

  I expect (2) to match most of what a greedy search finds. Greedy purity-driven search mostly rediscovers "smaller cells near the horizon, bigger cells near the car", which is exactly the perspective scaling YOLIC's hand Cityscapes layout already approximates (64×32 upper, 128×64 lower; brief §2.2). This is a guess, not evidence.
- **[I] Failure modes:**
  - (a) Purity objectives favour cells that follow typical class boundaries (road/sidewalk edges), which overfit the training camera's framing.
  - (b) Proxy-model F1 in the search loop is expensive (one training run per candidate) and noisy (brief §3.2: single-seed differences of 0.5–3 F1).
  - (c) Interaction with N3: past a cell size of about 1 feature stride, accuracy collapses whatever the layout, so the optimiser should be constrained by the resolution limit.
- **[I] Verdict sketch.** It has a weak novelty claim and serious evaluation design problems. It is best kept as an analysis inside N3 (granularity scaling), not as a headline contribution.

### Gaps
- I found no paper that searches over labelled cell layouts for a cell-wise classifier, with limited searching. NAS-over-layouts and "learnable anchors" searches returned nothing specific.
- Deformable RoI pooling and the YOLOv2 anchor clustering were not opened.

---

## Q5. Anything from 2024–2026 that already does layout-conditioned or region-query perception on edge devices?

### Takeaway
I found no 2024–2026 paper doing *layout-conditioned cell classification on edge devices*. The nearest recent work is:
- region-query feature pooling with heavy foundation models (TextRegion 2025; Region-Based Representations Revisited 2024);
- lightweight monocular column-wise free-space perception (StixelNExT 2024; StixelNExT++ 2025).

The gap appears real but narrow. Any reviewer will map A1 and E1 onto mask pooling.

### Cited Findings
- **TextRegion (2025) [V-snip].** SAM2 soft masks + mask-based attention pooling on frozen image-text features. Heavy and not edge-oriented. — [arXiv 2505.23769](https://arxiv.org/pdf/2505.23769)
- **Region-Based Representations Revisited (2024) [V-abs].** SAM + DINOv2 region features; linear decoders suffice; supports "custom queries". — [arXiv 2402.02352](https://arxiv.org/abs/2402.02352)
- **StixelNExT (2024) [V-abs].** Monocular, low-weight, LiDAR-supervised multi-layer stixels. — [arXiv 2407.08277](https://arxiv.org/abs/2407.08277)
  - Successor StixelNExT++ (2025) [V-snip, title only]: "Lightweight Monocular Scene Segmentation and Representation for Collective Perception". — [arXiv 2507.06687](https://arxiv.org/pdf/2507.06687)
- **WOW-Seg (2026) [V-snip].** Mask-region classification via "region-to-token conversion with mask-based feature aggregation". — [arXiv 2605.16903](https://arxiv.org/pdf/2605.16903)

### Inferences
- **[I] The defensible niche** is the combination: a sub-1-GFLOP edge model + region queries defined by the deployer + supervision from cell labels only + honest evaluation on held-out layouts.
- **[I] The single most important experiment for all of A1, E1 and P1** is the head-to-head against "light segmentation net + per-cell aggregation" at equal FLOPs and input size on Cityscapes, on both seen and unseen layouts. If segmentation wins everywhere, A1 and E1 are dominated. P1 then survives only for the no-masks scenario.

### Gaps
- My searches for 2024–2026 edge "region query" or "layout-conditioned" perception were few (about 3 queries). A dedicated pass might find recent embedded-robotics papers I missed.
- I did not open the WOW-Seg or StixelNExT++ papers beyond search titles and snippets.
