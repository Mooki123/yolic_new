# Prior art and reviewer baselines for A3 (temporal YOLIC), E2 (layout-derived foveated warp), E3 (uncertainty-gated cascade + frame skipping)

Tag convention: **[VERIFIED]** = I opened the page (abstract, full text, directory listing or README) in this session and the claim comes from it. **[SEARCH-SNIPPET]** = the claim comes only from a search-engine summary; I did not open the page. **[GUESSED]** = my own inference or recollection, with no source opened. Date of research: 2026-10-01.

---

## A3 — Prior art for temporal / recurrent heads, and whether EMA / k-of-n / hysteresis would match a learned GRU

### Takeaway
Recurrent and propagation-based temporal fusion for per-frame segmentation is a mature area, with CamVid and Cityscapes as the standard benchmarks: GRFP (gated recurrent flow propagation), temporal-consistency distillation (ECCV 2020), feature-propagation methods (WACV 2024, AAAI 2024) and similarity-weighted interpolation (CVPR 2025). The only direct EMA-vs-learned comparison I found (Linardos et al., BMVC 2019, video saliency) reports that **EMA and ConvLSTM were comparable**. This is weak evidence that the trivial baseline is hard to beat, but it is the only head-to-head I found. A YOLIC GRU paper that does not beat a tuned EMA / k-of-n / hysteresis filter on per-cell sigmoids would be rejected. YOLIC's GAP+FC design also lacks a spatial feature map to warp, so most flow-based machinery does not transfer directly.

### Cited Findings
- **[VERIFIED]** Linardos et al., "Temporal Recurrences for Video Saliency Prediction" (BMVC 2019). They added two recurrence types to a static saliency network, ConvLSTM and an EMA over convolutional features. Both reached state-of-the-art results and produced comparable saliency maps, i.e. the simpler EMA did as well as ConvLSTM. — [arXiv 1907.01869](https://arxiv.org/abs/1907.01869). Note: the EMA here is applied *inside the network to features and trained*, not as post-hoc output smoothing, and the task is saliency, not segmentation or detection.
- **[VERIFIED]** Nilsson & Sminchisescu, "Semantic Video Segmentation by Gated Recurrent Flow Propagation" (GRFP). A spatio-temporal recurrent unit propagates labelling information with optical flow and uses unlabelled frames to improve accuracy on Cityscapes and CamVid. — [arXiv 1612.08871](https://arxiv.org/abs/1612.08871). (Venue: the arXiv page I read did not state it; I recall CVPR 2018. **[GUESSED]**.)
- **[VERIFIED]** Vincent, Kim & Meeß, "High Temporal Consistency through Semantic Similarity Propagation…" (CVPR 2025).
  - Method: an efficient image segmenter's prediction is propagated with global registration (camera-motion compensation). It is linearly interpolated with the previous prediction, using weights computed from feature similarity, i.e. a *learned/adaptive EMA*.
  - Results: temporal consistency +12.5% on UAVid and +6.7% on RuralScapes over the per-frame base model, at comparable speed.
  - The abstract does not mention a fixed-α EMA baseline. — [arXiv 2503.15676](https://arxiv.org/abs/2503.15676)
- **[SEARCH-SNIPPET]** Liu, Shen, Yu et al., "Efficient Semantic Video Segmentation with Per-frame Inference" (ECCV 2020).
  - Temporal-consistency knowledge distillation (pairwise-frame and multi-frame) at *training* time only; inference stays per-frame, with zero runtime cost.
  - Improves temporal consistency on Cityscapes and CamVid. — [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550358.pdf); [Springer](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_21)
- **[SEARCH-SNIPPET]** 2024 work on stable, efficient video segmentation:
  - Lin et al., AAAI 2024, "Exploring Temporal Feature Correlation for Efficient and Stable Video Semantic Segmentation", says stability is under-explored in keyframe-based efficient VSS. — [mlanthology](https://mlanthology.org/aaai/2024/lin2024aaai-exploring/)
  - Baghbaderani et al., WACV 2024, temporally consistent VSS with occlusion-guided feature propagation. — [CVF PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Baghbaderani_Temporally-Consistent_Video_Semantic_Segmentation_With_Bidirectional_Occlusion-Guided_Feature_Propagation_WACV_2024_paper.pdf)
  - AuxAdapt: test-time adaptation for temporally consistent VSS. — [arXiv 2110.12369](https://arxiv.org/pdf/2110.12369)
- **[SEARCH-SNIPPET]** Output EMA is standard in older streaming work. Miksik et al. (ICRA 2013), "Efficient Temporal Consistency for Streaming Video Scene Analysis", describes EMA of per-frame outputs as the natural online baseline that compromises between noise suppression and lag. — [CMU PDF](https://www.ri.cmu.edu/pub_files/2013/5/miksik_icra_13.pdf)
- **[SEARCH-SNIPPET]** "CNN + temporal smoothing (averaging / HMM)" of per-frame classifier outputs is an established recipe for video frame classification, e.g. M2CAI surgical workflow. — [arXiv 1610.05541](https://arxiv.org/pdf/1610.05541)
- **[VERIFIED]** The YOLIC paper itself lists using consecutive frames as future work (txt:876-877, per `research/01_brief.md` §4.1). — [local brief](D:/Projects/yolic_new/research/01_brief.md)
- **[SEARCH-SNIPPET]** I found no YOLIC follow-up that adds temporal modelling. The only follow-up surfaced is "YOLIC Labeling" (Su et al., arXiv 2025), a SAM-assisted cell annotation tool. — search results only; no URL opened.

### Inferences
- **[GUESSED]** The reviewer baseline set is EMA of per-cell sigmoids (α tuned on val), k-of-n majority over the last n frames, and two-threshold hysteresis (on at p>τ_on, off at p<τ_off).
  - All three are zero-parameter and cost microseconds on a Pi.
  - A learned GRU must beat the *best tuned* one at matched latency/lag, otherwise the paper is "a GRU that rediscovered an EMA".
  - The Linardos result suggests the gap may be small.
- **[GUESSED]** The plausible advantages of a GRU over EMA:
  - (i) an input-dependent gate, e.g. trust the current frame more when the image changes;
  - (ii) cross-cell/cross-class context;
  - (iii) ego-motion handling.
  - With fixed cells on a moving vehicle, objects slide across cells at a rate that depends on speed. A per-cell EMA lags and smears them, so the ego-motion-warped variant is the part that could matter. That needs odometry or flow, which CamVid video provides only implicitly.
- **[GUESSED]** There is a subtle failure mode: smoothing raises single-frame recall on persistent objects, but adds *latency* to first detection of a suddenly appearing hazard, which matters for safety. Any A3 paper must report detection delay (frames to first correct fire), not just F1 and flicker. This also applies to the EMA baseline, so it is a fair axis of comparison.
- **[GUESSED]** There is a training-signal problem. With labels at 1 Hz (CamVid) and 30 frames between labels, a GRU trained with BPTT over 30 unlabelled steps sees one supervised step per clip. Expect the GRU to learn little more than a constant-α filter unless there is auxiliary supervision, e.g. pseudo-labels from a teacher on every frame (cf. Liu ECCV 2020 distillation).
- **[GUESSED]** Learned-vs-trivial verdict: likely **dominated or tied** by tuned EMA / hysteresis on per-frame F1. A learned GRU has a chance only on detection-delay vs flicker trade-off curves, or with ego-motion compensation. The novelty is thin either way: "recurrent head for a per-frame classifier" is well covered. The cell-specific angle (per-cell gating across a fixed layout) is a modest delta.

### Gaps
- I found **no paper that directly compares post-hoc output EMA vs a learned recurrent head for video semantic segmentation** with numbers. The Linardos comparison is for saliency and feature-level EMA. I could not open the CVPR 2025 SSP PDF (403 from CVF) to check whether it includes a fixed-α EMA ablation.
- Not opened, cited from memory only (**[GUESSED]** URLs/IDs; verify before citing):
  - Accel (Jain et al., CVPR 2019, arXiv 1807.06667?);
  - TDNet (Hu et al., CVPR 2020, arXiv 2004.01800?);
  - ConvLSTM (Shi et al., NeurIPS 2015);
  - NetWarp (Gadde et al., ICCV 2017);
  - Clockwork convnets (Shelhamer et al., ECCVW 2016);
  - DFF (Zhu et al., CVPR 2017).
- I did not find any "video semantic segmentation on edge / Raspberry Pi" paper with temporal fusion measured on a Pi-class CPU.

---

## A3 — Data feasibility (CamVid frame rates/labels/raw video; other small sparse-label video sets; YOLIC outdoor ordering)

### Takeaway
CamVid is feasible:
- 701 labelled frames, mostly at 1 Hz, plus one 101-frame 15 Hz subsequence.
- The full 30 Hz raw video is still downloadable, about 7.4 GB in 4 files: Panasonic P2 MXF plus one Lagarith AVI, so it needs codecs to decode.
- Caveat: 701 labelled frames over 4 sequences is a very small test bed, and CamVid shots come from a car, not a scooter.

The YOLIC outdoor set's filenames look like `YYMMDD_HHMMSS…` timestamps, so frame order is probably recoverable, but it is not documented. KITTI-STEP and VSPW are alternatives, but VSPW is about 43 GB and generic, not driving.

### Cited Findings
- **[VERIFIED]** CamVid official page: "over ten minutes of high quality 30Hz footage … with corresponding semantically labeled images at 1Hz and in part, 15Hz"; 701 labelled frames; 32 semantic classes. — [CamVid page](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  - Per-sequence counts:

    | Sequence | Labelled frames |
    |---|---|
    | seq06R0 | 101 at 1 Hz |
    | seq16E5 | 204 at 1 Hz |
    | seq16E5_15Hz | 101 at 15 Hz |
    | seq05VD | 101 at 1 Hz |
    | seq01TP | 124 at 1 Hz |

- **[VERIFIED]** Raw video inventory (fetched over plain HTTP with curl), total **7.42 GB**:

  | File | Size |
  |---|---|
  | 0016E5.MXF | 2.54 GB |
  | 0006R0.MXF | 1.68 GB |
  | 0005VD.MXF | 1.55 GB |
  | 01TP_extract.avi | 1.65 GB |

  — [inventory.txt](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/inventory.txt)
- **[VERIFIED]** Notes in the same inventory file:
  - Labelled images "were sampled from those big sequences at 1Hz".
  - The codec is "mostly the Panasonic P2 codec … (note, NOT square pixels)".
  - 01TP was re-encoded with the lossless Lagarith (LAGS) codec: 3690 frames plus pre- and post-roll; the original MXF is "over 3GB".
  - Frame extraction uses Windows DirectShow scripts.
  - The 15 Hz CamSeq01 subsequence (101 frames) "was used to test label-propagation". — [inventory.txt](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/inventory.txt)
- **[VERIFIED]** The download directory currently lists `01TP_extract.avi`, `0005VD.MXF`, `0006R0.MXF`, `0016E5.zip.001`, `0016E5.zip.002` and `md5sums`, so the files are still online as of 2026-10-01. HTTPS refused the connection; plain HTTP worked. — [UCL directory](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/)
- **[VERIFIED]** AR-Seg (THU-LYJ-Lab) ships a `camvid_decode.sh` that decodes those MXF/AVI files into per-sequence frame folders. This is a ready recipe for getting the 30 fps frames. — [AR-Seg pre-process README](https://github.com/THU-LYJ-Lab/AR-Seg/blob/main/pre-process/README.md)
- **[VERIFIED]** YOLIC outdoor (Dataset Ninja):
  - 20,380 frames, 13 GB (12.53 GB download), split 14,266/2,038/4,076.
  - Example filenames `200508_13460200004530.png`, `200425_17391000000360.png`.
  - No documented video IDs, frame numbers or sequence ordering. — [Dataset Ninja](https://datasetninja.com/outdoor-hazard-detection)
- **[VERIFIED]** The Kaggle page could not be read (JS-rendered); WebFetch returned only the title. — [Kaggle](https://www.kaggle.com/datasets/sukai3316/outdoor-hazard-detection-dataset)
- **[SEARCH-SNIPPET]** KITTI-STEP:
  - 21 train+val sequences (12 train / 9 val) and 29 test sequences;
  - labels are semi-automatic pseudo-labels refined by humans (the deeplab2 setup doc did not state the frame rate or sizes);
  - search snippets give about 18k frames over 50 videos, ~381 annotated frames per sequence, i.e. dense labelling of every frame.
  - Images must come from the KITTI tracking benchmark. — [deeplab2 KITTI-STEP doc (VERIFIED for split counts)](https://github.com/google-research/deeplab2/blob/main/g3doc/setup/kitti_step.md); [STEP arXiv 2102.11859](https://arxiv.org/abs/2102.11859)
- **[SEARCH-SNIPPET]** VSPW: 3,536 videos and 251,633 frames, 124 classes, dense labels at 15 fps, about 5 s per clip, about 43 GB download. Generic scenes, not driving. — [CVF VSPW](https://openaccess.thecvf.com/content/CVPR2021/html/Miao_VSPW_A_Large-scale_Dataset_for_Video_Scene_Parsing_in_the_CVPR_2021_paper.html)

### Inferences
- **[GUESSED]** The outdoor filename looks like `YYMMDD_HHMMSSxx` followed by a frame counter, e.g. `200508_134602 00004530` = 2020-05-08 13:46:02, frame 4530. If so, sorting by filename recovers order and capture session (date + start time = video ID). The frames were "hand-picked", though, so gaps between kept frames are probably irregular and large. That suits leakage-free splits by session, but probably not a 30 Hz recurrent model. To verify, download the data and look at the filename-number deltas.
- **[GUESSED]** KITTI-STEP is probably the best second dataset for A3: a driving domain, every frame labelled (enables dense temporal supervision and flicker metrics against ground truth), and a modest size (KITTI tracking images are a few GB). Cell labels can be derived from its semantic maps exactly as for Cityscapes.
- **[GUESSED]** With CamVid alone, test sets are about 233 labelled 1 Hz frames (the standard split), so per-frame F1 differences will be noisy. Temporal-consistency metrics on the unlabelled 30 Hz frames (flicker rate of cell decisions) need no labels and are the more convincing evaluation. The 101-frame 15 Hz subsequence allows a small labelled flicker or detection-delay evaluation.
- **[GUESSED]** P2 MXF decoding on a modern system may need ffmpeg; the official route is DirectShow plus Panasonic codecs on Windows. This has not been tested here.

### Gaps
- I did not confirm KITTI-STEP total download size or frame rate (KITTI is 10 Hz from memory, **[GUESSED]**).
- I could not confirm the outdoor filename semantics, or whether the indoor dataset (1 fps sampling per paper) keeps video IDs.
- I did not check whether ffmpeg decodes the P2 MXF files.

---

## E2 — Layout-derived foveated / saliency warp at equal FLOPs: prior art and baselines

### Takeaway
A **fixed, dataset-wide saliency warp** is already published:
- FOVEA's KDE S_D prior (ICCV 2021) for detection.
- LZU's "fixed saliency" (CVPR 2023), which for Cityscapes *segmentation* uses only a fixed, train-set-aggregated saliency and gains +0.3 to +2.4 mIoU over uniform downsampling.

Deriving the saliency from a cell layout instead of from boxes or boundaries is a small delta. The obvious reviewer baselines are:
- (a) crop to the union bounding box of the cells, then resize;
- (b) a non-square input matched to the cell region's aspect ratio at equal FLOPs.

For YOLIC's layouts, which are mostly rectangular bands, (a)+(b) probably capture most of the gain, since FOVEA/LZU gains are biggest when salient regions are small and scattered. A specific YOLIC complication: the GAP+FC head would have to relearn positions under the warp. That is fine for a fixed warp, but it removes any "plug-in" story.

### Cited Findings
- **[VERIFIED]** FOVEA (Thavamani, Li, Cebron, Ramanan, ICCV 2021): a differentiable resampling layer onto a fixed-size canvas that magnifies salient regions. — [arXiv 2108.12102](https://arxiv.org/abs/2108.12102); details from [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2108.12102)
  - Saliency sources:
    - S_D, a *dataset-wide spatial prior* from KDE on all training boxes ("small objects tend to exist near a fixed horizon");
    - S_I, a temporal prior from the previous frame's detections;
    - S_C, a mix of the two.
  - Argoverse-HD results:

    | Setting | AP | AP_S |
    |---|---|---|
    | Baseline at 0.5× (no fine-tuning) | 21.5 | 2.8 |
    | KDE S_D | 23.3 | 5.4 |
    | KDE S_I | 24.1 | 8.5 |
    | Baseline at 0.5×, fine-tuned | 24.2 | 4.9 |
    | Learned LKDE S_I | 28.1 | 10.3 |
    | Upper bound: uniform 0.75× | 29.2 | 11.6 |

  - Streaming AP rises from 17.8 to 23.0.
- **[VERIFIED]** LZU, "Learning to Zoom and Unzoom" (Thavamani, Li, Ferroni, Ramanan, CVPR 2023): a piecewise-bilinear invertible warp, so features can be unwarped. Applied to detection, Cityscapes semantic segmentation and nuScenes mono-3D. — [arXiv 2303.15390](https://arxiv.org/pdf/2303.15390); [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Thavamani_Learning_To_Zoom_and_Unzoom_CVPR_2023_paper.html); details from [ar5iv](https://ar5iv.labs.arxiv.org/html/2303.15390)
  - For Cityscapes segmentation the saliency is **fixed only**: ground-truth semantic boundaries aggregated over the train set, average-pooled to 45×45.
  - Cityscapes mIoU, uniform vs LZU-fixed:

    | Input | Uniform | LZU-fixed |
    |---|---|---|
    | 64×64 | 26.4 | 26.7 |
    | 128×128 | 39.3 | 41.7 |
    | 256×256 | 53.6 | 55.1 |
    | 512×512 | 63.8 | 64.2 |

  - Reported latency is 15.5 / 16.1 / 19.1 / 32.3 ms.
  - No comparison to cropping (per my reading of the extracted text).
  - Detection uses a fixed variant (KDE on all training boxes) and an adaptive one (KDE on previous-frame detections).
- **[SEARCH-SNIPPET]** Recasens et al., "Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks" (ECCV 2018) is the origin of the differentiable saliency sampler. — [CVF](https://openaccess.thecvf.com/content_ECCV_2018/html/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.html); [arXiv 1809.03355](https://arxiv.org/pdf/1809.03355); [code](https://github.com/recasens/Saliency-Sampler)
- **[SEARCH-SNIPPET]** Marin et al., "Efficient Segmentation: Learning Downsampling Near Semantic Boundaries" (ICCV 2019) uses content-adaptive downsampling that favours boundary locations, motivated by small objects and boundaries lost to uniform downsampling. — [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Marin_Efficient_Segmentation_Learning_Downsampling_Near_Semantic_Boundaries_ICCV_2019_paper.html); [arXiv 1907.07156](https://arxiv.org/pdf/1907.07156)
- **[SEARCH-SNIPPET]** Jin et al., "Learning to Downsample for Segmentation of Ultra-High Resolution Images" (ICLR 2022) is a learnable downsampler trained end-to-end with the segmenter. — [arXiv 2109.11071](https://arxiv.org/abs/2109.11071)
- **[SEARCH-SNIPPET]** ZoomTrack (2023) applies target-aware non-uniform resizing to tracking, so the idea is spreading to other tasks. — [arXiv 2310.10071](https://arxiv.org/pdf/2310.10071)
- **[VERIFIED, local]** YOLIC squashes 2048×1024 Cityscapes and 848×480 outdoor frames to 224×224. Cityscapes upper cells become about 7×3.5 px. The Cityscapes cells cover only x∈[512,1536], y∈[320,1024]. — [local brief §1.3, §2.2](D:/Projects/yolic_new/research/01_brief.md)

### Inferences
- **[GUESSED]** **Is it already done?** Largely yes, conceptually. "Fixed, dataset-derived saliency → non-uniform resample → same-FLOPs network" is FOVEA S_D and LZU-fixed. The YOLIC-specific delta has three parts:
  - (i) the saliency comes from the cell layout itself, with no labels needed;
  - (ii) no unwarp is needed, because outputs are per-cell and warp-invariant by construction;
  - (iii) the target is a Pi-class CPU at 224 px.
  - (ii) is a genuine simplification over LZU, but a small contribution on its own.
- **[GUESSED]** **Simplest baselines a reviewer will demand:**
  1. **Crop-to-cell-union + resize.** For Cityscapes, crop [512:1536]×[320:1024] (1024×704) and resize to 224×224. This alone raises linear resolution about 2× horizontally and about 1.45× vertically for every cell, at zero cost and without any warp.
  2. **Aspect-matched input at equal FLOPs**, e.g. 272×184 ≈ 50k px ≈ 224² for the crop's ~1.45:1 aspect. This removes the anisotropic squash.
  3. **Piecewise "two-band" crop:** far band at higher scale and near band at lower scale, concatenated, which is a 1-D warp.

  On Cityscapes most of the E2 gain probably comes from (1)+(2). A smooth 2-D foveation adds a smaller extra (LZU-fixed's gains over uniform are only +0.3 to +2.4 mIoU even without cropping).
- **[GUESSED]** **Failure modes:**
  - Warping breaks ImageNet-pretrained statistics (scale and aspect), which can cost accuracy at small backbones.
  - Bilinear resampling at high magnification gives blurry far cells. Magnifying a 64×32 px source cell does not add information beyond the original frame's resolution, though the source here is 2048 px, so there is headroom.
  - Shrinking near cells hurts large or close objects, which are the most safety-critical. FOVEA claims no loss on large objects, but YOLIC's near cells are the ones that matter.
  - The warp is tied to one camera, inheriting YOLIC's fixed-geometry assumption.
  - The resample costs CPU time on a Pi. It is cheap as a precomputed `remap`, but must be measured.
- **[GUESSED]** For the outdoor set (848×480, 104 cells), it is unknown whether the cells cover nearly the whole frame. If they do, crop gains vanish and only foveation remains.

### Gaps
- I did not find any paper that compares a fixed saliency warp against a simple ROI crop + aspect-matched resize at equal FLOPs. LZU and FOVEA compare against uniform downsampling of the full frame (FOVEA also against a higher-resolution upper bound).
- I did not verify FOVEA's BDD100K numbers or its precise fixed-prior fine-tuned numbers.

---

## E3 — Uncertainty-gated two-stage cascade + frame skipping: prior art, strongest baseline, worst-case latency

### Takeaway
Confidence-gated cascades are a textbook technique:
- IDK cascades (UAI 2018);
- NoScope difference detectors + specialised cascades (VLDB 2017);
- Skip-Convolutions (CVPR 2021);
- the theory paper "When does confidence-based deferral suffice?" (NeurIPS 2023), which finds plain max-confidence deferral is usually hard to beat.

Nothing found is specific to cell-wise outputs. The only cell-specific twist, aggregating uncertainty over safety-relevant cells, is a minor design choice.

For a real-time safety system, cascades and frame-skipping improve **average**, not **worst-case**, latency. The worst-case frame still pays stage 1 + stage 2. This point appears in the early-exit / real-time literature. A reviewer will ask why one wouldn't simply deploy the single model whose worst-case latency fits the frame deadline.

### Cited Findings
- **[SEARCH-SNIPPET]** IDK Cascades (Wang, Luo, Crankshaw, Tumanov, Yu, Gonzalez, UAI 2018) compose pre-trained models of increasing cost with an "I don't know" class. When an upstream model is confident, downstream models are skipped, cutting average inference time without accuracy loss. — [UAI PDF](https://www.auai.org/uai2018/proceedings/papers/212.pdf); [mlanthology](https://mlanthology.org/uai/2018/wang2018uai-idk/)
- **[VERIFIED]** Jitkrittum et al., "When Does Confidence-Based Cascade Deferral Suffice?" (NeurIPS 2023). Simple confidence-based deferral (max softmax) "often works well". Learned post-hoc deferral helps mainly when downstream models are specialists, labels are noisy, or there is train/test distribution shift. — [arXiv 2307.02764](https://arxiv.org/abs/2307.02764)
- **[SEARCH-SNIPPET]** "Revisiting Cascaded Ensembles for Efficient Inference" (2024) is recent evidence that cascades remain an active baseline area. — [arXiv 2407.02348](https://arxiv.org/html/2407.02348v1)
- **[SEARCH-SNIPPET]** "Learning to Cascade: Confidence Calibration for Improving the Accuracy and Computational Cost of Cascade Inference Systems" (AAAI 2021). — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16900/16707)
- **[SEARCH-SNIPPET]** NoScope (Kang et al., VLDB 2017) uses a difference detector to drop frames that barely change, then a specialised small model, then the reference model if uncertain. Reported up to 1000× speed-up on fixed-camera video queries. This is exactly E3's "frame skip + uncertainty-gated cascade", for fixed cameras. — [VLDB PDF](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf); [ar5iv](https://ar5iv.labs.arxiv.org/html/1703.02529)
- **[SEARCH-SNIPPET]** Habibian et al., Skip-Convolutions (CVPR 2021) gate residuals between frames per layer and region, cutting cost 3–4× on EfficientDet and HRNet without accuracy drop. — [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Habibian_Skip-Convolutions_for_Efficient_Video_Processing_CVPR_2021_paper.html)
- **[SEARCH-SNIPPET]** Other related work:
  - AdaFocus V2/V3: spatial and temporal dynamic video recognition. — [arXiv 2209.13465](https://arxiv.org/pdf/2209.13465)
  - FrameHopper: selective frame processing for real-time detection-driven video analytics. — [ResearchGate](https://www.researchgate.net/publication/359411518_FrameHopper_Selective_Processing_of_Video_Frames_in_Detection-driven_Real-Time_Video_Analytics)
  - "Scaling Video Analytics on Constrained Edge Nodes". — [arXiv 1905.13536](https://arxiv.org/pdf/1905.13536)
  - "Cascading IDK Classifiers to Accelerate Object Detection" (COMPSAC 2025). — [PDF](https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/compsac25.pdf)
  - "Optimal Synthesis of Robust IDK Classifier Cascades" (ACM). — [ACM](https://dl.acm.org/doi/10.1145/3609129)
- **[VERIFIED]** Worst-case latency is recognised as a problem. Daghero, Moghaddam & Kjærgaard, "Hierarchical adaptive control for real-time dynamic inference at the edge" (arXiv 2604.26470, 2026), frame dynamic models as an "accuracy versus average latency tradeoff". Their first contribution is "a budgeted SP-cascade formulation that preserves worst-case latency constraints", which implies plain cascades do not. — [arXiv 2604.26470](https://arxiv.org/abs/2604.26470)
- **[SEARCH-SNIPPET]** Early exit "reduces average latency when prediction confidence is high, [but] the execution time becomes unpredictable and depends on input data", and tail latency stays high for inputs that don't exit early. Thermal throttling and class drift can make dynamic inference violate latency constraints. — search summary of [arXiv 2604.26470](https://arxiv.org/html/2604.26470) and [HELIOS, MLSys 2026](https://lca.ece.utexas.edu/pubs/kumar_mlsys26.pdf). I did not open HELIOS, so treat its attribution as unverified.
- **[SEARCH-SNIPPET]** Safety-critical early-exit work emphasises per-exit uncertainty estimation, e.g. "Early-Exit Neural Networks with Nested Prediction Sets". — [arXiv 2311.05931](https://arxiv.org/pdf/2311.05931)
- **[SEARCH-SNIPPET]** Predictable real-time perception pipelines for AVs, e.g. "Prophet" (Liu et al. 2022). — [PDF](https://weisongshi.org/papers/liu22-prophet.pdf)

### Inferences
- **[GUESSED]** **Already done?** Yes, for the generic mechanism. Small-model → uncertainty gate → big model, plus a frame-difference skip, is IDK cascades + NoScope applied to cells. A cell-specific novelty claim would have to come from something like "gate on safety-cell uncertainty, re-run stage 2 only on a crop of uncertain cells". Even that resembles AdaFocus or Skip-Conv spatial gating.
- **[GUESSED]** **Strongest baselines** (all needed):
  - (a) Stage-1 model alone with per-class thresholds tuned on val (cf. P2). This is often most of the accuracy for free.
  - (b) A single mid-size model whose *fixed* latency equals the cascade's *average* latency. For example, MobileNetV2 at 160 px or ShuffleNetV2 ×1.0 at 224 px, a different point on the same Pareto front.
  - (c) Stage-2 at a lower frame rate (run the big model every k frames, carry state with EMA). This is the trivial temporal-amortisation baseline.
  - (d) Max-sigmoid-margin deferral vs any fancier uncertainty (per Jitkrittum 2023, expect max-confidence to be near-best).
  - Plausibly (b) or (c) matches the cascade at equal average compute, because single-model scaling curves at this size are smooth.
- **[GUESSED]** **Worst-case argument:**
  - A scooter hazard warning must meet a per-frame deadline. With a cascade, the worst-case frame costs t1 + t2 > t2. So the deadline must accommodate a *larger* worst case than just running stage 2, unless the system drops frames when overloaded.
  - Frame skipping makes it worse in a different way: it adds detection delay for hazards that appear in skipped frames, and difference detectors are tuned on static fixed cameras (NoScope's setting), not on a moving ego-camera, where nearly every frame changes.
  - On a moving scooter, skip rates are likely low, which shrinks the claimed savings.
  - An honest E3 paper reports p99 latency, the deadline-miss rate and detection delay, not just mean FLOPs.
- **[GUESSED]** **Data:** the outdoor, indoor and Cityscapes sets suffice for the cascade part (single images). Frame skipping needs ordered video (CamVid 30 Hz, KITTI-STEP), where *labels* exist only sparsely (CamVid) or densely (KITTI-STEP). Without a Pi, laptop-CPU latency is only a proxy, and gating overhead (two model loads in Pi RAM, cache effects) can't be measured.
- **[GUESSED]** **Overall verdict:** E3 is low-novelty and likely matched by baseline (b) or (c). It is best repositioned as an *analysis* ("how often is the hard path needed per dataset and per cell region"), or folded into P2 / A3 as an ablation.

### Gaps
- Not opened, cited from memory (**[GUESSED]** IDs; verify before citing):
  - BranchyNet (Teerapittayanon et al., ICPR 2016, arXiv 1709.01686?);
  - MSDNet (Huang et al., ICLR 2018, arXiv 1703.09844?);
  - Big-Little Net (Chen et al., ICLR 2019, arXiv 1807.03848?);
  - Big/Little DNN for ultra-low-power inference (Park et al., CODES+ISSS 2015);
  - Chameleon (Jiang et al., SIGCOMM 2018);
  - FFS-VA (Zhang et al., 2018/2019).
- I found no paper that explicitly argues "cascades are unsuitable for hard-real-time safety perception because of worst-case latency" as its main thesis. The point appears only as motivation in deadline-aware adaptive-inference papers (2604.26470, HELIOS).
- I found no prior work on cascades for cell-wise / grid-classification detectors specifically.
