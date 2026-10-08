# 01 — Understanding brief: YOLIC (Su et al., Image and Vision Computing 147, 2024, 105095)

Scope: understanding only. This brief proposes no research ideas.

Sources read: `yolic.pdf` / `yolic_paper_text.txt` (full), and all 10 scripts in this repo (every line).
Upstream: https://github.com/kai3316/YOLIC_code holds the same 10 files.
Also checked: the project page https://kai3316.github.io/yolic.github.io/ and the Dataset Ninja mirror of the outdoor dataset.

**Tags.** **[V]** = VERIFIED, read directly in the paper, the code, or a cited web page. **[I]** = INFERRED, my reasoning or arithmetic from verified facts, not stated anywhere.
Paper references use line numbers in `yolic_paper_text.txt` (`txt:L`). Code references use `file:line`.

---

## 1. Core claim, mechanism, assumptions

### 1.1 Core claim
- **[V]** Claim: YOLIC ("You Only Look at Interested Cells") localizes and classifies objects on edge devices. It reaches detection performance "comparable to" YOLOv5/v6/v8 while being faster, exceeding 30 FPS on a Raspberry Pi 4B CPU (txt:26-28). Headline speed: 40.06 FPS for INT8 ShuffleNetV2 under ncnn (txt:124-126, Table 9).
- **[V]** The paper claims it is "the first method with 224×224 inputs to surpass 30 FPS detection on a Raspberry CPU" with accuracy comparable to YOLO (txt:135-138).
- **[V]** The paper itself says YOLIC is *not* instance-level detection. It "localizes and categorizes object parts within predefined cells" (txt:115-122, 184-190).

### 1.2 Mechanism (what the method actually is)
- **[V]** **Fixed cell layout.** A designer fixes, in pixel coordinates, a set of N "Cells of Interest" (CoIs) on the camera frame. They can be rectangles (outdoor, Cityscapes) or polygons (indoor) (txt:340-350; `outdoor_pred.py:50-116`, `indoor_pred.py:47-74`, `cityscapes_yolic.py:33-106`).
- **[V]** **Labels are per-cell multi-hot vectors.** Each cell gets M object bits plus 1 "background/road" bit, so an image's label is N×(M+1) binary values with no coordinates (txt:351-356, Eq. 1).
  - Outdoor: 104×12 = 1248.
  - Indoor: 30×7 = 210.
  - Cityscapes: 256×4 = 1024 (txt:544, 657, 750-752; `outdoor_yolic.py:47-51`, `indoor_yolic.py:36-40`, `cityscapes_yolic.py:31-32,112`).
- **[V]** **The network is an ImageNet classifier with a wider final layer.** The backbone (MobileNetV2 or ShuffleNetV2 x1.0) feeds global average pooling, then Dropout, then one Linear layer with N×(M+1) outputs (txt:307-327, Table 1). In code, torchvision `mobilenet_v2` has `classifier[1]` replaced (`outdoor_yolic.py:50-51`).
- **[V]** **Loss and inference.** Training uses sigmoid plus binary cross-entropy averaged over all outputs (Eq. 2; `nn.BCEWithLogitsLoss`, `outdoor_yolic.py:155`). At inference each output is thresholded at 0.5. There is no NMS and no box regression (txt:400-421).
- **[I]** **Where localization comes from.** All spatial information passes through a *globally average-pooled* 1280-d (MobileNetV2) or 1024-d (ShuffleNetV2) vector. The FC layer must therefore decode "which object is in which cell" from a position-free pooled descriptor. That only works if the backbone encodes absolute position implicitly, for example through zero-padding and border effects, and if scenes have a stable layout. The paper never discusses this.
- **[V]** **Training recipe.** Adam with lr 1e-3, MultiStepLR at epochs 100 and 125 (γ = 0.1), 150 epochs, batch 32, 224×224 input, ImageNet-pretrained backbone, flip and color-jitter augmentation. Quantization-aware training (QAT) produces the "Q" models (txt:422-473, 493-500). The code matches this except where noted in §2.

### 1.3 Assumptions the method rests on
1. **[I]** **Fixed camera geometry.** Camera pose, intrinsics and resolution stay constant between labelling and deployment, so a fixed pixel region always maps to the same physical region, e.g. "0–6 m ahead" (txt:529-536, 627-635). Cells are hard-coded pixel boxes (`outdoor_pred.py:50-68`). The distance meaning of a cell relies on a flat ground plane and a rigid mount.
2. **[I]** **A layout prior exists and suffices.** Objects of interest appear in predictable image regions, and the application needs only "is class k present in region i", not instances, counts, or exact extents (txt:100-104, 115-122).
3. **[I]** **Global pooling keeps enough position information.** A GAP + FC head can recover per-cell presence for every cell at once. This becomes harder as cells shrink: Cityscapes upper cells are 64×32 px of a 2048×1024 frame, roughly 7×3.5 px after resizing to 224×224 (`cityscapes_yolic.py:33`, resize at :124).
4. **[I]** **Low resolution is enough.** A 224×224 input with *non-uniform* aspect squashing is sufficient: 848×480 and 2048×1024 frames are both resized to 224×224 (`transforms.Resize((224,224))`, `outdoor_yolic.py:122`, `cityscapes_yolic.py:124`). The paper justifies this only by saying the cells are predefined (txt:313-317).
5. **[I]** **Cell outputs need no structure.** Treating the N×(M+1) outputs as independent sigmoids with plain BCE is adequate. There is no class-imbalance handling, no spatial or temporal consistency, and the decision threshold stays at 0.5.

---

## 2. Code vs paper, and hard-coded settings

### 2.1 Where the code differs from the paper (or is buggy)

| # | Finding | Evidence | Tag |
|---|---|---|---|
| D1 | **The flip augmentation flips the image vertically, not horizontally.** `image.flip(1)` runs *after* `ToTensor()`, so the tensor is C×H×W and dim 1 is height. The label permutation, however, mirrors cells **left↔right**. Image and labels therefore disagree on roughly 50% of outdoor and indoor training samples. The paper says "random horizontal flipping" (txt:499). | `outdoor_yolic.py:64,108-115` (transform order :120-125); `indoor_yolic.py:53,98-102`; `indoor_eval.py:57`, `outdoor_eval.py:72` | **[V]** that the code does `flip(1)` on a CHW tensor (standard PyTorch semantics; torch is not installed here, so I did not run it). **[I]** that it is a bug and that it adds label noise / a vertical-flip "task" |
| D2 | **Outdoor training applies the random flip to validation and test too.** The outdoor dataset class has no `train` flag, unlike indoor. Checkpoint selection (`save_mode=True` on val) therefore uses a randomly augmented, label-misaligned validation set. The test accuracy logged each epoch is also augmented. | `outdoor_yolic.py:88-117,137-139,251` vs `indoor_yolic.py:98,125-126` | **[V]** |
| D3 | **The outdoor eval script computes binary (Risk/Road) metrics on cells 64–95 only** (`range(768, 1152, 12)`), i.e. the bottom two road rows (y = 374–480 px). The full-range loop is commented out, and so is the per-class report (Table 2). Table 3 may therefore cover only 32 of 104 cells. | `outdoor_eval.py:156-157,231-236` | **[V]** code; **[I]** link to Table 3 (see §6) |
| D4 | **The paper's "background wins" inference rule is not implemented.** The paper says that when both the background and an object probability exceed 0.5, the result relies on the background bit (txt:411-414). The eval scripts score every bit independently. The prediction/visualisation scripts draw the object whenever any object bit is set, regardless of background. | `outdoor_eval.py:158-181`; `outdoor_pred.py:182-213` | **[V]** |
| D5 | **The indoor visualisation script plots ground truth, not predictions.** `each = orig[...]` is active and `each = pred[...]` is commented out. If Fig. 6 ("output from four distinct YOLIC models") came from this script, it shows labels. | `indoor_pred.py:136-138` | **[V]** code; **[I]** Fig. 6 link |
| D6 | **Colour order differs between training and qualitative prediction.** Training loads images as **RGB** through PIL. The outdoor and indoor prediction scripts load **BGR** through `cv2.imread` and feed it with no conversion. The Cityscapes prediction script converts only the frame it draws on. | `outdoor_yolic.py:101` vs `outdoor_pred.py:158,225-228`; `indoor_pred.py:102,178-181` | **[V]** |
| D7 | **No ImageNet mean/std normalisation**, although the backbones are ImageNet-pretrained (`ToTensor()` only). The paper says nothing about normalisation. | `outdoor_yolic.py:120-129`, `indoor_yolic.py:107-116`, `cityscapes_yolic.py:122-131` | **[V]** |
| D8 | **Colour jitter uses `hue=0.5`, the maximum possible.** Hue is rotated arbitrarily, which matters for colour-defined classes such as traffic cone, weed and zebra crossing. The paper only says "color jittering". | `outdoor_yolic.py:123` (same in indoor and Cityscapes) | **[V]** value; **[I]** impact |
| D9 | **Cityscapes training uses an unmentioned random rescale-and-crop** (resize to 2198×1099, crop 2048×1024). It also uses a correct horizontal flip, applied to the image *and* the mask before cell encoding. | `cityscapes.py:155-171` | **[V]** |
| D10 | **Cityscapes model selection uses the training set.** `evaluate()` iterates `train_loader` with augmentation on and keeps the checkpoint with the best *train* accuracy. The paper says this ("training and validation on the training set", txt:705-708). Val is used only as the test set. | `cityscapes_yolic.py:189-225` | **[V]** |
| D11 | **"polegroup" pixels are labelled as Vehicle.** The custom class table maps polegroup to train_id 18, the same id as bicycle, and the Vehicle group is train_ids 13–18. Caravan, trailer and license plate are mapped to "car" (13). | `cityscapes.py:44,55-56,60` with `cityscapes_yolic.py:107` | **[V]** |
| D12 | **The Cityscapes background ("Road") bit uses a magic cell index, 216.** All-void cells (e.g. ego-vehicle hood, train_id 255) get background = 1 when `num > 216` and an all-zero label when `num ≤ 216`. A companion branch (`value == [0,255] and num >= 216`) can never fire, because it sits inside the "an object is present" branch. | `cityscapes.py:120-135` | **[V]** code; **[I]** "dead branch" |
| D13 | **Cell labels are "≥1 pixel present".** One pixel of a person sets People = 1 for the cell. There is no area threshold. | `cityscapes.py:116-117,137-138` | **[V]** |
| D14 | **The ShuffleNetV2 (S2), QAT (QS2/QM2), ncnn export, Raspberry Pi benchmark and all YOLO-baseline code are missing.** `outdoor_yolic.py` imports a `shufflenet` module that is not in the repo but never uses it. `cityscapes_pred.py` imports a missing `cityscapes2`. The commented QAT stub uses the `'x86'` qconfig, not the ARM (`qnnpack`) backend. | `outdoor_yolic.py:29`; `cityscapes_pred.py:13`; `outdoor_eval.py:44-50` | **[V]** |
| D15 | **Weight file names don't line up across scripts**, and one suggests a 300-epoch Cityscapes run while the paper says 150 epochs. | `outdoor_yolic.py:49` → `mobilenet_outdoor.pth.tar`; `outdoor_eval.py:53` → `mobilenet_outdoor_weight.pth.tar`; `cityscapes_eval.py:115` → `mobilenet_cityscapes_new300.pth.tar` | **[V]** names; **[I]** 300-epoch meaning |
| D16 | **The outdoor eval script reads from a `data_noflip` folder**, which suggests a "flipped" copy of the data existed at some point. | `outdoor_eval.py:129-130` | **[V]** path; **[I]** meaning (see §6) |
| D17 | **The "All" columns in Tables 2, 4 and 6 aren't produced by any script.** They equal the unweighted mean of per-class P and R over object classes (excluding Road/Background), with F1 the harmonic mean of those two means. Checked for indoor YOLIC-M2 (P 0.9445, R 0.9251, F1 0.9347), outdoor YOLIC-M2 (P 0.8889, R 0.8477), outdoor YOLOv8-N (P 0.8489, R 0.6799) and Cityscapes YOLIC-M2 (P 0.8474, R 0.7947, F1 0.8202). | arithmetic on Tables 2/4/6; `classification_report` at `indoor_eval.py:203` would also include the Road and dummy rows | **[I]** (arithmetic matches to 4 decimals) |
| D18 | **Minor paper inconsistencies.** Text says "People" while Table 2 and code say "Creature" (txt:539 vs Table 2; `outdoor_eval.py:61`). Table 1 lists every ShuffleUnit output as "×116" channels although the columns say 116/232/464 (txt:374-376). | as cited | **[V]** |
| D19 | **Reported #Params (3.5 M M2, 2.28 M S2) match the *Cityscapes* head (1024 outputs).** The outdoor head (1248 outputs) gives about 3.8 M for M2. | Table 8/9 vs Eq. 1; MobileNetV2 trunk ≈2.22 M + 1280×1024 FC | **[I]** |

### 2.2 Everything hard-coded to one dataset, layout or setting

| What | Where | Tag |
|---|---|---|
| Cell count and classes per script (104/11, 30/6, 256/3) | `outdoor_yolic.py:47-48`, `outdoor_eval.py:42-43`, `outdoor_pred.py:44-45`, `indoor_yolic.py:36-37`, `indoor_eval.py:42-43`, `indoor_pred.py:44-45`, `cityscapes_yolic.py:31-32`, `cityscapes_eval.py:109-110` | [V] |
| Outdoor cell geometry in absolute pixels of an **848×480** frame (Dataset Ninja confirms 480×848) | `outdoor_pred.py:50-116` | [V] |
| Indoor polygon geometry in absolute pixels (848×480 frame) | `indoor_pred.py:47-74` | [V] |
| Cityscapes cell geometry in absolute pixels of a **2048×1024** frame (16×10 cells of 64×32 in x∈[512,1536], y∈[320,640], plus 16×6 cells of 128×64 for y∈[640,1024]) | `cityscapes_yolic.py:33-106` (duplicated in `cityscapes_eval.py:33-106`, `cityscapes_pred.py:28-101`) | [V] |
| Flip permutation tables, valid only for these exact left-right-symmetric layouts (I checked symmetry about x = 424 for both outdoor and indoor) | `outdoor_yolic.py:109-115`, `outdoor_eval.py:119-125`, `indoor_yolic.py:101-102`, `indoor_eval.py:98-99` | [V] tables; [I] symmetry check |
| Cityscapes class grouping into People / Vehicle / Other / Road via custom train_ids | `cityscapes_yolic.py:107-108`; custom table `cityscapes.py:25-61` | [V] |
| Magic cell index 216 in background labelling | `cityscapes.py:126,131` | [V] |
| Rescale-crop sizes 2198×1099 → 2048×1024 | `cityscapes.py:160-164` | [V] |
| Binary-eval cell range 768–1152 (cells 64–95) | `outdoor_eval.py:156` | [V] |
| Input 224×224 squash, threshold 0.5, Adam 1e-3, milestones [100,125], 150 epochs, batch 32 | e.g. `outdoor_yolic.py:32-37,52,122,156`; threshold `outdoor_eval.py:158` | [V] |
| Split: `train_test_split` on **unsorted `os.listdir`**, `random_state=2`, 0.3 then 0.6666. Because listdir order depends on the OS and filesystem, the exact split isn't portable. | `outdoor_yolic.py:133-135`, `indoor_yolic.py:120-122`, eval/pred scripts | [V] code; [I] portability |
| Absolute Windows paths of the author's machine | `outdoor_eval.py:129-130` | [V] |
| Class-name lists and drawing colours | `outdoor_eval.py:61-63`, `outdoor_pred.py:117-120`, `indoor_eval.py:48`, `cityscapes_eval.py:117`, `cityscapes_pred.py:112-113` | [V] |
| Only the MobileNetV2 path is runnable (head index `classifier[1]`, 1280-d) | `*_yolic.py` | [V] |
| `Image.ANTIALIAS` was removed in Pillow 10; this machine has Pillow 12.3, so Cityscapes training crashes as written | `cityscapes.py:162` | [V] |
| `num_workers=8` and module-level model construction. On Windows (spawn), every worker re-runs the top-level script code. | `outdoor_yolic.py:50,141-150` | [V] code; [I] Windows/RAM impact |

---

## 3. Audit of the paper's own evaluation

### 3.1 Are the baselines fair and strong?
- **[V]** **Different training budgets and procedures.** YOLIC: fixed 150 epochs and fixed hyper-parameters, best checkpoint chosen on val. YOLO: "maximum of 300 epochs … automatic framework settings for the optimizer and batch size, and early stopping" with mosaic disabled (txt:504-520). There was no equal-effort tuning for either side, and no tuning is reported for YOLIC.
- **[V]** **Different input resolution.** YOLO accuracy is reported only at 640×640 (Tables 2 and 6). YOLIC accuracy is at 224×224. Speed is compared at both 640 and 224 (Tables 8 and 9), but **YOLO accuracy at 224 is never reported**. So the paper never shows the accuracy-vs-speed pair at the operating point it uses to claim the speed win.
- **[I]** **Probably not the same metric.** YOLO numbers have 3 significant decimals (e.g. 0.4540, 0.8670), which is typical of Ultralytics `val` output. Ultralytics P/R use box-IoU matching at a confidence it chooses. YOLIC numbers are per-cell, per-class presence counts at threshold 0.5. The paper does not say how YOLO box outputs were mapped back to cells. No baseline code exists (D14). The tables may therefore compare different metrics.
- **[I]** **The baselines look under-trained or mis-specified.**
  - The larger models are much *worse* than the nano ones: outdoor YOLOv8-S All F1 0.4949 vs YOLOv8-N 0.7551, and YOLOv5-S 0.3820 vs YOLOv5-N 0.4081.
  - TrafficSign is at or near 0 for every YOLO model (YOLOv8-N: P = 1.000, R = 0.0183).
  - YOLOv5-N Fence has P = 0.0937.
  - These patterns point to a problem converting cell labels into box labels, or to early stopping firing too soon. They don't look like a capability limit of YOLO.
- **[I]** **The outdoor lead partly comes from one class.** On outdoor, the TrafficSign column alone pulls YOLOv8-N's macro recall down by about 0.07. Without TrafficSign, my arithmetic from Table 2 gives YOLIC-M2 F1 ≈ 0.877 vs YOLOv8-N ≈ 0.787 (with it: 0.868 vs 0.755). YOLIC still leads, by a smaller margin.
- **[I]** **Missing baselines a reviewer would expect:**
  - (a) a lightweight **semantic-segmentation** model (e.g. LR-ASPP-MobileNetV3, Fast-SCNN, BiSeNet), pooled into cells. This is the natural baseline because Cityscapes labels are pixel masks and YOLIC's own labels are derived from them.
  - (b) a standard detector trained on **real object boxes**, with its boxes rasterised into cells afterwards. YOLO was only given cell-shaped boxes.
  - (c) the authors' own earlier per-cell crop classifier [23]/[33].
  - (d) the same backbone with a **spatial head** (1×1 conv on the 7×7 map) instead of GAP + FC.
  - (e) a YOLO/NanoDet model *with a ShuffleNetV2 backbone*. Without it, the backbone's effect on speed can't be separated from the method's.
- **[V]** **No baseline on the indoor task at all** (txt:664-667).

### 3.2 Seeds, variance, statistics
- **[V]** Each result is a single run, from one fixed seed (`--seed 1`) and one fixed split (`random_state=2`). No standard deviations, confidence intervals or significance tests appear anywhere. (`outdoor_yolic.py:40,57,134`; all tables.)
- **[I]** Many of the paper's comparisons differ by 0.5–3 F1 points: S2 vs M2, Q vs FP, YOLIC-M2 vs YOLOv5-N on Cityscapes (0.8202 vs 0.8134). From a single run these are within plausible seed-to-seed noise, so they don't establish a ranking.
- **[V]** **FPS has no variance either.** The paper gives no repeat count, no warm-up, no thread count, and does not say what the timing includes (pre-processing, NMS) or which framework produced Table 8. (Table 8, txt:795-807.)

### 3.3 Other things a reviewer would want
- **[I]** **Possible data leakage in the in-house splits.** Outdoor and indoor are frames from continuous video. Indoor frames were taken "at one-second intervals from multiple videos" (txt:637-639); outdoor is 20,380 frames from campus footage (txt:545-547). The split is a *random per-frame* split (`outdoor_yolic.py:134-135`), so near-duplicate neighbouring frames almost certainly fall into both train and test. That inflates test scores. Nothing tests on a different video, route, day or site.
- **[I]** **The metrics hide class priors.** There is no mAP/AP and no PR curve, only a single 0.5 threshold. Binary "Road" F1 is dominated by easy, plentiful empty cells. Per-class support counts are not reported.
- **[V]** **No ablations at all.** The paper never varies the number or size of cells, the background bit on/off, multi-label vs softmax, input size, head design, loss, or augmentation.
- **[I]** **No robustness tests.** Nothing covers night, rain, motion blur, or camera pitch and roll changes. The pitch/roll gap matters for a scooter, where vibration and pitch move the ground plane relative to fixed cells.
- **[I]** **Speed-measurement concerns.**
  - "FP16" on a Pi 4 CPU (Cortex-A72, ARMv8.0-A, no FP16 arithmetic) probably means FP32 compute.
  - YOLIC-QM2 is faster in PyTorch (35.72) than in ncnn (28.42), which is unusual and unexplained.
  - YOLIC-M2 at 13.48 FPS is close to YOLOv5-N@224 at 12.07 FPS (Table 8), so most of the speed gap comes from S2, i.e. from the ShuffleNetV2 backbone.

### 3.4 Which claims the evidence supports
| Claim | Verdict | Tag |
|---|---|---|
| YOLIC-S2/QS2 runs >30 FPS at 224 on a Pi 4B | Plausible: the GFLOPs (0.30 G) are consistent with it. But it can't be reproduced, because no benchmark code was released and the setup is under-specified. | [V] numbers; [I] plausibility |
| Faster than YOLOv5-N/v8-N at the same 224 input on the Pi | Supported *for speed* (Tables 8/9), though the accuracy at that setting is not reported. | [V] |
| Beats YOLO on the outdoor in-house data (All F1 0.868 vs 0.755) | Weakly supported. The baselines look broken (S < N, TrafficSign ≈ 0), the metrics probably differ, the split likely leaks, and there is one run. | [I] |
| "Comparable" to YOLO on public Cityscapes | **Overstated.** YOLIC-M2 All F1 0.8202 is *below* YOLOv5-S (0.8296), YOLOv8-N (0.8268) and YOLOv8-S (0.8333). It beats only YOLOv5-N and YOLOv6-N. The quantized and S2 versions (0.77–0.79) are below all but YOLOv6-N. The text still says YOLIC "maintain[s] a competitive edge" (txt:765-767). | [V] table; [I] verdict |
| Same, restricted to the *object* classes (People, Vehicle) | **Contradicted.** "All" includes "Other", a catch-all of stuff classes (sidewalk, building, vegetation, sky…, `cityscapes_yolic.py:107-108`). Boxes fit stuff classes badly, and YOLIC wins there (0.92 vs about 0.78). On People + Vehicle only, my macro-F1 from Table 6 gives YOLIC-M2 ≈ 0.768 vs YOLOv8-N ≈ 0.831, YOLOv5-N ≈ 0.822, YOLOv8-S ≈ 0.851. People: 0.674 vs 0.78–0.83. | [I] arithmetic from [V] table |
| Works with irregular cells (indoor) | Shown to train and score high, but with no baseline and a likely leaking split. | [I] |
| "First method with 224 input >30 FPS on Raspberry Pi CPU" | Unsupported as stated. The paper does no systematic survey, and it omits ultra-light detectors (e.g. the Yolo-Fastest family) that report Pi-class speeds. I have not checked those numbers. | [I] |
| "Reduces annotation effort" (txt:353-356) | Not measured anywhere. | [V] absence |
| Quantization keeps accuracy | Supported within single-run noise: about −1 to −2 F1 points (Tables 2–7). The QAT code is not released. | [V] tables; [V] code absence |

---

## 4. Limitations

### 4.1 Admitted by the paper
- **[V]** The head grows with the number of CoIs (N×(M+1)×feature-dim), which increases parameters and memory (txt:918-921).
- **[V]** Not instance-level. It suits applications that "do not require precise instance-level detection" (txt:115-122).
- **[V]** YOLO "may not be as suitable" for CoI classification as YOLIC (txt:514-516), an implicit admission that the comparison is off-task for YOLO.
- **[V]** No YOLO comparison on the irregular-cell indoor task (txt:664-667).
- **[V]** Consecutive frames are not used; the paper names temporal use as a possible improvement (txt:876-877).

### 4.2 Not admitted (with evidence)
- **[I]** **Tied to one camera.** Cells are absolute pixel coordinates for one resolution and mount (`outdoor_pred.py:50-116`, `cityscapes_yolic.py:33-106`). Changing the camera height, pitch, lens or resolution invalidates both labels and model. Each new deployment needs new annotation and retraining, so a trained model doesn't transfer.
- **[I]** **The localization bottleneck.** Position must survive global average pooling (Table 1, GAP layer). Consistent with this, the class with the smallest typical footprint does worst: Cityscapes People F1 is 0.60–0.67 (Table 6), and small upper cells shrink to about 7×3.5 px at 224. This is consistent with the hypothesis but not proven by it.
- **[I]** **Coarse, noisy labels.** "Any pixel present" labelling (`cityscapes.py:137`), with no area threshold, makes labels sensitive to single-pixel boundaries. The arbitrary void rule (`cityscapes.py:126-131`) and the polegroup→Vehicle mapping (`cityscapes.py:44`) add noise.
- **[I]** **"Other" class.** On Cityscapes, "Other" is present in a large share of cells (it includes building, sky, vegetation and sidewalk), which inflates the "All" average. Class frequencies are never reported.
- **[I]** **Generalisation is unmeasured.** Outdoor and indoor are single-site datasets (University of Aizu campus, txt:545-547), with random per-frame splits (D-table and §3.3).
- **[I]** **Augmentation bug (D1).** Probably the reported in-house numbers *understate* what YOLIC can reach, but it also makes the paper's results hard to reproduce exactly.
- **[I]** **No distance or count output.** Distance comes only from the cell prior on a flat ground plane. Multiple objects of one class in a cell can't be told apart.
- **[I]** **Plain BCE with a fixed 0.5 threshold.** There is no handling of the heavy background/foreground imbalance (`outdoor_yolic.py:155`). For a safety use case, recall on People/Creature (0.79–0.82 outdoors, Table 2) is never discussed as a risk.
- **[V]** **Reproducibility gaps.** Half the reported model variants (S2, QS2, QM2), all baselines and all speed numbers have no released code (D14). The dataset split is not pinned to file lists (`outdoor_yolic.py:133`).

---

## 5. Resources

### 5.1 What the method needs
- **[V]** RGB frames from a fixed camera, a cell layout (hand-designed; the authors' Cell Designer tool), and per-image multi-hot labels of length N×(M+1) (txt:340-399).
- **[I]** These labels can come from three sources:
  - (a) direct cell annotation with the authors' YOLIC-Labeling tool;
  - (b) pixel-level semantic masks, as the Cityscapes conversion shows (`cityscapes.py:111-139`);
  - (c) bounding boxes, by box–cell overlap.
  Any public segmentation dataset with a roughly fixed forward-facing camera can be converted. Examples, not verified for licence here: Cityscapes, BDD100K, Mapillary Vistas, CamVid, KITTI.

### 5.2 Public availability

| Resource | Status | Tag |
|---|---|---|
| Outdoor Hazard Detection dataset (20,380 frames, 848×480, 104 cells × 12) | Public on Kaggle (`sukai3316/outdoor-hazard-detection-dataset`). Dataset Ninja lists about **13 GB**, licence **DbCL v1.0**, split 14,266/2,038/4,076. Dataset Ninja also gives conflicting counts ("20,278 labeled; 4,102 without annotations"), and the Kaggle page couldn't be read without JavaScript. | [V] links and Ninja figures; whether official split lists ship is unknown |
| Indoor Obstacle Avoidance dataset (6,410 frames, 30 cells × 7) | Linked from the project page on Kaggle (`sukai3316/indoor-obstacle-avoidance-dataset`). I could not read its size or licence. | [V] link only |
| Cityscapes (2,975 train / 500 val, fine pixel masks) | Public, free for non-commercial research. **Requires registration.** Needed: `leftImg8bit_trainvaltest` (~11 GB) + `gtFine` (~0.25 GB). | [V] paper txt:703-708; sizes [I] |
| Original object-level boxes or masks for outdoor/indoor | Probably **not** available; the released labels appear to be cell-level only. A YOLO baseline on real object boxes can't be built from them. | [I] — see §6 |
| Code | Training/eval for MobileNetV2 only (this repo). No S2/QAT/ncnn/Pi/YOLO-baseline code. | [V] |
| Pretrained YOLIC weights | Not linked on the project page; the scripts expect local `.pth.tar` files. | [V] |
| Tools | Cell Designer (github.com/kai3316/Cell-designer), YOLIC-Labeling (github.com/kai3316/YOLIC-Labeling) | [V] links (not inspected) |

### 5.3 Fit to your constraints (RTX 4050 Laptop 6 GB, i7-12650H, 7.6 GB RAM, 3 months, 4th-year B.Tech, IEEE journal)
- **[I]** **GPU is sufficient.** MobileNetV2/ShuffleNetV2 at 224×224, batch 32, uses well under 6 GB, and mixed precision gives extra headroom. YOLOv8-N/S training at 640 also fits at batch 16–32.
- **[I]** **RAM is the main bottleneck.** With 7.6 GB total, `num_workers=8` on Windows spawns 8 processes. Each re-runs the top-level script code (model build, file listing), so RAM can run out. Use 2–4 workers, wrap top-level code in `if __name__ == '__main__':`, or run under WSL2/Linux.
- **[I]** **Data loading will be CPU-bound, Cityscapes especially.** Cityscapes decodes 2048×1024 PNGs and runs `np.unique` over 256 cells for every sample, every epoch (`cityscapes.py:115-117`). Pre-computing cell labels and caching 224×224 images once would speed this up a lot.
- **[I]** **Rough time per run.** The outdoor script evaluates the *whole training set* every epoch (`outdoor_yolic.py:250`), about 34 k images per epoch. That gives an estimated 1–2 min/epoch, or **~3–5 h per 150-epoch run**. Cityscapes is similar. A YOLOv8-N baseline at 640 for up to 300 epochs is estimated at 10–20 h. Multi-seed studies (≥3 seeds × several configs) are feasible within 3 months if you schedule runs carefully. All of these are estimates; time a single epoch to confirm.
- **[I]** **Disk:** about 25–30 GB for the outdoor set plus Cityscapes, plus caches.
- **[I]** **The speed claims need a Raspberry Pi 4B,** which is not in your listed hardware. Without one, timings on the laptop CPU are only a proxy and can't support or refute the paper's Pi FPS numbers.
- **[I]** **No Pillow downgrade is needed** if `Image.ANTIALIAS` is replaced with `Image.LANCZOS`. The current Pillow 12.3 crashes on the original line.
- **[V]** PyTorch is not installed in this environment (checked with `python -c "import torch"`).

---

## 6. Open questions I could not resolve
1. Was Table 3 (outdoor binary) computed on all 104 cells, or only cells 64–95 as `outdoor_eval.py:156` does? Same question for whether the Table 2 per-class numbers came from the commented block (`outdoor_eval.py:231`).
2. How were YOLO predictions converted to per-cell decisions, and how were the YOLO P/R values computed: Ultralytics box-IoU P/R, or per-cell presence? Were YOLO models COCO-pretrained? Which early-stopping patience was used?
3. Were the published results produced with the vertical-flip bug (D1), and with augmented validation during checkpoint selection (D2)? Or by a different, unreleased script?
4. What is `data_noflip`? Was there an offline-flipped copy of the outdoor data, and if so, could flipped twins of a frame land on both sides of the split?
5. Do the Kaggle releases include official train/val/test file lists, video or sequence IDs, and any object-level boxes or masks? I could not read the Kaggle pages. Without sequence IDs, a leakage-free split can't be rebuilt.
6. Were the Cityscapes results from a 150-epoch or a 300-epoch model (`mobilenet_cityscapes_new300.pth.tar`)?
7. Speed setup: framework and precision for Table 8 (true FP16, or FP32?), thread count, OS, whether pre-processing and NMS are timed, number of repeats. Why is the QM2 model faster in PyTorch than in ncnn? Which quantization backend was used on ARM, given the `'x86'` stub in code?
8. Which model sizes (#Params, FLOPs) go with which dataset head? Table 8 seems to report the Cityscapes head (D19).
9. Fig. 6 (indoor): predictions, or ground truth (D5)? Were Figs. 4 and 8 generated with BGR input (D6)?
10. What is the label distribution per class and per cell (share of positive cells) in each dataset? How dominant is "Other" on Cityscapes?
11. What licence does the indoor dataset carry, and what are its exact size and format?
12. How does the "Pioneering" (>30 FPS) claim compare against ultra-light detectors not cited (e.g. Yolo-Fastest/FastestDet) under identical Pi conditions? Not checked in this pass.
