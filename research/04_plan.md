# 04 — Research plan: survivor #1, a benchmark of cell-presence heads

Source of the idea: `research/03_review.md` §"Ranked survivors" #1 (N2 + E1 + C2-baseline + N1 + R2 + S2).
Inputs read: `research/01_brief.md`, `research/03_review.md`, and the code at the lines cited below.
No code is written in this step.

**Two placeholders in the request were left blank. I filled them as follows; change them if they're wrong.**
- **[N] GPU-days for the go/no-go = 2.** The go/no-go below is estimated at about 12–14 GPU-hours.
- **[constraints]** are the ones in `01_brief.md` §5.3: one RTX 4050 Laptop (6 GB), i7-12650H, 7.6 GB RAM, Windows, no Raspberry Pi, 3 months, 4th-year B.Tech, target an IEEE journal.
  The start date is Mon 2026-10-05.

**How numbers are tagged.**
- *(paper)* means a number from the YOLIC paper.
- *(brief)* or *(review)* means a number derived in those files.
- *(est.)* means my estimate, which must be measured before anyone relies on it.

All thresholds in §3 were fixed on 2026-10-02 and must not change after results come in.

---

## 1. Paper story

**Working title.** *Keep the Map: A Leakage-Free, Matched-Compute Benchmark of Cell-Presence Heads for Edge Hazard Perception.*

**One-sentence claim (the hypothesis that the go/no-go tests).**
YOLIC's global-average-pool + FC head is the weak point of fixed-cell hazard classifiers. At matched compute, under a leakage-free, multi-seed protocol with one shared per-cell metric, keeping the spatial map is worth at least +2 F1 on people and vehicles, and the gain is concentrated in the small, far cells. "Keeping the spatial map" means either a cell-pooling head or a light segmenter pooled into cells.

**Contributions.**
1. **A protocol for fixed-cell classifiers, with open code.** It applies to YOLIC, UFLD-style and other grid classifiers, and has five parts:
   - leakage-free splits (outdoor frames grouped by perceptual hash or session; Cityscapes validation by held-out city);
   - three or more seeds per configuration;
   - one per-cell metric for every model family: F1 at 0.5, F1 at a tuned threshold, per-class AP, and per-cell-size breakdowns;
   - FLOPs and parameter counts;
   - a laptop-CPU latency labelled as a proxy.
2. **A matched-compute head benchmark with a failure analysis.** It compares:
   - GAP+FC (YOLIC);
   - a UFLD flatten head;
   - 1×1 conv + cell pooling at stride 32 and stride 16;
   - an LR-ASPP-MobileNetV3-Small segmenter pooled into cells;
   - ShuffleNetV2-class and YOLOv8-N detectors trained on **real instance boxes** and rasterised to cells.

   All results are broken down by cell size and by distance from the image border (the useful part of N1/N3).
3. **Measured effect of each protocol artefact**, as ΔF1 with confidence intervals:
   - the flip bug (D1);
   - augmented validation used for checkpoint selection (D2/D10);
   - label rules (D11 polegroup, D12 void/216, ≥1 px vs ≥k px);
   - random-frame split vs grouped split on the outdoor set.

**Headline result.** Write it in this form once the numbers exist. The thresholds in §3 decide whether it can be written at all.
> "On Cityscapes People+Vehicle, CellPool-M2 reaches macro-F1 X ± s against Y ± s for bug-fixed YOLIC-M2 (3 seeds) at ≤1.2× its MACs. +Δs of that gain is on the 160 small cells and +Δl on the 96 large ones. A random per-frame split inflates outdoor object F1 by Z points over a grouped split. The paper's reported lead over detectors [holds / shrinks to … / reverses] once the detectors are trained on real boxes."

Reference points the reader will compare against:
- YOLIC-M2 on Cityscapes: All F1 **0.8202** *(paper)*.
- People+Vehicle macro-F1: about **0.768** for YOLIC-M2 vs about **0.831** for YOLOv8-N *(brief, arithmetic from Table 6)*.

**Fallback story, if the go/no-go says the head doesn't matter.** See §3.4: *"Protocol, not architecture, drives fixed-cell classifier results."*

---

## 2. Experiments

### 2.1 Datasets and splits

| Dataset | Role | Split (fixed before any run) |
|---|---|---|
| **Cityscapes** fine (2,975 train / 500 val) | Primary. It has pixel masks (for the segmenter) and instance masks (for the detectors' real boxes). | Hold out whole train cities totalling about 15% of the train set as `val'`, used for checkpoint selection and threshold tuning. The official val (500) is the **test** set and is never used for selection. Write the city list to `splits/cityscapes_val_cities.txt` before the first run. |
| **Outdoor Hazard** (Kaggle, about 20k frames, 104 cells × 12) | Protocol ladder, leakage test, head benchmark. No masks or boxes, so no segmenter or detector here. | Two splits, both 70/10/20. (a) The paper's `train_test_split(random_state=2)` applied to the **sorted** filename list. (b) A grouped split: 64-bit pHash, frames joined if Hamming distance ≤ 8, connected components used as groups, then GroupShuffleSplit with seed 0. If filenames turn out to encode sessions, use the session as the group and keep pHash as a check. Freeze both as `.txt` file lists. |
| **Indoor** (Kaggle, 6.4k frames, 30 polygon cells) | Stretch goal: a head benchmark on irregular polygon cells. | Grouped split as for outdoor. Run it only if the licence is OK and W10 has slack. |

### 2.2 The shared metric (one implementation for every model family)

- **Per-class cell P/R/F1 at 0.5.** Count TP, FP and FN for each (cell, class) bit, pooled over all cells and test images.
  - This is exactly what `cityscapes_eval.py:144-155` and the `classification_report` call at `:218` compute, because the dummy "Background" index turns each bit into one TP, FP or FN.
  - A unit test must reproduce the old script's per-class numbers to 1e-6 on the same predictions.
- **Aggregates:**
  - macro over **object classes**: People+Vehicle on Cityscapes, the 11 hazard classes outdoors;
  - the paper's "All" (macro P and R excluding Road, F1 as their harmonic mean; D17), kept for comparability only.
- **Threshold-free and tuned:**
  - per-class AP;
  - F1 at a per-class threshold tuned on `val'`. This row stands in for C1, since logit adjustment is equivalent to threshold tuning.
- **Rule variants:** binary Risk/Road over **all** cells, not only cells 64–95 (D3). The paper's "background wins" rule (D4) is reported as an alternative decision rule.
- **Breakdowns:**
  - by cell-size group: Cityscapes has 160 cells of 64×32 and 96 cells of 128×64; outdoor uses its row bands;
  - by distance from the image border;
  - by per-class positive rate, with supports reported.
- **Statistics:**
  - mean ± std over seeds;
  - for each paired head comparison, a bootstrap 95% CI over test images (1,000 resamples, seeds pooled);
  - Holm correction across the head-comparison family.
- **Cost:**
  - parameters and MACs at the model's input size, measured with one counter (fvcore) for every model;
  - ONNX Runtime latency on the i7-12650H: single thread, 50 warm-up runs, 500 timed runs, median and p90, **labelled as a laptop proxy**;
  - no Pi FPS claims.

### 2.3 Fixed training protocol ("v1")

Settings:
- Adam, lr 1e-3, MultiStepLR at epochs [100, 125], 150 epochs, batch 32, 224×224 squash. This is the paper's recipe, deliberately not retuned.
- Horizontal flip done correctly.
- ImageNet normalisation.
- ColorJitter with hue 0.1 (other jitter values unchanged).
- AMP.

Selection and evaluation:
- Keep the checkpoint with the best `val'` object macro-F1 at 0.5.
- Don't augment validation.
- Never evaluate on test during training.

Seeds control head initialisation, data order and augmentation. The splits stay fixed.

### 2.4 Every baseline the review demanded

Every row is trained under v1 with 3 seeds unless marked otherwise.

| # | Baseline | Datasets | Why the review demands it |
|---|---|---|---|
| B0 | **YOLIC-M2 as released.** The original scripts, changing only the Pillow `ANTIALIAS` call, the worker count and the paths. 1 seed with the untouched script, plus 3 seeds through a bug-compatible mode of the new pipeline. | CS, Outdoor | Reproduction anchor. Must land within ±0.03 of the paper's All F1. |
| B1 | **YOLIC-M2 fixed** (v1) | CS, Outdoor | "YOLIC with the bugs fixed" |
| B2 | **YOLIC-S2**: ShuffleNetV2 x1.0 + GAP+FC (torchvision; the repo's `shufflenet` module is missing, D14) | CS, Outdoor | The paper's speed model. Separates backbone from head. |
| B3 | **UFLD flatten head**: 1×1 conv to 8 channels → flatten → FC 2048 → FC N(M+1) | CS, Outdoor | "flatten head in the style of UFLD" |
| B4 | **CellPool-s32**: 1×1 conv on the 7×7 map, area-weighted pooling per cell (cell masks rasterised on a 4×-upsampled grid), plus a per-(cell, class) bias | CS, Outdoor | "1×1 conv + cell pooling at stride 32"; the E1/CAM identity |
| B5 | **CellPool-s16**: as B4, with the last MobileNetV2 stage dilated. MACs reported; it is *not* compute-matched. | CS, Outdoor | "… and stride 16" |
| B6 | **S2-CellPool-s32** | CS, Outdoor | Head effect on the speed backbone |
| B7 | **MNV3-Small + GAP+FC** | CS | Backbone control for B8 |
| B8 | **LR-ASPP-MNV3-Small pooled into cells**, trained on 4-group pixel masks at 224 input. Cell score = max over pixels in the cell (matches the "≥1 px" rule). Built by hand, since torchvision only ships the Large builder and has no Cityscapes weights. | CS | **Strongest obvious baseline.** The review calls the pooled light segmenter "the baseline that matters" (C2) and says it "likely matches whenever masks exist" (A1). |
| B9 | LR-ASPP-MNV3-Large pooled (1 seed, upper reference, not compute-matched) | CS | Ceiling for the segmenter family |
| B10 | **FastestDet** (ShuffleNetV2) at 224, trained on real instance boxes (People = person+rider; Vehicle = car, truck, bus, train, motorcycle, bicycle). Cell score = max score of any box overlapping the cell. | CS | "FastestDet/Yolo-FastestV2 trained on real boxes at 224 and rasterised" |
| B11 | **YOLOv8-N at 224** on real boxes; YOLOv8-N at 640 (1 seed, reference) | CS | The paper's own strongest-looking baseline, retrained correctly at YOLIC's operating point |
| B12 | **Box oracle**: GT boxes rasterised with the same rule, scored against the mask-derived labels | CS | The ceiling any box detector can reach under the cell rule. It separates "detector is weak" from "boxes over-cover cells". |
| B13 | **Per-class tuned thresholds** on B1 and on the best head (post-hoc, no training) | all | C1 stand-in |
| B14 | **Crop-to-cell-union input** with matched aspect ratio (CS: crop [512:1536]×[320:1024]) on B1 and the best head | CS | E2 stand-in |
| — | The paper's reported numbers, as quoted rows | — | Context only, never mixed with our metric |

The detectors (B10–B12) only cover People and Vehicle, because the "Other" and "Road" stuff classes have no boxes. That is why **People+Vehicle macro-F1 is the primary cross-family metric.**

### 2.5 Ablations

| Ablation | Settings | Runs |
|---|---|---|
| **Protocol ladder, Cityscapes** (cumulative) | B0 → +selection on `val'` (D10) → +label fixes (polegroup → pole, D11; void cells masked instead of the 216 rule, D12) → +norm/hue (D7/D8) = B1 | 3 steps × 3 seeds |
| **Protocol ladder, outdoor** (cumulative, random split) | B0 → +flip fix (D1) → +clean val selection (D2) → +norm/hue = B1 | 3 steps × 3 seeds |
| **Leakage** | B1 on the random split vs the grouped split. Also 2 extra grouped-split draws × 1 seed, to show variance from the split itself. | 3 + 2 |
| **Head family** | B1–B9 (§2.4) | see table |
| **Pooling operator** in CellPool | average vs max (s32, CS only) | 1 extra × 3 |
| **Label rule** (S2) | ≥1 px (paper) vs ≥16 px vs ≥1% of the cell's area. Train and evaluate under the same rule. Applied to B1 and the best head. | 2 rules × 2 × 3 |
| **Input** | 224 squash vs crop-to-union (B14) | 2 × 3 |
| **Decision rule** | 0.5 / tuned / background-wins (D4) | post-hoc |
| **Cell size and border** | breakdown of every model above | post-hoc |

### 2.6 Run budget (est.; re-estimate after timing one epoch in W2)

| Block | Runs | Est. h/run | GPU-h |
|---|---|---|---|
| Cityscapes classifiers (B0-compat, ladder, B1–B9, pooling, label rule, crop) | about 55 | 0.5 (cached 224 inputs, no per-epoch evaluation on the train set) | about 28 |
| Cityscapes detectors (B10 ×3, B11-224 ×3, B11-640 ×1) | 7 | 3 (224), 10–20 (640, brief) | about 33 |
| Outdoor (B0, ladder, leakage, B1–B6 on the grouped split) | about 32 | 1.5 | about 48 |
| Untouched original scripts (CS and outdoor, 1 seed each) | 2 | 3–5 (brief) | about 8 |
| Indoor (stretch) | 9 | 0.5 | about 5 |
| **Total** | | | **about 120 h ≈ 5 GPU-days**, about 7 with a 40% re-run margin |

Seeds: **3 for everything in the main tables**, 5 only for comparisons that land in the ambiguous zone of §3. Single-seed rows (B9, B11-640, split-variance draws) are marked as such in the tables.

---

## 3. Go/no-go experiment

### 3.1 What it tests

It tests the paper story's load-bearing assumption: *a compute-matched head that keeps the spatial map beats GAP+FC by more than seed noise.* It also tests whether contribution 3's leakage effect is large enough to carry a fallback paper.

### 3.2 Runs (budget cap **2 GPU-days**; est. 12–14 GPU-h)

**Arm H (head), on Cityscapes under protocol v1, 3 seeds each:**
- **A** = B1 (YOLIC-M2 fixed, GAP+FC);
- **B** = B4 (CellPool-s32, compute-matched);
- **C** = B8 (LR-ASPP-MNV3-S pooled; compute-matched if its MACs are ≤1.2× A's);
- **D** = B5 (CellPool-s16; informative only, not used in the decision).

**Reproduction anchor:** B0 with the untouched `cityscapes_yolic.py`, 1 seed.

**Arm L (leakage), on outdoor:** B1, 1 seed on the random split and 1 seed on the grouped split.

**Prerequisites, with no GPU:**
- Cityscapes cache;
- outdoor pHash grouping;
- the metric module passing its equivalence unit test.

If the arms reach 48 GPU-hours without a decision, stop and re-scope (§3.4, row "over budget").

### 3.3 Decision rule (thresholds fixed 2026-10-02)

Primary metric: **People+Vehicle macro-F1 at 0.5 on Cityscapes val (test), all 256 cells, mean of 3 seeds.**
Δ = (best of B, C) − A, in F1 points. Only compute-matched alternatives count.

| Check | Condition | Action |
|---|---|---|
| **Gate: reproduction** | A's "All" F1 and B0's "All" F1 each within **0.8202 ± 0.030** | If either misses, debug the pipeline before reading Δ. If the untouched B0 itself misses, record it as a finding and email the authors. Either way it's a gate, not a kill. |
| **GO** | **Δ ≥ 2.0** *and* all 3 seeds of the winner score above all 3 seeds of A (exact one-sided permutation p = 0.05) | Proceed with the paper story in §1. |
| **Ambiguous** | 1.0 ≤ Δ < 2.0, or the seeds overlap | Add 2 seeds to A and the winner (about 1 GPU-h). Go if the 5-seed mean Δ ≥ **1.5** *and* the paired-bootstrap 95% CI excludes 0; otherwise treat as FAIL. |
| **REVERSE** | A beats both B and C by ≥ 2.0 under the same seed rule | Go, with the head contribution reframed: "GAP+FC is a sufficient head, and YOLIC's deficit against detectors on objects comes from resolution and labels, not the head." The benchmark still stands. |
| **FAIL** | \|Δ\| < 1.0 after the above | The head story is dead. Go to §3.4. |

Arm L threshold: **leakage is material** if random-split object macro-F1 − grouped-split object macro-F1 **≥ 3.0 points**. 3.0 is well above the single-seed noise expected from the paper's 0.5–3 point gaps (brief §3.2).

### 3.4 If it fails

| Outcome | Action |
|---|---|
| Head FAIL, leakage ≥ 3.0 | **Fallback framing:** *"Protocol, not architecture, drives fixed-cell classifier results."* Contributions become (1) the protocol, (3) the artefact and leakage effect sizes, and (2) head equivalence reported as a measured negative result. Keep B10–B12 (detectors vs cells on real boxes). Venue: still IEEE Access; or TMLR / MLRC / ReScience C if dropping the IEEE requirement is acceptable. |
| Head FAIL, leakage < 3.0 | **Kill survivor #1.** Pivot to survivor #2 (layout transfer from cell labels). It reuses the Cityscapes cache, the metric module, the layout rasteriser and the CellPool head built in W1–W2, so roughly 2 weeks of work carry over. |
| Gate fails and can't be fixed in one week | Pause. Write up the reproduction gap (with the authors' reply, if any) as a short reproducibility report, and choose between survivors #2 and #3 using the working pipeline. |
| Over budget (>48 GPU-h without a decision) | Drop arm L and B5, and decide on A vs B only with the same thresholds. |

**Decision date: Fri 2026-10-23.**

---

## 4. Code changes against this repo (plan only)

**Strategy.**
- Tag the current tree `v0-released` and **do not edit the original 10 scripts**. They are the B0 reference.
- All new work goes in a package `yolic_bench/` plus `configs/`, `layouts/`, `splits/` and `tools/`.
- The B0 "bug-compatible mode" re-creates the released behaviour behind flags, so every artefact in the ladder can be switched on and off on its own.

### 4.1 Changes and generalisations, file by file

| Where | What it does now | What changes in `yolic_bench/` | Generalise to |
|---|---|---|---|
| `outdoor_yolic.py:29` | imports the missing `shufflenet` | use `torchvision.models.shufflenet_v2_x1_0` | backbone factory {mnv2, shufflenetv2, mnv3s} |
| `outdoor_yolic.py:47-51`, `indoor_yolic.py:36-40`, `cityscapes_yolic.py:31-32,111-112` (also the eval and pred scripts, brief §2.2) | `NumCell` / `NumClass` constants; `model.classifier[1] = nn.Linear(1280, N*(M+1))` | `heads.py`: `GapFc`, `UfldFlatten`, `CellPool(stride, pool)`, built from `(backbone, layout, classes)` | N and M derived from the layout and class config |
| `outdoor_pred.py:50-116`, `indoor_pred.py:47-74`, `cityscapes_yolic.py:33-106` (duplicated in `cityscapes_eval.py:33-106`, `cityscapes_pred.py:28-101`) | cell geometry as absolute pixels for one frame size | `layouts/{outdoor,indoor,cityscapes}.json` in normalised coordinates plus the source frame size. `layout.py` rasterises cell masks at any resolution (needed by CellPool, segmenter pooling, detector rasterisation and crop-to-union). | any rectangle or polygon layout and any input size |
| `outdoor_yolic.py:109-115`, `outdoor_eval.py:119-125`, `indoor_yolic.py:101-102`, `indoor_eval.py:98-99` | hard-coded flip permutation tables | derive the permutation by mirroring the layout geometry. **Unit test:** the derived permutation equals the hard-coded tables, which also proves the `outdoor_pred.py` cell order matches the label order. | any left-right-symmetric layout; refuse to flip an asymmetric one |
| `outdoor_yolic.py:64`, `indoor_yolic.py:53` | `image.flip(1)` on a CHW tensor flips vertically (D1) | flip the PIL image before `ToTensor` (or `flip(-1)`); `--compat-flip-bug` reproduces the old behaviour | — |
| `outdoor_yolic.py:88-117,138-139` | the outdoor dataset has no train flag, so the flip also hits val and test (D2) | the augmentation is owned by the train transform only; `--compat-aug-val` | — |
| `outdoor_yolic.py:120-129`, `cityscapes_yolic.py:122-131` (same in indoor) | no Normalize (D7); `hue=0.5` (D8) | `configs/*.yaml` transform block; v1 = Normalize + hue 0.1 | config |
| `outdoor_yolic.py:131-135`, `indoor_yolic.py:118-122`, `outdoor_eval.py:129-133` | `train_test_split` over **unsorted** `os.listdir`; absolute Windows paths | `tools/make_splits.py` writes frozen `splits/*.txt` (random-sorted and pHash-grouped); the paths come from config | any dataset with a file list and optional group key |
| `outdoor_yolic.py:57-59` (all `*_yolic.py`) | seeds torch only; `random.random()` at `:108` runs in workers | seed `random`, `numpy` and `torch`, with a DataLoader `generator` and `worker_init_fn`; log the seed per run | — |
| `outdoor_yolic.py:141-150`, `cityscapes_yolic.py:137-142`; model built at module level (`:50`, `:111`) | `num_workers=8` on Windows spawn re-runs the whole module per worker (RAM risk) | everything inside `main()`; workers from config (default 3); persistent workers | — |
| `outdoor_yolic.py:159-174,197-233,250-252` | selects on cell exact-match accuracy; evaluates the **whole train set** and the test set every epoch | select on `val'` object macro-F1; never touch train or test during training; test evaluation runs once on the selected checkpoint | — |
| `cityscapes_yolic.py:189-225` | checkpoint selected on **train** accuracy, with augmentation on (D10) | select on `val'` (held-out cities) | — |
| `cityscapes.py:44` | polegroup → train_id 18 = Vehicle (D11) | label table in config; v1 maps polegroup → pole (train_id 5, "Other") | configurable class grouping |
| `cityscapes_yolic.py:107-108`, `cityscapes_eval.py:107-108` | People/Vehicle/Other/Road groups as train_id tuples | `configs/cityscapes_groups.yaml` | any mask dataset with a group map |
| `cityscapes.py:120-135` | background bit set by "no object" plus the magic index 216 for void cells; dead branch at `:126` (D12) | explicit rule: background = no object present; cells whose void fraction exceeds τ_void are **masked out of loss and metric**; `--compat-216` reproduces the old rule | no index-dependent logic |
| `cityscapes.py:137` | object present if ≥1 pixel (D13) | `min_pixels` / `min_fraction` parameter (label-rule ablation) | config |
| `cityscapes.py:149-177` | decodes the full 2048×1024 PNG and runs `np.unique` over 256 cells per sample per epoch; `Image.ANTIALIAS` at `:162` crashes on Pillow 12 | `tools/cache_cityscapes.py`: images as a uint8 memmap at 1024×512; cell labels precomputed **from full-resolution masks** for each discrete augmentation state (flip × 8 fixed rescale-crop offsets replacing the random crop at `:159-171`); masks at reduced resolution for B8. Use `Image.LANCZOS`. | the cache is keyed by (layout, label rule, class groups), so other layouts (survivor #2) reuse it |
| `cityscapes.py:90-98` | `os.listdir` order | sorted listing | — |
| `cityscapes_eval.py:135-168`, `outdoor_eval.py:147-181` | Python loop building Gt/Pred lists; outdoor scores only cells 64–95 (`:156`, D3); per-class report commented out (`:231-236`) | `metrics.py`: vectorised TP/FP/FN per (cell, class), optional cell-subset masks, macro over object classes, "All" (D17), AP, threshold sweep, background-wins (D4), size and border breakdowns, bootstrap. **Unit test against `pred_cm`.** | any layout or class set |
| `cityscapes_eval.py:115`, `outdoor_eval.py:53` | inconsistent weight file names (D15) | one run directory per `config/seed`, holding the checkpoint, metrics JSON and git hash | — |
| `outdoor_pred.py:158`, `indoor_pred.py:102,136-138` | BGR input (D6); indoor plots ground truth, not predictions (D5) | new `viz.py` (RGB, predictions vs GT side by side) used only for figures | — |
| `outdoor_eval.py:44-50` | commented QAT stub with the `'x86'` qconfig | out of scope (no Pi); not ported | — |

### 4.2 New components (no counterpart in the repo)

- `tools/make_splits.py`: pHash grouping (Hamming ≤ 8; also a sensitivity check at 6 and 12) and the Cityscapes held-out-city `val'`. It also writes the nearest-neighbour Hamming distance from test to train for each split, which is the leakage evidence.
- `yolic_bench/seg_baseline.py`: LR-ASPP on `mobilenet_v3_small` features (torchvision `LRASPP` class with custom channels), trained on 4-group masks at 224, with a cell max-pool readout.
- `tools/cityscapes_boxes.py`: boxes from `gtFine_instanceIds` for the 8 instance classes. Crowd regions (id < 1000) become one box per connected component, flagged so they can be excluded in a sensitivity check. Output in YOLO txt format.
- `yolic_bench/rasterise.py`: detections to per-cell scores (max box score over boxes overlapping the cell by ≥1 px), shared by B10–B12.
- `tools/latency.py`: ONNX export, single-thread ORT timing protocol (§2.2), MACs and parameters via fvcore.
- External, pinned by commit hash: FastestDet and Ultralytics YOLOv8. Use their default recipes, and report their own box mAP on Cityscapes val as a sanity check that they were not under-trained (the accusation 01_brief §3.1 makes of the paper).

---

## 5. Risks

| # | Risk | Mitigation, or how it is measured |
|---|---|---|
| R1 | Head differences are within seed noise | This is what the go/no-go tests (§3.3), with the fallback in §3.4. Report seed std for every row. |
| R2 | We can't reproduce the paper's Cityscapes number | Reproduction gate (±0.03), run on the untouched script. Email the authors in W1 about D1–D3, the YOLO-to-cell mapping and the 150- vs 300-epoch weights (brief §6 Q1–Q3, Q6). |
| R3 | Outdoor filenames carry no session info, and pHash over- or under-groups | Report the test-to-train nearest-neighbour Hamming histogram for both splits, and Δ at Hamming thresholds 6/8/12. If the leakage Δ flips sign with the threshold, report it as threshold-sensitive. |
| R4 | "Matched compute" is contestable (CellPool-s16, LR-ASPP) | One MAC counter for every model; "matched" = ≤1.2× B1. Plot F1 against MACs as a Pareto chart, so models that are not matched are shown, not hidden. |
| R5 | Our detector baselines are under-trained, the same flaw we point out in the paper | Equal epoch budget, default recipes, 3 seeds, and their own box mAP reported against published Cityscapes numbers where available. |
| R6 | Box-to-cell rasterisation is biased for or against detectors | The B12 box oracle measures the ceiling under the rule. Score detectors against both mask-derived and box-derived cell labels. |
| R7 | Label-rule choices change the ranking | Main tables use the paper's ≥1 px rule; the S2 ablation (§2.5) shows whether the ranking holds. |
| R8 | The metric implementation is wrong | The unit test reproduces the old `pred_cm` + `classification_report` output exactly before any result is trusted. |
| R9 | RAM (7.6 GB) or Windows spawn crashes; laptop thermal throttling | memmap caches, `main()` guard, 3 workers, WSL2 if needed. Log img/s per epoch to catch throttling. Runs go overnight from a queue file. |
| R10 | Compute overrun | Time one cached epoch in W2 and recompute §2.6. Cut in this order: indoor → B9 → crop-to-union → outdoor B5/B6 → label-rule rule #3. |
| R11 | Novelty objection: "a re-evaluation of a ~10-citation method" (review) | Frame it as a protocol and head benchmark for the whole family of grid/cell classifiers (YOLIC, UFLD-style). Release the code and the frozen splits. Avoid a "takedown" tone. |
| R12 | The authors' follow-ups pre-empt part of contribution 2 (Selective Multi-Branch may contain a head ablation) | Read the full texts of Selective Multi-Branch and Cost-Sensitive YOLIC in W1 (review "open checks" #5). If a head ablation exists, cite it and position against it. |
| R13 | Registration and download delays (Cityscapes registration; outdoor set about 13 GB) | Register and start downloads on day 1. Arm H needs only Cityscapes; arm L can slip by one week without changing the decision date. |
| R14 | No Pi, so there is no speed claim | State it explicitly; laptop latency is labelled as a proxy. Decide by W4 whether to buy a Pi 4B; if bought, add ncnn FPS for at most 4 models, off the critical path. |
| R15 | Claims of bugs in the authors' code | Contact the authors before submission. Phrase every artefact as "the released code does X" with file:line, and quantify its effect rather than asserting the paper is wrong. |

---

## 6. Venue and timeline

### 6.1 Venue

- **Primary: IEEE Access.** It takes benchmark and empirical-protocol papers, its review is fast enough for a 3-month project, and it satisfies the IEEE-journal constraint. Two things are unverified here and should be checked on the IEEE site before W11: current review times and the article processing charge (Access is gold open access).
- **Stretch: IEEE Open Journal of Intelligent Transportation Systems**, if the GO result is strong (≥3 F1 with a clear small-cell story). The scope fit for scooter and pedestrian hazard perception needs checking.
- **Not realistic:** IEEE T-ITS, T-IV or TCSVT. These need a new method or on-device evidence, which this project doesn't have.
- **Fallback-framing venue:** IEEE Access still works. If the IEEE constraint can be dropped, TMLR or the MLRC track suits a protocol and reproducibility paper better.

### 6.2 Timeline (13 weeks; 2026-10-05 → submit by 2026-12-31, hard stop 2027-01-08)

| Week | Dates | Work | Exit criterion |
|---|---|---|---|
| W1 | Oct 5–11 | Install the environment (PyTorch, maybe WSL2). Register for Cityscapes and download it plus the outdoor set. Email the authors. Read Selective Multi-Branch and Cost-Sensitive YOLIC. Write `layout.py`, `metrics.py` and the unit tests (metric equivalence, flip permutation). | Both unit tests pass |
| W2 | Oct 12–18 | Cityscapes cache, heads, training loop; time one epoch; build pHash splits; start the go/no-go runs | §2.6 re-estimated from measured time |
| W3 | Oct 19–23 | Finish arms H and L; apply §3.3 | **Go/no-go decision, Fri Oct 23** |
| W4–W6 | Oct 26 – Nov 15 | Full Cityscapes benchmark: ladder, B2–B9, label rule, crop-to-union. Box extraction plus detectors B10–B12. Pi purchase decision (W4). | All Cityscapes rows at 3 seeds |
| W7–W8 | Nov 16–29 | Outdoor: ladder, leakage plus split-variance draws, heads on the grouped split | All outdoor rows at 3 seeds |
| W9 | Nov 30 – Dec 6 | Post-hoc work: AP, tuned thresholds, D4 rule, size and border breakdowns, bootstrap CIs, latency proxy, Pareto plot | Every table and figure generated from run directories by one script |
| W10 | Dec 7–13 | Buffer: re-runs, ambiguous-zone extra seeds, indoor stretch if slack remains | — |
| W11–W12 | Dec 14–27 | Writing (IEEE template), figures, code and splits release prep | Full draft |
| W13 | Dec 28 – Jan 8 | Advisor read-through, include any author reply, final checks, submit | Submitted |

---

## 7. Things to confirm before W1

1. Is **N = 2 GPU-days** the go/no-go budget you meant?
2. Is the IEEE requirement firm? It decides whether the fallback goes to Access or to TMLR/MLRC.
3. Is an IEEE Access APC affordable, or is there institutional coverage?
4. Will you buy a Raspberry Pi 4B? If not, the paper makes no FPS claims.
