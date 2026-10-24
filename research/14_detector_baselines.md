# 14 — Detector baselines on Cityscapes (plan 04 B11, B12) and CPU latency

Written 2026-10-24, **before any detector run**. FastestDet (B10) dropped by the user to save time.

## Question
Does a cell classifier (YOLIC, and the best model so far F+AL, research/13) beat a modern small detector trained on
**real boxes**, scored with the same per-cell metric, and at what CPU latency?

## Data (same splits as every other Cityscapes run)
- Boxes from `gtFine_instanceIds`: People = person, rider (label ids 24-25); Vehicle = car, truck, bus, caravan,
  trailer, train, motorcycle, bicycle (26-33) — the same groups as the cell labels (v1). Instances give one box each;
  crowd regions (pixels of these classes without an instance id) give one box per connected component, kept in
  training and in the oracle because the >= 1 px cell labels include them.
- Train = the 2,406 train-city images, val = val' (bremen, cologne, krefeld; checkpoint selection), test = the
  official val (500). Images written once at 1024x512.

## Models
- **YOLOv8-N** (Ultralytics, COCO-pretrained, its default recipe and augmentation), 150 epochs (YOLIC's budget),
  best checkpoint by Ultralytics' own val fitness on val'. **imgsz 224, 3 seeds** (YOLIC's input size; letterboxed,
  so a 2:1 frame is seen at 224x112) and **imgsz 640, 1 seed** (its usual size, reference).
- **Box oracle (B12):** ground-truth boxes rasterised with the same rule — the ceiling for any box detector under the
  cell rule (boxes over-cover cells).

## Cell scoring (shared metric)
A cell's class score = the max confidence of any predicted box of that class overlapping the cell by >= 1 source pixel
(predictions at confidence >= 0.001, Ultralytics NMS defaults). Void cells masked as for every model.
Reported for every model: People+Vehicle macro-F1 **at 0.5**, and **at per-class thresholds tuned on val'** (detector
confidences are not calibrated like sigmoid cell outputs, so the tuned row is the fair cross-family comparison).
Sanity check that the detectors are not under-trained: their own box mAP50 / mAP50-95 on test.

## Comparisons (3-seed means, paired bootstrap over test images, tuned thresholds)
YOLOv8-N@224 vs A (YOLIC) and vs F+AL; YOLOv8-N@640 reported alongside (1 seed, no CI claim).
No pass/fail threshold: this is a measurement. The claims it can support, stated now:
- "F+AL beats a detector of similar compute" needs F+AL − YOLOv8-N@224 > 0 with the 95% CI excluding 0.
- "F+AL matches the detector at its usual resolution" needs |F+AL − YOLOv8-N@640| <= 1.0.

## CPU latency (laptop proxy, i7-12650H in WSL2)
ONNX Runtime, 1 thread, batch 1, 50 warm-up + 500 timed runs, median and p90, models only (detector NMS excluded and
reported separately): A, AL, F, F+AL, YOLOv8-N@224 (224x112 input), YOLOv8-N@640 (640x320). Labelled as a proxy; no
embedded-device claims.

## Results (2026-10-24 17:50)
Cityscapes test, People+Vehicle macro-F1 (tuned = per-class thresholds chosen on val', each model its own); 3 seeds
unless noted; latency = ONNX Runtime, 1 thread, batch 1, median of 500 (one run of all six models, same conditions).
| Model | @0.5 | **tuned** | far / near (tuned) | params | box mAP50 / 50-95 | CPU ms |
|---|---|---|---|---|---|---|
| A: YOLIC | 77.77 | **78.00** | 78.1 / 75.1 | 3.54M | — | 8.34 |
| AL: Align | 79.47 | **79.68** | 79.3 / 86.6 | 2.24M | — | 9.12 |
| F: foveated | 83.28 | **83.41** | 83.6 / 78.2 | 4.88M | — | 8.65 |
| **F+AL** | 86.83 | **86.85** | 86.9 / 85.8 | 2.28M | — | **9.75** |
| YOLOv8-N@224 | 68.23 | **78.25** | 78.3 / 79.4 | 3.16M | 28.6 / 14.3 | 6.65 (no NMS) |
| YOLOv8-N@640 (1 seed) | 84.42 | **87.04** | 87.3 / 84.3 | 3.16M | 56.0 / 32.4 | 44.81 (no NMS) |
| Box oracle (GT boxes) | — | **94.25** | — | — | — | — |
Claims (as stated before the runs):
- **F+AL beats YOLOv8-N at a similar input size: +8.60 [7.93, 9.36]** → supported.
- **F+AL matches YOLOv8-N at its usual 640 resolution: −0.19** (|gap| <= 1, one detector seed) → supported, at about
  4.6x lower CPU latency (9.75 vs 44.81 ms, before the detector's NMS) and 0.72x its parameters.
- YOLIC vs YOLOv8-N@224: −0.24 [−0.88, +0.41] — a tie; the paper's "YOLIC beats detectors" holds neither way here
  (the detector at 640 beats YOLIC by about 9).
Notes: at a fixed 0.5 threshold the detectors look 3-10 points worse than tuned (uncalibrated confidences), which a
fixed-threshold comparison would wrongly credit to the cell models. Latency varies about 30% between runs on this
laptop (an earlier run put A at 6.14 ms), so only within-run ratios are reported. The box oracle (94.25) shows even
perfect boxes over-cover cells; the remaining gap for F+AL is about 7 points.

### Update (2026-10-25 00:00): YOLOv8-N@640 at 3 seeds
87.04 / 86.94 / 86.80 (mean 86.92, tuned), box mAP50 / 50-95 55.8 / 32.4. **F+AL − YOLOv8-N@640 = −0.07
[−0.58, +0.42]**: statistically indistinguishable (|gap| <= 1 holds with the whole CI inside ±0.6), at about 4.6x lower
single-thread CPU latency.
