#!/usr/bin/env bash
# research/14: YOLOv8-N on Cityscapes boxes (224 x 3 seeds, 640 x 1 seed), then CPU latency for every model.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start yolov8n imgsz $1 seed $2"
  .venv/bin/python tools/yolo_baseline.py --imgsz $1 --seed $2 >> runs/logs/yolov8n_$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED yolov8n $1 seed $2"; }
for seed in 0 1 2; do run 224 $seed; done
run 640 0
echo "$(date '+%F %T') start latency"
.venv/bin/python tools/latency.py >> runs/logs/latency.log 2>&1 || echo "$(date '+%F %T') FAILED latency"
echo "$(date '+%F %T') queue done"
