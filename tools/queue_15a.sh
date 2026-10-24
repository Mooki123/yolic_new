#!/usr/bin/env bash
# research/14 follow-up: YOLOv8-N@640 seeds 1-2 (seed 0 exists).
cd "$(dirname "$0")/.."
for seed in 1 2; do
  echo "$(date '+%F %T') start yolov8n imgsz 640 seed $seed"
  .venv/bin/python tools/yolo_baseline.py --imgsz 640 --seed $seed >> runs/logs/yolov8n_640_seed$seed.log 2>&1 || echo "$(date '+%F %T') FAILED yolov8n 640 seed $seed"
done
echo "$(date '+%F %T') queue done"
