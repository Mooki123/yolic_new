#!/usr/bin/env bash
# research/15: Align ablations on F+AL, 4 variants x 3 seeds, after the YOLOv8-N@640 seeds (queue_15a).
cd "$(dirname "$0")/.."
until grep -q "queue done" runs/logs/queue_15a.log 2>/dev/null; do sleep 60; done
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for seed in 0 1 2; do
  for v in no_lse no_mean no_lateral no_cellbias; do run cityscapes_f_align_$v $seed; done
done
echo "$(date '+%F %T') queue done"
