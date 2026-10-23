#!/usr/bin/env bash
# research/06 stage 1 (V2-V5, seed 0, selection on val') then the outdoor B1 grouped baseline seeds 1-2 (stage 3).
# Finished runs are skipped and interrupted ones resume, so re-running is safe.
cd "$(dirname "$0")/.."
until grep -q "^done" runs/logs/cache_views_06.log 2>/dev/null; do
  grep -q Traceback runs/logs/cache_views_06.log 2>/dev/null && { echo "$(date '+%F %T') FAILED view cache"; exit 1; }
  sleep 20
done
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for cfg in cityscapes_f_v2 cityscapes_f_v3 cityscapes_f_v4 cityscapes_f_v5; do run $cfg 0; done
echo "$(date '+%F %T') stage 1 done"
for seed in 1 2; do run outdoor_b1_grouped $seed; done
echo "$(date '+%F %T') queue done"
