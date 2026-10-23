#!/usr/bin/env bash
# research/06 per-view-BN re-run, phase B (after phase A): stage-1 winner V4 seeds 1-2 on Cityscapes, then the
# context rule with V4's split on outdoor (grouped split), 3 seeds.
cd "$(dirname "$0")/.."
until grep -q "queue done" runs/logs/queue_06_pvbn_a.log 2>/dev/null; do sleep 30; done
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for seed in 1 2; do run cityscapes_f_v4_pvbn $seed; done
echo "$(date '+%F %T') stage 2 done"
for seed in 0 1 2; do run outdoor_f_ctx_grouped_pvbn $seed; done
echo "$(date '+%F %T') queue done"
