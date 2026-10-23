#!/usr/bin/env bash
# research/11 idea 2: session-balanced training. Stage 1: beta 0.5 and 1.0 (seed 0); pick by grouped val; stage 2: winner seeds 1-2.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for cfg in outdoor_b1_grouped_sb05 outdoor_b1_grouped_sb10; do run $cfg 0; done
WIN=$(.venv/bin/python -c "
import json
v={c: json.load(open(f'runs/{c}/seed0/metrics.json'))['best_val_primary_macro_f1'] for c in ('outdoor_b1_grouped_sb05','outdoor_b1_grouped_sb10')}
print(max(v, key=v.get))")
echo "$(date '+%F %T') stage 1 done; winner on val: $WIN"
for seed in 1 2; do run $WIN $seed; done
echo "$(date '+%F %T') queue done"
