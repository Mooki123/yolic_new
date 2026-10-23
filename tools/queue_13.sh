#!/usr/bin/env bash
# research/13: Align head x foveated input on Cityscapes, 3 seeds, seed-major.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for seed in 0 1 2; do for cfg in cityscapes_align cityscapes_f_align; do run $cfg $seed; done; done
echo "$(date '+%F %T') queue done"
