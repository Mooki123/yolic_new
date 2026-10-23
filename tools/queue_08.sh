#!/usr/bin/env bash
# research/08 go/no-go (seed 0): the MIL heads on L1, then the oracle on each target layout.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for cfg in cityscapes_mil_max_s16 cityscapes_mil_lse_s16 cityscapes_oracle_union cityscapes_oracle_shift cityscapes_oracle_fine; do run $cfg 0; done
echo "$(date '+%F %T') queue done"
