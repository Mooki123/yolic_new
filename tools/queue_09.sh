#!/usr/bin/env bash
# research/09: confirm layout transfer. Decisive runs first (T and its same-architecture oracles), seed-major,
# then YOLIC's oracles and the 224-input MIL-max for the 3-seed version of research/08.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
run cityscapes_oracle_poly 0
for seed in 0 1 2; do
  for cfg in cityscapes_mil_max_s16_r168 cityscapes_oracleT_shift cityscapes_oracleT_fine; do run $cfg $seed; done
  [ $seed = 0 ] && run cityscapes_oracleT_poly 0
done
echo "$(date '+%F %T') decisive runs done"
for seed in 1 2; do
  for cfg in cityscapes_oracle_shift cityscapes_oracle_fine cityscapes_mil_max_s16; do run $cfg $seed; done
done
echo "$(date '+%F %T') queue done"
