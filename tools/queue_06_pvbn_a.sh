#!/usr/bin/env bash
# research/06 re-run with per-view BatchNorm, phase A: stage 1 (V1-V5, seed 0, val' selection), then F (= V1)
# seeds 1-2 to replace research/05. Finished runs are skipped and interrupted ones resume.
cd "$(dirname "$0")/.."
run() { echo "$(date '+%F %T') start $1 seed $2"
  .venv/bin/python -m yolic_bench.train configs/$1.yaml --seed $2 >> runs/logs/$1_seed$2.log 2>&1 || echo "$(date '+%F %T') FAILED $1 seed $2"; }
for v in f_fovea f_v2 f_v3 f_v4 f_v5; do run cityscapes_${v}_pvbn 0; done
echo "$(date '+%F %T') stage 1 done"
for seed in 1 2; do run cityscapes_f_fovea_pvbn $seed; done
echo "$(date '+%F %T') queue done"
