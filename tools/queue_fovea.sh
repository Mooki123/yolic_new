#!/usr/bin/env bash
# research/05 go/no-go: F (foveated) and R (336x168 control), 3 seeds each, one run at a time, seed-major.
# Finished runs are skipped and interrupted ones resume, so re-running is safe.
# Detached:  setsid nohup tools/queue_fovea.sh > runs/logs/queue_fovea.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for cfg in cityscapes_f_fovea cityscapes_r_336x168; do
    echo "$(date '+%F %T') start $cfg seed $seed"
    .venv/bin/python -m yolic_bench.train configs/$cfg.yaml --seed $seed >> runs/logs/${cfg}_seed$seed.log 2>&1 \
      || echo "$(date '+%F %T') FAILED $cfg seed $seed"
  done
done
echo "$(date '+%F %T') queue done"
