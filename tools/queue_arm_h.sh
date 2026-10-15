#!/usr/bin/env bash
# Go/no-go arm H (plan 04, section 3.2): A/B/C/D x 3 seeds, one run at a time, seed-major so a partial
# queue is already informative; then the B0 reproduction anchor (untouched released scripts, seed 1).
# A finished run is skipped by train.py, an interrupted one resumes, so re-running this script is safe.
# Detached:  setsid nohup tools/queue_arm_h.sh > runs/logs/queue_arm_h.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
mkdir -p runs/logs
for seed in 0 1 2; do
  for cfg in cityscapes_b1 cityscapes_b4_cellpool_s32 cityscapes_b8_lraspp_s cityscapes_b5_cellpool_s16; do
    echo "$(date '+%F %T') start $cfg seed $seed"
    .venv/bin/python -m yolic_bench.train configs/$cfg.yaml --seed $seed >> runs/logs/${cfg}_seed$seed.log 2>&1 \
      || echo "$(date '+%F %T') FAILED $cfg seed $seed"
  done
done

B0=runs/cityscapes_b0_released/seed1
if [ ! -f $B0/eval.log ]; then
  echo "$(date '+%F %T') start B0 train"
  (cd $B0 && MPLBACKEND=Agg ../../../.venv/bin/python cityscapes_yolic.py --seed 1 > train.log 2>&1) \
    || echo "$(date '+%F %T') FAILED B0 train"
  echo "$(date '+%F %T') start B0 eval"
  (cd $B0 && MPLBACKEND=Agg ../../../.venv/bin/python cityscapes_eval.py > eval.log 2>&1) \
    || echo "$(date '+%F %T') FAILED B0 eval"
fi
echo "$(date '+%F %T') queue done"
