#!/usr/bin/env bash
# After queue_06b: resume outdoor_b1_grouped seed 1 from its epoch-130 checkpoint (its first resume hit the RNG-state bug).
cd "$(dirname "$0")/.."
until grep -q "queue done" runs/logs/queue_06b.log 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start outdoor_b1_grouped seed 1 (resume)"
.venv/bin/python -m yolic_bench.train configs/outdoor_b1_grouped.yaml --seed 1 >> runs/logs/outdoor_b1_grouped_seed1.log 2>&1 \
  || echo "$(date '+%F %T') FAILED outdoor_b1_grouped seed 1"
echo "$(date '+%F %T') queue done"
