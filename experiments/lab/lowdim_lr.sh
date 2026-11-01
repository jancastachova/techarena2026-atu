#!/usr/bin/env bash
# Small MFs are step-limited at lr 3e-3 (dim 8: 0.788 at 4500 steps, 0.802 at 9000, proj8 cosine). Higher learning rates and
# longer training, days 0-5 -> day 6, 3 seeds (26 vectors_pool.py --tune; projected cosine = full cosine for dim 8).
set -euo pipefail
for cfg in "12 0.01" "8 0.01" "12 0.02"; do set -- $cfg; echo "### dim $1 lr $2"
  .venv/bin/python experiments/26_audience/vectors_pool.py --cutoff 7 --dim $1 --lr $2 \
      --tune aux:0.3+steps:9000,aux:0.3+steps:20000 --tune-seeds 3 2>&1 | grep -E "===|Error|Traceback" || true
done | tee experiments/logs/2026-10-30_26_tune_lowdim_lr.txt
