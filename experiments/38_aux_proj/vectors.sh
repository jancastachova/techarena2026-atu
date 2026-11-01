#!/usr/bin/env bash
# Experiment 38: item vectors for the projected cosine view from an MF tuned FOR that view. The core MF's settings
# (click-log weight 0.1, 3000 steps) were picked by its own dot-product AUC; the projected cosine (top-8 directions,
# 34_cosine/cos_view.py --proj 8) prefers more click-log weight and longer training (26 vectors_pool.py --tune, days 0-5
# -> day 6, 6 seeds, logs 2026-10-29_26_tune_proj8_*): base 0.82400, aux 0.3 + 4500 steps 0.82586 (+0.0019).
#     bash experiments/38_aux_proj/vectors.sh 0 9
set -euo pipefail
for s in $(seq "${1:-0}" "${2:-9}"); do
  .venv/bin/python experiments/26_audience/vectors_pool.py --cutoff 7 --aux-weight 0.3 --steps 4500 \
      --save-vectors data/cache/38_vectors_c7_s$s.npz --vec-seed $s
done
