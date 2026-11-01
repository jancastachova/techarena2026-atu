#!/usr/bin/env bash
# Experiment 40: item vectors for the projected view from an MF trained TWICE as long (9000 steps, click-log weight 0.3). The
# projected cosine peaks near 9000 steps while the raw MF score falls (26 vectors_pool.py --tune, days 0-5 -> day 6, 3 seeds,
# log 2026-10-30_26_tune_lowdim_proj.txt): 4500 steps 0.82473, 9000 0.82523, 15000 0.82508, 25000 0.82488.
#     bash experiments/40_long_mf/vectors.sh 0 9
set -euo pipefail
for s in $(seq "${1:-0}" "${2:-9}"); do
  .venv/bin/python experiments/26_audience/vectors_pool.py --cutoff 7 --aux-weight 0.3 --steps 9000 \
      --save-vectors data/cache/40_vectors_c7_s$s.npz --vec-seed $s
done
.venv/bin/python experiments/34_cosine/cos_view.py --proj 8 --vectors $(for s in $(seq "${1:-0}" "${2:-9}"); do printf "40_vectors_c7_s%s " $s; done) \
    --out data/cache/40_cosproj8_c7.npz
