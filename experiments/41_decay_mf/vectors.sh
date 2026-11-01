#!/usr/bin/env bash
# Experiment 41: item vectors for the projected view from an MF with a COSINE learning-rate decay (13000 steps, click-log
# weight 0.3). The decay lets the long run settle instead of drifting (26 vectors_pool.py --tune, days 0-5 -> day 6, 3 seeds,
# projected cosine; logs 2026-10-30_26_tune_lrdecay.txt / _lrdecay2.txt): 40's setting (9000 steps, constant lr) 0.82523;
# cosine decay 9000 0.82530, 13000 0.82581, 18000 0.82591, 25000 0.82585 (plateau from 13000).
#     bash experiments/41_decay_mf/vectors.sh 0 9
set -euo pipefail
for s in $(seq "${1:-0}" "${2:-9}"); do
  .venv/bin/python experiments/26_audience/vectors_pool.py --cutoff 7 --aux-weight 0.3 --steps 13000 --lr-decay cos \
      --save-vectors data/cache/41_vectors_c7_s$s.npz --vec-seed $s
done
.venv/bin/python experiments/34_cosine/cos_view.py --proj 8 --vectors $(for s in $(seq "${1:-0}" "${2:-9}"); do printf "41_vectors_c7_s%s " $s; done) \
    --out data/cache/41_cosproj8_c7.npz
