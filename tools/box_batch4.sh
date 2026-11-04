#!/usr/bin/env bash
# Batch 4 (2026-11-01, vast.ai box): 30 more CORE MF seeds (30-59) with experiment 13's MF settings (26 vectors_pool.py
# defaults = 13: dim 32, l2 0.05, lr 3e-3, 3000 steps, click-log 0.1, mean pooling), 6 blocks of 5 seeds in parallel.
# RESULTS.md: more MF seeds keep helping slightly (10 -> 15 -> 30). Then MF60 vs MF30 as the core features of SET40.
# train.csv is a placeholder of the right size (the base cache 13_base_c7_9c47a117c762 is reused; key checked).
set -uo pipefail
PY=.venv/bin/python; M=/Users/ivanbibat/Documents/GitHub/techarena2026-atu/data
L=experiments/logs/2026-11-01_box_batch4; mkdir -p $L
for O in 30 35 40 45 50 55; do
  $PY experiments/26_audience/vectors_pool.py --cutoff 7 --train $M/train.csv --evaluation $M/evaluation.csv \
      --mf-seeds 5 --seed-offset $O --mf-only > $L/mf_o$O.txt 2>&1 &
done
wait
grep -h "26p_mf\|wrote\|Error\|Traceback" $L/mf_o*.txt | tail -12
