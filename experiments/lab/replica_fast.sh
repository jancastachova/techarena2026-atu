#!/usr/bin/env bash
# Fast replica round (no separate MF calibration; replica_cells prints the overall cosine AUC to match the real one):
#     bash experiments/lab/replica_fast.sh <name> <replica_sim.py args...>
set -euo pipefail
NAME=$1; shift
PY=.venv/bin/python
$PY experiments/lab/replica_sim.py "$@" --out data/cache/replica_$NAME.npz --truth-out data/cache/replica_${NAME}_truth.npz
V=()
for s in 0 1 2; do
  $PY experiments/26_audience/vectors_pool.py --stage mf --base data/cache/replica_$NAME.npz --cutoff 7 --pretrain none --init-std 0.005 \
      --pool mean --pool-gamma 0.5 --save-vectors data/cache/replica_${NAME}_vec_s$s.npz --vec-seed $s > /dev/null
  V+=(data/cache/replica_${NAME}_vec_s$s.npz)
done
$PY experiments/lab/replica_cells.py data/cache/replica_$NAME.npz "$(IFS=,; echo "${V[*]}")"
