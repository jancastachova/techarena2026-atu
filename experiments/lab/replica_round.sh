#!/usr/bin/env bash
# One replica round: simulate, calibrate the MF (3 seeds, days 0-5 -> day 6), save 3 vector seeds, compare cells with the real data.
#     bash experiments/lab/replica_round.sh <name> <replica_sim.py args...>
set -euo pipefail
NAME=$1; shift
PY=.venv/bin/python
$PY experiments/lab/replica_sim.py "$@" --out data/cache/replica_$NAME.npz --truth-out data/cache/replica_${NAME}_truth.npz
$PY experiments/13_retune_7days/retune.py --stage tune --base data/cache/replica_$NAME.npz --cutoff 7 --tune base --tune-seeds 3 | grep "===" || true
V=()
for s in 0 1 2; do
  $PY experiments/26_audience/vectors_pool.py --stage mf --base data/cache/replica_$NAME.npz --cutoff 7 --pretrain none --init-std 0.005 \
      --pool mean --pool-gamma 0.5 --save-vectors data/cache/replica_${NAME}_vec_s$s.npz --vec-seed $s > /dev/null
  V+=(data/cache/replica_${NAME}_vec_s$s.npz)
done
$PY experiments/lab/replica_cells.py data/cache/replica_$NAME.npz "$(IFS=,; echo "${V[*]}")"
