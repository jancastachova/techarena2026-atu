#!/usr/bin/env bash
# Representation screens, wave 2 (2026-11-01, box CPU; fills the idle threads next to box_reptune.sh). Same base and judge.
set -uo pipefail
PY=.venv/bin/python; M=/Users/ivanbibat/Documents/GitHub/techarena2026-atu/data
L=experiments/logs/2026-11-01_box_reptune; mkdir -p $L
B="aux:0.3+steps:13000+lrdecay:cos"
run() { local name=$1 tune=$2; shift 2
  $PY experiments/26_audience/vectors_pool.py --cutoff 7 --train $M/train.csv --evaluation $M/evaluation.csv --tune "$tune" \
      --tune-seeds 3 "$@" 2>&1 | grep -E "===|Error|Traceback" > $L/$name.txt; }
run d48_l2   "$B+l2:0.08" --dim 48 &
run std05    "$B+std:0.05" &
run uoff     "$B+uoff:1.0" &
run boot     "$B+boot" &
run logq     "$B+logq:1" &
run hdrop2   "$B+hdrop:0.2" &
run s18_l2   "aux:0.3+steps:18000+lrdecay:cos+l2:0.08" &
run impr2    "$B+impr:2.0" &
wait
