#!/usr/bin/env bash
# Representation screens, wave 3 (2026-11-01, box CPU; fills the remaining idle threads). Same base and judge as box_reptune.sh.
set -uo pipefail
PY=.venv/bin/python; M=/Users/ivanbibat/Documents/GitHub/techarena2026-atu/data
L=experiments/logs/2026-11-01_box_reptune; mkdir -p $L
B="aux:0.3+steps:13000+lrdecay:cos"
run() { local name=$1 tune=$2; shift 2
  $PY experiments/26_audience/vectors_pool.py --cutoff 7 --train $M/train.csv --evaluation $M/evaluation.csv --tune "$tune" \
      --tune-seeds 3 "$@" 2>&1 | grep -E "===|Error|Traceback" > $L/$name.txt; }
run negclk   "$B" --neg clicks &
run lowrank  "$B+lowrank:0.05" &
run poolidf  "$B" --pool idf &
run poolsqrt "$B" --pool sqrtset &
run lrlin    "aux:0.3+steps:13000+lrdecay" &
run aux02    "aux:0.2+steps:13000+lrdecay:cos" &
wait
