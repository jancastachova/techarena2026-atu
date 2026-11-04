#!/usr/bin/env bash
# Representation screens (2026-11-01, box CPU): MF variants on top of today's best vector setting (aux 0.3, 13000 steps, cosine
# lr decay: projected cosine 0.82581 on day 6, 3 seeds), each in its own process (26 vectors_pool.py --tune, days 0-5 -> day 6).
# A variant is worth a view rebuild only at about +0.0005 on the projected cosine.
set -uo pipefail
PY=.venv/bin/python; M=/Users/ivanbibat/Documents/GitHub/techarena2026-atu/data
L=experiments/logs/2026-11-01_box_reptune; mkdir -p $L
B="aux:0.3+steps:13000+lrdecay:cos"
run() { local name=$1 tune=$2; shift 2
  $PY experiments/26_audience/vectors_pool.py --cutoff 7 --train $M/train.csv --evaluation $M/evaluation.csv --tune "$tune" \
      --tune-seeds 3 "$@" 2>&1 | grep -E "===|Error|Traceback" > $L/$name.txt; }
run base   "$B" &
run choice "$B" --loss choice &
run unif   "$B" --neg uniform &
run uw05   "$B+uw:0.5" &
run impr05 "$B+impr:0.5" &
run l2_08  "$B+l2:0.08" &
run aux05  "aux:0.5+steps:13000+lrdecay:cos" &
wait
cat $L/*.txt
