#!/usr/bin/env bash
# First job on a new machine (2026-11-01, vast.ai): the SET40 lab model on day 6 must reproduce the Mac's result
# (half-B 0.84505, eval-weighted 0.82839, gap-weighted 0.82626; data/cache/lab_preds_40_d6.npz) before any new result is
# trusted. Prints the run, the wall time and the paired comparison with the Mac predictions (B-A should be ~0).
set -euo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-01_box_verify.txt
source experiments/lab/bundle_args.sh
{
t0=$(date +%s)
$PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds data/cache/lab_preds_box40_d6.npz | grep -E "base \(="
echo "wall time $(( $(date +%s) - t0 )) s"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_40_d6.npz:base (= the best member)" "data/cache/lab_preds_box40_d6.npz:base (= the best member)" --n 50 | grep -E "plain|weighted"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
