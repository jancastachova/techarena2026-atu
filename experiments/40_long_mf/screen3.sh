#!/usr/bin/env bash
# Experiment 40 screen 3: the residual-direction view (cosresid8, built in 36 from the old 22/26 vectors) rebuilt from the
# 9000-step vectors, swapped into SET40. Screen on day 6 (paired vs SET40); confirm on day 4 only if it gains.
set -euo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-10-30_40_screen3.txt
source experiments/lab/bundle_args.sh
[ -f data/cache/40_cosresid8_c7.npz ] || $PY experiments/34_cosine/cos_view.py --proj 8 --resid \
    --vectors $(for s in 0 1 2 3 4 5 6 7 8 9; do echo 40_vectors_c7_s$s; done) --out data/cache/40_cosresid8_c7.npz
NEW=(); for x in "${SET40[@]}"; do [ "$x" = "cosresid8=data/cache/36_cosresid8_c7.npz" ] && x="cosresid8=data/cache/40_cosresid8_c7.npz"; NEW+=("$x"); done
{
echo "### day 6: cosresid8 from 40 vectors"
TEST_DAY=6 $PY experiments/lab/extra_member.py --seq "$SEQ" "${NEW[@]}" --save-preds data/cache/lab_preds_40r_d6.npz | grep "base (="
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_40_d6.npz:base (= the best member)" "data/cache/lab_preds_40r_d6.npz:base (= the best member)" | grep -E "plain|eval-weighted"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
