#!/usr/bin/env bash
# Experiment 38 screen: projected cosine view from the aux-0.3 / 4500-step vectors, on SET36 (lab/extra_member.py, half B,
# paired intervals by lab/paired_ci.py). TEST_DAY=5 in the environment runs the fresh day-5 holdout instead.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-29_38_screen_d$T.txt
source experiments/lab/bundle_args.sh
[ -f data/cache/38_cosproj8_c7.npz ] || $PY experiments/34_cosine/cos_view.py --proj 8 \
    --vectors $(for s in 0 1 2 3 4 5 6 7 8 9; do echo 38_vectors_c7_s$s; done) --out data/cache/38_cosproj8_c7.npz
NEW=data/cache/38_cosproj8_c7.npz
# SET36 with the current projected view swapped for the new one
REPL=(); for x in "${SET36[@]}"; do [ "$x" = "cosproj8=data/cache/34_cosproj8_c7.npz" ] && x="cosproj8=$NEW"; REPL+=("$x"); done
{
echo "### add: SET36 + new projected view"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${SET36[@]}" --extra proj38=$NEW --save-preds data/cache/lab_preds_38_add_d$T.npz
echo "### replace: SET36 with cosproj8 -> new"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${REPL[@]}" --save-preds data/cache/lab_preds_38_repl_d$T.npz
echo "### paired: add vs SET36"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_add_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_add_d$T.npz:+ proj38"
echo "### paired: replace vs SET36"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_add_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_repl_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
