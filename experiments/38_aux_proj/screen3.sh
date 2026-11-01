#!/usr/bin/env bash
# Experiment 38 screen 3: reopened user-activity + scoring-day features (lab/activity_view.py, caches 38_act_{a,b,c}) as a
# bundle on SET36 with the aux-0.3 projected view (the screen-1 "replace" set). Each group alone is printed too
# (--extra), the bundle is the base of the run. TEST_DAY=5 runs the fresh day-5 holdout.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-29_38_screen3_d$T.txt
source experiments/lab/bundle_args.sh
REPL=(); for x in "${SET36[@]}"; do [ "$x" = "cosproj8=data/cache/34_cosproj8_c7.npz" ] && x="cosproj8=data/cache/38_cosproj8_c7.npz"; REPL+=("$x"); done
{
echo "### replace set + activity bundle (base) and each group alone on top of the bundle"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${REPL[@]}" --base-extra act_a=data/cache/38_act_a_c7.npz \
    --base-extra act_b=data/cache/38_act_b_c7.npz --base-extra act_c=data/cache/38_act_c_c7.npz \
    --save-preds data/cache/lab_preds_38_act_d$T.npz
echo "### paired: bundle vs replace set"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_repl_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_act_d$T.npz:base (= the best member)"
echo "### paired: bundle vs SET36"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_add_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_act_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
