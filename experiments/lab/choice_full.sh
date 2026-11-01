#!/usr/bin/env bash
# Choice model on the FULL SET38 (337 columns) vs the lean set (169), 3 seeds, day 6; paired vs the lean LightGBM and vs the
# lean choice model.
set -euo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-10-30_lab_choice_full_d6.txt
source experiments/lab/bundle_args.sh
{
EXPORT=data/cache/lab_export_set38_d6.npz $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET38[@]}" | tail -1
$PY experiments/lab/choice_model.py data/cache/lab_export_set38_d6.npz --seeds 3 --out data/cache/lab_preds_choice_full_s3_d6.npz \
    --blend data/cache/lab_preds_38_act_d6.npz
rm -f data/cache/lab_export_set38_d6.npz
echo "### paired: full-set choice vs lean choice (3 seeds each)"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_choice_s3_d6.npz:base (= the best member)" "data/cache/lab_preds_choice_full_s3_d6.npz:base (= the best member)"
echo "### paired: lean choice (3 seeds) vs lean LightGBM"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_lean0_d6.npz:base (= the best member)" "data/cache/lab_preds_choice_s3_d6.npz:base (= the best member)"
} 2>&1 | grep -v "epoch" | tee $L
