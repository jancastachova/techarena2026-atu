#!/usr/bin/env bash
# Experiment 40 screen 2: audience views rebuilt from the 9000-step vectors (26_audience/audience.py, 5 seeds; log
# 2026-10-30_40_audience.txt). Day 6 alone: soft-projected aud 0.82855 (34c 0.82726), hop4 0.82635 (34b 0.82492), hop2c
# 0.82642 (34b 0.82561); audc05 worse (0.82016 vs 0.82679), kept old. On SET40, paired vs SET40: (a) soft_aud swapped,
# (b) + cos_hop4 and cos_hop2c swapped. TEST_DAY=5 for the fresh holdout.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_40_screen2_d$T.txt
source experiments/lab/bundle_args.sh
A=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; A+=("$x"); done
B=(); for x in "${A[@]}"; do
  [ "$x" = "cos_hop4=data/cache/34b_hop4_c7.npz" ] && x="cos_hop4=data/cache/40b_hop4_c7.npz"
  [ "$x" = "cos_hop2c=data/cache/34b_hop2c_c7.npz" ] && x="cos_hop2c=data/cache/40b_hop2c_c7.npz"; B+=("$x"); done
{
echo "### (a) soft_aud from 40 vectors"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${A[@]}" --save-preds data/cache/lab_preds_40a_d$T.npz
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_40_d$T.npz:base (= the best member)" "data/cache/lab_preds_40a_d$T.npz:base (= the best member)"
echo "### (b) + hop4, hop2c from 40 vectors"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${B[@]}" --save-preds data/cache/lab_preds_40b_d$T.npz
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_40_d$T.npz:base (= the best member)" "data/cache/lab_preds_40b_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
