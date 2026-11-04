#!/usr/bin/env bash
# 2026-11-03 box GPU: XGBoost pairwise ranking over random 1000-event groups (lab/extra_member.py MODEL=xgb XGB_RANK=1000) on
# SET45, days 6/5/4/3 -- the GPU twin of the lambdarank member -- added to 47 at 1/8 (trees 1/8 each, choice 0.25 each).
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night11; C=data/cache/n1
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
for T in 6 5 4 3; do
  [ -f $C/d${T}_xrank.npz ] || MODEL=xgb XGB_RANK=1000 XGB_DEVICE=cuda NTHREADS=8 TEST_DAY=$T $PY experiments/lab/extra_member.py \
      --seq "$SEQ" "${SET45[@]}" --save-preds $C/d${T}_xrank.npz > $L/xrank_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47X.npz lgb=$C/d${T}_lgb.npz:0.125 xgb=$C/d${T}_xgb.npz:0.125 w2=$C/d${T}_w2.npz:0.125 \
      x=$C/d${T}_xrank.npz:0.125 a=$C/d${T}_chSP.npz:0.25 b=$C/d${T}_chD.npz:0.25 > /dev/null
  { echo "### day $T xgb rank member: $(grep 'base (=' $L/xrank_d$T.txt)"
    echo "## day $T: 47 (A) vs 47 + xgb rank at 1/8 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47X.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/xrankcmp_d$T.txt
  cat $L/xrankcmp_d$T.txt
done
echo "night13 done"
