#!/usr/bin/env bash
# 2026-11-03 box: CHOICE member on SET45 + the SPECTRAL taste view (lab/spectral_view.py -> 47_spec_c7.npz, 16 per-direction
# products u_k v_k of 40's vectors) -- a learned taste metric instead of the fixed projected cosine. Same settings as the build
# (default, 3 seeds); judged inside the 45 ensemble (lgb 0.2 / xgb 0.2 / w2 0.2 / choice 0.4) vs choice S, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --device cuda --seeds 3 \
      --out $C/d${T}_chSP.npz > $L/chSP_d$T.txt 2>&1
done
for T in 6 5 4 3; do
  until [ -f $C/d${T}_chS.npz ] && [ -f $C/d${T}_xgb.npz ] && [ -f $C/d${T}_lgb.npz ] && [ -f $C/d${T}_w2.npz ]; do sleep 30; done
  $PY experiments/lab/ens_mix.py $C/d${T}_ensS.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 ch=$C/d${T}_chS.npz:0.4 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensSP.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 ch=$C/d${T}_chSP.npz:0.4 > /dev/null
  { echo "### day $T choice S: $(grep 'choice model ' $L/chS_d$T.txt)"; echo "    choice SP: $(grep 'choice model ' $L/chSP_d$T.txt)"
    echo "## day $T ensemble: choice S (A) vs choice + spectral (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensS.npz:base (= the best member)" "$C/d${T}_ensSP.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted"
  } >> $L/compare_spec.txt
done
cat $L/compare_spec.txt
