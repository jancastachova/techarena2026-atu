#!/usr/bin/env bash
# 2026-11-03 box: two avenues aimed at the METRIC (global AUC: 99.997% of its pairs are cross-event), judged against 47
# (trees 1/6 each, choice SP + D 0.25 each), days 6/5/4/3, paired eval- / gap-weighted.
#  GPU: choice members SP and D with a cross-event pairwise loss (lab/choice_model.py --pair 1 / 0.3).
#  CPU: a lambdarank LightGBM member over random 1000-event groups (lab/global_rank2.py) on SET45, added to 47 at 1/8.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night11; C=data/cache/n1; mkdir -p $L
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
for T in 6 5 4 3; do
  [ -f $C/d${T}_rank_raw.npz ] || ( GR_NO_BINARY=1 GR_GROUPS=1000 GR_OUT=n1/d${T}_rank_raw.npz NTHREADS=16 TEST_DAY=$T \
      $PY experiments/lab/global_rank2.py --seq "$SEQ" "${SET45[@]}" > $L/rank_d$T.txt 2>&1 ) &
done
(
for P in 1 0.3; do for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --pair $P --device cuda --seeds 3 \
      --out $C/d${T}_chSPp$P.npz > $L/chSPp${P}_d$T.txt 2>&1
  $PY experiments/lab/choice_model.py $C/expD_d$T.npz --pair $P --device cuda --seeds 3 --out $C/d${T}_chDp$P.npz > $L/chDp${P}_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47p$P.npz lgb=$C/d${T}_lgb.npz:0.16667 xgb=$C/d${T}_xgb.npz:0.16667 \
      w2=$C/d${T}_w2.npz:0.16667 a=$C/d${T}_chSPp$P.npz:0.25 b=$C/d${T}_chDp$P.npz:0.25 > /dev/null
  { echo "### day $T pair $P: SP $(grep 'choice model ' $L/chSPp${P}_d$T.txt)"
    echo "   D $(grep 'choice model ' $L/chDp${P}_d$T.txt)"
    echo "   (no pair: SP $(grep 'choice model ' experiments/logs/2026-11-03_box_night1/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'), D $(grep 'choice model ' experiments/logs/2026-11-03_box_night1/chD_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: 47 (A) vs 47 with pair-$P choice members (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47p$P.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/pair${P}_d$T.txt
  cat $L/pair${P}_d$T.txt
done; done
) &
wait
for T in 6 5 4 3; do
  $PY -c "import numpy as np; z=np.load('$C/d${T}_rank_raw.npz'); np.savez('$C/d${T}_rank.npz', rows_B=z['rows_B'], **{'base (= the best member)|B': z['lambdarank G1000|B']})"
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47R.npz lgb=$C/d${T}_lgb.npz:0.125 xgb=$C/d${T}_xgb.npz:0.125 w2=$C/d${T}_w2.npz:0.125 \
      r=$C/d${T}_rank.npz:0.125 a=$C/d${T}_chSP.npz:0.25 b=$C/d${T}_chD.npz:0.25 > /dev/null
  { echo "### day $T rank member: $(grep 'lambdarank G1000' $L/rank_d$T.txt)"
    echo "## day $T: 47 (A) vs 47 + lambdarank at 1/8 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47R.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/rankcmp_d$T.txt
done
cat $L/rankcmp_d*.txt
echo "night11 done"
