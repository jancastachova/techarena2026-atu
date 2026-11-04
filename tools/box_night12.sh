#!/usr/bin/env bash
# 2026-11-03 box: the cross-event pairwise loss at w = 10 (dominant; w = 1 was exactly flat on day 6), choice SP + D, days 6
# then 5, judged as 47 with both members replaced. Results also kept for a combination with the lambdarank member.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night11; C=data/cache/n1
for T in 6 5; do P=10
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --pair $P --device cuda --seeds 3 \
      --out $C/d${T}_chSPp$P.npz > $L/chSPp${P}_d$T.txt 2>&1 &
  $PY experiments/lab/choice_model.py $C/expD_d$T.npz --pair $P --device cuda --seeds 3 --out $C/d${T}_chDp$P.npz > $L/chDp${P}_d$T.txt 2>&1 &
  wait
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47p$P.npz lgb=$C/d${T}_lgb.npz:0.16667 xgb=$C/d${T}_xgb.npz:0.16667 \
      w2=$C/d${T}_w2.npz:0.16667 a=$C/d${T}_chSPp$P.npz:0.25 b=$C/d${T}_chDp$P.npz:0.25 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47p${P}b.npz lgb=$C/d${T}_lgb.npz:0.16667 xgb=$C/d${T}_xgb.npz:0.16667 \
      w2=$C/d${T}_w2.npz:0.16667 a=$C/d${T}_chSP.npz:0.125 b=$C/d${T}_chD.npz:0.125 c=$C/d${T}_chSPp$P.npz:0.125 d=$C/d${T}_chDp$P.npz:0.125 > /dev/null
  { echo "### day $T pair $P: SP $(grep 'choice model ' $L/chSPp${P}_d$T.txt)"; echo "   D $(grep 'choice model ' $L/chDp${P}_d$T.txt)"
    echo "## day $T: 47 (A) vs 47 with pair-$P members (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47p$P.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
    echo "## day $T: 47 (A) vs 47 with pair-$P members ADDED, choice 4 x 0.125 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47p${P}b.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/pair${P}_d$T.txt
done
cat $L/pair10_d*.txt; echo "night12 done"
