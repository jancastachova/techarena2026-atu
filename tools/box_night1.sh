#!/usr/bin/env bash
# Third vast.ai box (2026-11-03, EPYC 64 cores / RTX 4090). The 45 build's members on lab splits, days 6/5/4/3 (day 3 never
# used for selection), and the CHOICE member (weight 0.40, the strongest single member) on the full feature set D45 (SET38
# with 40's projected and audience views, 337 features) vs SET45 (169) -- the small D38 edge of 2026-11-01, now inside the 45
# config. Choice variants on the GPU: S = SET45 default x3 (as the build), D = D45 default x3, D3 = D45 1024,512 drop 0.3 x5.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=8
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
D45=(); for x in "${SET38[@]}"; do
  [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"
  [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; D45+=("$x"); done
L=experiments/logs/2026-11-03_box_night1; mkdir -p $L; C=data/cache/n1
mkdir -p $C
for T in 6 5 4 3; do
  X() { TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "$@"; }
  [ -f $C/d${T}_lgb.npz ] || ( X "${SET45[@]}" --save-preds $C/d${T}_lgb.npz 2>&1 | grep "base (=" > $L/lgb_d$T.txt ) &
  [ -f $C/d${T}_w2.npz ] || ( LGBP='{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.3}' X "${SET45[@]}" --save-preds $C/d${T}_w2.npz 2>&1 | grep "base (=" > $L/w2_d$T.txt ) &
  grep -q exported $L/exp45_d$T.txt 2>/dev/null || ( EXPORT=$C/exp45_d$T.npz X "${SET45[@]}" > $L/exp45_d$T.txt 2>&1 ) &
  grep -q exported $L/expD_d$T.txt 2>/dev/null || ( EXPORT=$C/expD_d$T.npz X "${D45[@]}" > $L/expD_d$T.txt 2>&1 ) &
done
# GPU queue: XGBoost members, then choice variants as their exports appear
(
for T in 6 5 4 3; do
  [ -f $C/d${T}_xgb.npz ] || MODEL=xgb XGB_DEVICE=cuda TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET45[@]}" --save-preds $C/d${T}_xgb.npz 2>&1 | grep -E "base \(=|Error" > $L/xgb_d$T.txt
done
for T in 6 5 4 3; do
  until grep -q exported $L/exp45_d$T.txt 2>/dev/null && grep -q exported $L/expD_d$T.txt 2>/dev/null; do sleep 30; done
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --device cuda --seeds 3 --out $C/d${T}_chS.npz > $L/chS_d$T.txt 2>&1
  $PY experiments/lab/choice_model.py $C/expD_d$T.npz --device cuda --seeds 3 --out $C/d${T}_chD.npz > $L/chD_d$T.txt 2>&1
  $PY experiments/lab/choice_model.py $C/expD_d$T.npz --device cuda --seeds 5 --hidden 1024,512 --drop 0.3 --lr 5e-4 --epochs 20 \
      --out $C/d${T}_chD3.npz > $L/chD3_d$T.txt 2>&1
done
) &
wait
for T in 6 5 4 3; do
  for v in S D D3; do
    $PY experiments/lab/ens_mix.py $C/d${T}_ens$v.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 ch=$C/d${T}_ch$v.npz:0.4 > /dev/null
  done
  { echo "### day $T members: lgb $(cat $L/lgb_d$T.txt)"; echo "  w2 $(cat $L/w2_d$T.txt)"; echo "  xgb $(cat $L/xgb_d$T.txt)"
    for v in S D D3; do echo "  choice $v: $(grep 'choice model ' $L/ch${v}_d$T.txt)"; done
    for v in D D3; do echo "## day $T ensemble: choice S (A) vs choice $v (B)"
      TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensS.npz:base (= the best member)" "$C/d${T}_ens$v.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted"; done
  } >> $L/compare.txt
done
cat $L/compare.txt
