#!/usr/bin/env bash
# Choice member on the FULL 36-group feature set (2026-11-01, box): SET38 with the 40 projected view (D38, 337 features)
# instead of SET40 (169). Trees did not gain from the extra groups (lab/diversity_screen.sh D38); a neural net may. Same
# setting and seeds as box_choice10.sh (C3 x10), judged in the ensemble LightGBM 0.4 : W2 0.2 : choice 0.4 vs choice on SET40.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D38=(); for x in "${SET38[@]}"; do [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"; D38+=("$x"); done
L=experiments/logs/2026-11-01_box_choiced38; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T EXPORT=data/cache/lab_export_d38_d$T.npz $PY experiments/lab/extra_member.py --seq "$SEQ" "${D38[@]}" > $L/export_d$T.txt 2>&1 ) &
done
wait
while pgrep -f "box_choice10.sh" > /dev/null; do sleep 20; done
for T in 6 5 4; do
  $PY experiments/lab/choice_model.py data/cache/lab_export_d38_d$T.npz --seeds 10 --device cuda --hidden 1024,512 --drop 0.3 \
      --lr 5e-4 --epochs 20 --out data/cache/lab_preds_chD38x10_d$T.npz > $L/c3x10_d$T.txt 2>&1
done
for T in 6 5 4; do
  $PY /tmp/w.py $T data/cache/lab_preds_chD38x10_d$T.npz data/cache/lab_preds_ensw_D38x10_d$T.npz
  echo "### day $T: choice D38 x10 alone: $(grep 'choice model ' $L/c3x10_d$T.txt)" >> $L/compare.txt
  echo "### day $T: ensemble (choice 0.4) choice on D38 vs on SET40 (both C3 x10)" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_ensw_C3x10_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_ensw_D38x10_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
