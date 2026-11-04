#!/usr/bin/env bash
# 2026-11-03 fourth box GPU: spectral LOADINGS (lab/spectral_view.py --sq --k 8: (v.e_k)^2, (u.e_k)^2) added to the spectral
# choice member SP; screened alone vs SP on days 6 and 5 (3 seeds each, paired). Exports rebuilt here (the old box is gone).
set -uo pipefail
cd /workspace/t; PY=.venv/bin/python; L=experiments/logs/2026-11-03_box2; C=data/cache/n1; mkdir -p $L $C
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
for T in 6 5; do ( EXPORT=$C/exp45_d$T.npz NTHREADS=4 TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET45[@]}" > $L/exp45_d$T.txt 2>&1 ) & done
$PY experiments/lab/spectral_view.py --sq --k 8 --out data/cache/48_specsq_c7.npz > $L/specsq.txt 2>&1
wait
for T in 6 5; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --device cuda --seeds 3 --out $C/d${T}_chSP.npz > $L/chSP_d$T.txt 2>&1
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --add data/cache/48_specsq_c7.npz --device cuda --seeds 3 \
      --out $C/d${T}_chSQ.npz > $L/chSQ_d$T.txt 2>&1
  { echo "### day $T: SP $(grep 'choice model ' $L/chSP_d$T.txt)"; echo "    SQ $(grep 'choice model ' $L/chSQ_d$T.txt)"
    TEST_DAY=$T NTHREADS=4 $PY experiments/lab/paired_ci.py "$C/d${T}_chSP.npz:base (= the best member)" "$C/d${T}_chSQ.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/sq_d$T.txt &
done
wait; cat $L/sq_d*.txt; echo "sq done"
