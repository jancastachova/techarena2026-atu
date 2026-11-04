#!/usr/bin/env bash
# Core-MF seed screen (2026-11-01, vast.ai box): SET40 with the core MF features averaged over MORE seeds, paired vs the box
# SET40 (core MF30 = 13's + 15's caches, 15 seeds each) on days 6/5/4 in parallel. Caches are weighted by seed count
# (13 and 15 listed 3x each, every 5-seed block once), so the core is the plain mean over all seeds.
#     bash tools/box_mfscreen.sh LABEL LOGDIR [LOGDIR ...]      (LOGDIR: batch logs with "MF scores: 26p_mf_c7_<hash>.npz")
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=${NTHREADS:-24}
LABEL=$1; shift
NEW=$(for d in "$@"; do grep -ho "26p_mf_c7_[0-9a-f]*\.npz" $d/mf_o*.txt; done | sort -u)
N=$(echo "$NEW" | grep -c npz)
CORE="13_mf_c7_bb1fcb0a9c89.npz,13_mf_c7_bb1fcb0a9c89.npz,13_mf_c7_bb1fcb0a9c89.npz,15_mf_c7_46f30c940a6a.npz,15_mf_c7_46f30c940a6a.npz,15_mf_c7_46f30c940a6a.npz,$(echo $NEW | tr ' ' ',')"
source experiments/lab/bundle_args.sh
L=experiments/logs/2026-11-01_box_mf_$LABEL.txt
echo "$LABEL: $N new 5-seed caches -> core MF$((30 + 5 * N)); core list $CORE" > $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --mf-core "$CORE" --seq "$SEQ" "${SET40[@]}" --save-preds data/cache/lab_preds_${LABEL}_d$T.npz 2>&1 \
      | grep -E "base \(=" > experiments/logs/tmp_${LABEL}_d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T" >> $L; cat experiments/logs/tmp_${LABEL}_d$T.txt >> $L
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_boxbase_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_${LABEL}_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L
done
cat $L
