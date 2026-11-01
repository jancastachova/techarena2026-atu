#!/usr/bin/env bash
# OVERFITTING AUDIT on day 4 (2026-10-30): day 4 has never been a test day (every screen used day 6 or the day-5 holdout), so
# it is an unselected holdout. LightGBM (lab/extra_member.py, TEST_DAY=4: fit days 2-3, day-4 halves), each set paired vs
# SET40 (the current build's features). A component whose gain does not show here was likely kept by selection on day 6.
#   SET35      the 35 set (last build whose lab gain transferred ~1:1)     -> did everything since 35 pay?
#   SET38      the full 36-group set with the 38 view                       -> was pruning to 15 groups safe?
#   SETLEAN    the lean set with the 38 view                                -> does the 40 view hold?
#   no_act     SET40 without act_b / act_c (reopened 2026-10-29)
#   lean10     SET40 without its 5 weakest groups (uoff10, deep, sqrtview, choice, cos_hop2c)
set -euo pipefail
PY=.venv/bin/python; export TEST_DAY=4; L=experiments/logs/2026-10-30_lab_audit_d4.txt
source experiments/lab/bundle_args.sh
drop() { local out=() skip=0; for x in "${SET40[@]}"; do
  if [ "$x" = "--base-extra" ]; then continue; fi
  local nm=${x%%=*}; for d in "$@"; do [ "$nm" = "$d" ] && skip=1; done
  [ $skip = 1 ] && { skip=0; continue; }; out+=(--base-extra "$x"); done; printf '%s\n' "${out[@]}"; }
NOACT=(); while IFS= read -r l; do NOACT+=("$l"); done < <(drop act_b act_c)
LEAN10=(); while IFS= read -r l; do LEAN10+=("$l"); done < <(drop uoff10 deep sqrtview choice cos_hop2c)
run() { local name=$1; shift; echo "### $name"; $PY experiments/lab/extra_member.py --seq "$SEQ" "$@" --save-preds data/cache/lab_preds_audit_${name}_d4.npz | grep "base (="; }
{
run set40 "${SET40[@]}"
run set35 "${SET35[@]}"
run set38 "${SET38[@]}"
run setlean "${SETLEAN[@]}"
run no_act "${NOACT[@]}"
run lean10 "${LEAN10[@]}"
for n in set35 set38 setlean no_act lean10; do
  echo "### paired: $n (A) vs SET40 (B)"
  $PY experiments/lab/paired_ci.py "data/cache/lab_preds_audit_${n}_d4.npz:base (= the best member)" "data/cache/lab_preds_audit_set40_d4.npz:base (= the best member)" | grep -E "plain|eval-weighted"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
