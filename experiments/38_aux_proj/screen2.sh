#!/usr/bin/env bash
# Experiment 38 screen 2: SET36 with cosproj8 AND the four user-side cosine audience views (hop2, hop2c, hop3, hop4) swapped
# for their versions from the aux-0.3 / 4500-step vectors (38b_*; the item-side audience views and the plain cosine were
# worse alone, so they stay). TEST_DAY=5 runs the fresh day-5 holdout.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-29_38_screen2_d$T.txt
source experiments/lab/bundle_args.sh
REPL=()
for x in "${SET36[@]}"; do
  case "$x" in
    cosproj8=data/cache/34_cosproj8_c7.npz) x="cosproj8=data/cache/38_cosproj8_c7.npz" ;;
    cos_hop2=data/cache/34b_hop2_c7.npz|cos_hop2c=data/cache/34b_hop2c_c7.npz|cos_hop3=data/cache/34b_hop3_c7.npz|cos_hop4=data/cache/34b_hop4_c7.npz)
      x="${x/34b_/38b_}" ;;
  esac
  REPL+=("$x")
done
printf '%s\n' "${REPL[@]}" | grep -c "38" >&2
{
echo "### replace: cosproj8 + hop2/hop2c/hop3/hop4 -> 38"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${REPL[@]}" --save-preds data/cache/lab_preds_38_repl5_d$T.npz
echo "### paired: replace5 vs SET36"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_add_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_repl5_d$T.npz:base (= the best member)"
echo "### paired: replace5 vs replace cosproj8 only"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_repl_d$T.npz:base (= the best member)" "data/cache/lab_preds_38_repl5_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
