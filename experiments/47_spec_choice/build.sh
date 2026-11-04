#!/usr/bin/env bash
# Experiment 47 (2026-11-03): the 45 build with its CHOICE member (weight 0.40) replaced by two choice members at 0.20 each:
#   SP  = the choice model on SET45 + the SPECTRAL taste view (lab/spectral_view.py -> 47_spec_c7.npz: for the 16 strongest
#         principal directions e_k of 40's item vectors, (u.e_k)(v.e_k) with u the unit mean taste of the known clicks and v
#         the candidate's unit vector, 10 seeds averaged). The projected cosine fixes the metric (top 8, equal weights); here
#         the network learns it.
#   D   = the choice model on D45 (SET38 with 40's projected and audience views, 337 features).
# Trees (lgb, xgb, shallow) at 1/6 each, the two choice members 0.25 each (choice total 0.5; tools/box_hedge2.sh: vs 0.4,
# eval-weighted days 3/4/5/6 +4/+5/+2/+7 x1e-5, 0.6 +4/+7/+2/+12; the even split taken, not the larger number). Lab (tools/box_night1.sh, box_night2.sh, box_hedge.sh; members re-run on
# the box, ensemble paired vs the 45 composition, eval- / gap-weighted): day 6 +0.00019 / +0.00017, day 5 +0.00013 /
# +0.00011, day 4 +0.00002 / +0.00003, day 3 (never used) +0.00005 / +0.00009. Spectral alone in the choice member: +18 / +10
# / -1 / -3 (x1e-5, days 6/5/4/3); as LightGBM features: no (day 4 -0.00037).
# Choice members built on the box (tools/box_choiceSP_build.sh, tools/box_choiceD45_build.sh, model seeds 0-5); the trees
# are 45's, re-used through 43_choice_weight/swap_choice.py on 45_aud40_s<seed>.csv.
#     bash experiments/47_spec_choice/build.sh "0 1 2 3 4 5"
set -euo pipefail
PY=.venv/bin/python; B=data/cache/box; SEEDS=${1:-0 1 2 3 4 5}; OUT=()
for S in $SEEDS; do
  $PY experiments/43_choice_weight/swap_choice.py submissions/45_aud40_s$S.csv $B/24_choice_full_45_s${S}_test.npy \
      $B/24_choice_full_sp_s${S}_test.npy submissions/47_spec_choice_s$S.csv --tree-w 0.16666666666666666 --new-w 0.25 \
      --new2 $B/24_choice_full_d45_s${S}_test.npy --new2-w 0.25
  OUT+=(submissions/47_spec_choice_s$S.csv)
done
N=$(echo $SEEDS | wc -w | tr -d ' ')
$PY experiments/15_more_seeds/average.py "${OUT[@]}" --out submissions/47s${N}_spec_choice.csv
$PY src/check_submission.py submissions/47s${N}_spec_choice.csv
