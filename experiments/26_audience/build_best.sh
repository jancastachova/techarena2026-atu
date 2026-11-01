#!/usr/bin/env bash
# Rebuilds the current best submission, submissions/26_ensemble_bundle_4builds.csv (public 0.822791):
# the mean of four 24_ensemble/ensemble.py builds (LightGBM + XGBoost + MLP on the full feature set,
# rank average), at model seeds 0 and 1, each with the 30-seed core MF (13 + 15) and with the 45-seed
# core (+ seeds 30-44). Every cache below is in data/cache (git-ignored); see experiments/RESULTS.md
# for how each was made. Run from the repository root:
#     bash experiments/26_audience/build_best.sh
# Takes about 4 x 35 min. Checks the output and its rank correlation with the previous best.
set -euo pipefail
PY=.venv/bin/python
SEQ=data/cache/18_mf_c7_3a4b8237d50b.npz,data/cache/18_mf_c7_19cd2f7e5252.npz,data/cache/18_mf_c7_cd85c39e1c55.npz,data/cache/18_mf_c7_352a14674518.npz,data/cache/18_mf_c7_d0b5a6545b44.npz,data/cache/18_mf_c7_b1d4761aa9f8.npz,data/cache/18_mf_c7_9cf491798aa3.npz,data/cache/18_mf_c7_b70ecf371aee.npz,data/cache/18_mf_c7_6ba8ab937a14.npz,data/cache/18_mf_c7_61e6684ca79a.npz,data/cache/18_mf_c7_1a50e53a9712.npz,data/cache/18_mf_c7_e9863b0f4de3.npz
EXTRAS=(
  --extra idf=data/cache/25_mf_c7_8849dbfd9b9f.npz,data/cache/25_mf_c7_f7999fa81611.npz
  --extra idf15=data/cache/25_mf_c7_2fcfab9db1d6.npz
  --extra idf2=data/cache/25_mf_c7_1c3fd1d6abe6.npz
  --extra clicks=data/cache/25_mf_c7_ee7001a259a8.npz,data/cache/25_mf_c7_3e0d10f600d1.npz
  --extra impronly=data/cache/25_mf_c7_773e620088b3.npz
  --extra aud=data/cache/26s5_aud_c7.npz
  --extra audc=data/cache/26s5_audc_c7.npz
  --extra hop2c=data/cache/26v5_hop2c_c7.npz
  --extra hop3=data/cache/26h5_hop3_c7.npz
  --extra hop2=data/cache/26h10_hop2_c7.npz
  --extra idf_hop2=data/cache/26idf5_hop2_c7.npz
  --extra uoff10=data/cache/25_mf_c7_9226150de4c2.npz
  --extra dim64=data/cache/26p_mf_c7_80f68f787203.npz
  --extra choice=data/cache/25_mf_c7_3b91a877c2b2.npz
  --extra clicks_idf=data/cache/25_mf_c7_15af6d8dba56.npz
  --extra clicks_unif=data/cache/25_mf_c7_3ddfb006ec71.npz
  --extra uvec09=data/cache/20_mf_c7_c6b710c438b5.npz
  --extra deep=data/cache/21_mf_c7_c101aad57f12.npz
)
CORE45=data/cache/26p_mf_c7_5e7ac6880392.npz
for seed in 0 1; do
  $PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
      --model-seed $seed --out submissions/26_ensemble_bundle_rebuild_s$seed.csv
  $PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
      --mf-core-extra $CORE45 --model-seed $seed --out submissions/26_ensemble_bundle_mf45_rebuild_s$seed.csv
done
$PY experiments/15_more_seeds/average.py submissions/26_ensemble_bundle_rebuild_s0.csv submissions/26_ensemble_bundle_rebuild_s1.csv \
    submissions/26_ensemble_bundle_mf45_rebuild_s0.csv submissions/26_ensemble_bundle_mf45_rebuild_s1.csv \
    --out submissions/26_ensemble_bundle_4builds_rebuild.csv
$PY src/check_submission.py submissions/26_ensemble_bundle_4builds_rebuild.csv
$PY - <<'PYEOF'
import pandas as pd
from scipy.stats import spearmanr
a = pd.read_csv("submissions/26_ensemble_bundle_4builds_rebuild.csv")
b = pd.read_csv("submissions/26_ensemble_bundle_4builds.csv")
m = a.merge(b, on="rowid")
rho = spearmanr(m.iloc[:, 1], m.iloc[:, 2]).correlation
print(f"Spearman with the uploaded best: {rho:.5f}")
if rho < 0.995:
    raise SystemExit("rank correlation below 0.995: check the inputs before uploading")
PYEOF
