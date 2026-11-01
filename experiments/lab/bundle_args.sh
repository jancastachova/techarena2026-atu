# Shared arguments for lab checks on the 26_ensemble_bundle feature set (source this file).
SEQ=data/cache/18_mf_c7_3a4b8237d50b.npz,data/cache/18_mf_c7_19cd2f7e5252.npz,data/cache/18_mf_c7_cd85c39e1c55.npz,data/cache/18_mf_c7_352a14674518.npz,data/cache/18_mf_c7_d0b5a6545b44.npz,data/cache/18_mf_c7_b1d4761aa9f8.npz,data/cache/18_mf_c7_9cf491798aa3.npz,data/cache/18_mf_c7_b70ecf371aee.npz,data/cache/18_mf_c7_6ba8ab937a14.npz,data/cache/18_mf_c7_61e6684ca79a.npz,data/cache/18_mf_c7_1a50e53a9712.npz,data/cache/18_mf_c7_e9863b0f4de3.npz
BUNDLE=(
  --base-extra idf=data/cache/25_mf_c7_8849dbfd9b9f.npz,data/cache/25_mf_c7_f7999fa81611.npz
  --base-extra idf15=data/cache/25_mf_c7_2fcfab9db1d6.npz
  --base-extra idf2=data/cache/25_mf_c7_1c3fd1d6abe6.npz
  --base-extra clicks=data/cache/25_mf_c7_ee7001a259a8.npz,data/cache/25_mf_c7_3e0d10f600d1.npz
  --base-extra impronly=data/cache/25_mf_c7_773e620088b3.npz
  --base-extra aud=data/cache/26s5_aud_c7.npz
  --base-extra audc=data/cache/26s5_audc_c7.npz
  --base-extra hop2c=data/cache/26v5_hop2c_c7.npz
  --base-extra hop3=data/cache/26h5_hop3_c7.npz
  --base-extra hop2=data/cache/26h10_hop2_c7.npz
  --base-extra idf_hop2=data/cache/26idf5_hop2_c7.npz
  --base-extra uoff10=data/cache/25_mf_c7_9226150de4c2.npz
  --base-extra dim64=data/cache/26p_mf_c7_80f68f787203.npz
  --base-extra choice=data/cache/25_mf_c7_3b91a877c2b2.npz
  --base-extra clicks_idf=data/cache/25_mf_c7_15af6d8dba56.npz
  --base-extra clicks_unif=data/cache/25_mf_c7_3ddfb006ec71.npz
  --base-extra uvec09=data/cache/20_mf_c7_c6b710c438b5.npz
  --base-extra deep=data/cache/21_mf_c7_c101aad57f12.npz
)
# experiment 34 (public 0.82389): bundle + sqrt view + cosine view + eight cosine audience views
SET34=("${BUNDLE[@]}" --base-extra sqrtview=data/cache/33_sqrtpool_c7.npz --base-extra cos=data/cache/34_cos_c7.npz)
for k_ in aud audc audc05 wb hop2 hop2c hop3 hop4; do SET34+=(--base-extra cos_$k_=data/cache/34a_${k_}_c7.npz); done
# experiment 35 candidate: bundle + sqrt view + cosine + projected cosine (k=8) + unit-clicker cosine audience (34b)
SET35=("${BUNDLE[@]}" --base-extra sqrtview=data/cache/33_sqrtpool_c7.npz --base-extra cos=data/cache/34_cos_c7.npz --base-extra cosproj8=data/cache/34_cosproj8_c7.npz)
for k_ in aud audc audc05 wb hop2 hop2c hop3 hop4; do SET35+=(--base-extra cos_$k_=data/cache/34b_${k_}_c7.npz); done
# experiment 36 candidate: SET35 + temperature 0.6 projected view + residual-direction cosine + soft projection (K=20) + soft
# audience (34c aud); lab/extra_member.py half B +0.00007 [-0.00007, +0.00023], eval-weighted +0.00035 [+0.00005, +0.00069]
SET36=("${SET35[@]}" --base-extra temp06=data/cache/34_cosproj8_t0.6_c7.npz --base-extra cosresid8=data/cache/36_cosresid8_c7.npz
       --base-extra cossoft20=data/cache/34_cossoft20_c7.npz --base-extra soft_aud=data/cache/34c_aud_c7.npz)
# experiment 38: SET36 with the projected view from the MF tuned for it (click-log weight 0.3, 4500 steps; 38_aux_proj) + the
# reopened user-activity / scoring-day bundle (lab/activity_view.py). Paired vs SET36, plain / eval-weighted: day 6 +0.00037 /
# +0.00041, day-5 holdout +0.00045 / +0.00050 (all intervals exclude 0).
SET38=()
for x in "${SET36[@]}"; do [ "$x" = "cosproj8=data/cache/34_cosproj8_c7.npz" ] && x="cosproj8=data/cache/38_cosproj8_c7.npz"; SET38+=("$x"); done
SET38+=(--base-extra act_a=data/cache/38_act_a_c7.npz --base-extra act_b=data/cache/38_act_b_c7.npz --base-extra act_c=data/cache/38_act_c_c7.npz)
# LEAN set (2026-10-30, lab/lean_screen.sh): the 15 view groups that carry all of SET38 (top 15 by split gain on day 6, 14 shared
# with day 5); eval-weighted day 6 / day 5 0.82779 / 0.82100 vs SET38 0.82794 / 0.82087 with 169 instead of 337 columns.
SETLEAN=()
for x in "${SET38[@]}"; do
  case "${x%%=*}" in soft_aud|cosproj8|cos_hop4|clicks|cos_audc05|uvec09|clicks_unif|act_b|act_c|cosresid8|choice|cos_hop2c|uoff10|deep|sqrtview) SETLEAN+=(--base-extra "$x");; esac
done
# experiment 40: SETLEAN with the projected view rebuilt from 9000-step vectors (40_long_mf/vectors.sh); paired vs SETLEAN,
# plain / eval-weighted: day 6 +0.00057 / +0.00060, day 5 +0.00053 / +0.00061 (logs 2026-10-30_40_screen_d6/d5.txt)
SET40=(); for x in "${SETLEAN[@]}"; do [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"; SET40+=("$x"); done
# LEAN8 (2026-10-30 overfitting audit, lab/audit_d4.sh on the never-used day 4): SET40 without act_b / act_c (day 4 +0.00002)
# and without its 5 weakest groups uoff10, deep, sqrtview, choice, cos_hop2c (day 4 -0.00008 with them): 8 view groups.
SET40L=(); _skip=0
for x in "${SET40[@]}"; do
  [ "$x" = "--base-extra" ] && continue
  case "${x%%=*}" in act_b|act_c|uoff10|deep|sqrtview|choice|cos_hop2c) continue;; esac
  SET40L+=(--base-extra "$x")
done
