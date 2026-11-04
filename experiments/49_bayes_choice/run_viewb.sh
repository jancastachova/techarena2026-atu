#!/usr/bin/env bash
# All-day Bayes chains for the view, cutoffs 2-6: bash run_viewb.sh <box 0-2>
cd /workspace/t; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
BOX=$1; i=0
for C in 4 3 2 5 6; do for S in 0 1 2 3; do for F in F1 F2; do
  i=$((i+1)); [ $((i % 3)) -eq $BOX ] || continue
  [ $F = F1 ] && FL="--cats 8 --coupling 1 --norm-strength 10 --xprec 1 --y0 1.5" || FL="--cats 8 --coupling 3 --norm-strength 30 --xprec 1 --y0 2"
  echo "python experiments/49_bayes_choice/bayes.py --cutoff $C --all-day --seed $S --log-every 100 --map 60 --hyper-start 99999 --sweeps 640 --burn 60 --thin 1 $FL --out data/cache/49/V${F}_s${S}_c$C.npz > experiments/logs/49/V${F}_s${S}_c$C.txt 2>&1"
done; done; done > /tmp/vbcmds.txt
xargs -P 3 -I{} bash -c "{}" < /tmp/vbcmds.txt
echo viewb box $BOX finished
