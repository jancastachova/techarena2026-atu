#!/usr/bin/env bash
# More point-estimate views, fit days 2-6 and evaluation: bash run_view3.sh <box 0-2>
cd /workspace/t; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
BOX=$1; i=0
for V in c12 c9 gsa2 n1; do
  case $V in c12) FL="--cats 12";; c9) FL="--cats 9";; gsa2) FL="--cats 8 --sd-a 0.2";; n1) FL="--cats 8 --norm-strength 10 --xprec 1";; esac
  for C in 6 5 4 3 2 7; do
    i=$((i+1)); [ $((i % 3)) -eq $BOX ] || continue
    A="--all-day"; [ $C = 7 ] && A=""
    echo "python experiments/49_bayes_choice/bayes.py --cutoff $C $A --log-every 59 --map 60 --hyper-start 99999 --sweeps 60 --burn 59 --thin 1 --coupling 1 --y0 1.5 $FL --out data/cache/49/W${V}_c$C.npz > experiments/logs/49/W${V}_c$C.txt 2>&1"
  done
done > /tmp/wcmds.txt
xargs -P 3 -I{} bash -c "{}" < /tmp/wcmds.txt
echo view3 box $BOX finished $(ls data/cache/49 | grep -c "^W")
