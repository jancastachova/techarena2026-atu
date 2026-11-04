#!/usr/bin/env bash
# Second point-estimate grid, 3 fits at once per GPU: bash run_grid2.sh <box 0-2> "<cutoffs>"
cd /workspace/t; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
BOX=$1; CUTS=$2
GA="--coupling 1 --y0 1.5"
cat > /tmp/g2list.txt <<L
c9 --cats 9 $GA
c12 --cats 12 $GA
c24 --cats 24 $GA
c32 --cats 32 $GA
c16g --cats 16 $GA
c16k3 --cats 16 --coupling 3 --y0 1.5
c24k3 --cats 24 --coupling 3 --y0 1.5
c16n --cats 16 --coupling 1 --norm-strength 30 --xprec 1 --y0 1.5
gy12 --cats 8 --coupling 1 --y0 1.2
gy2 --cats 8 --coupling 1 --y0 2
gx4 --cats 8 $GA --xprec 4
gx16 --cats 8 $GA --xprec 16
gso --cats 8 $GA --so 0.7
gso2 --cats 8 $GA --so 1.5
gsb --cats 8 $GA --sb 0.5
gsa2 --cats 8 $GA --sd-a 0.2
gsa5 --cats 8 $GA --sd-a 0.5
gk2 --cats 8 --coupling 2 --y0 1.5
gk05 --cats 8 --coupling 0.5 --y0 1.5
gk0 --cats 8 --coupling 0 --y0 1.5
gpw8 --cats 8 $GA --pw 0.8
gpw12 --cats 8 $GA --pw 1.25
K6 --cats 8 $GA --K 6
K10 --cats 8 $GA --K 10
L
i=0
while read -r n flags; do
  i=$((i+1)); [ $((i % 3)) -eq $BOX ] || continue
  for C in $CUTS; do
    O=data/cache/49/G${n}_c$C.npz; [ -f $O ] && continue
    E=99; [ $C = 3 ] && E=4
    echo "python experiments/49_bayes_choice/bayes.py --cutoff $C --end $E --log-every 59 --map 60 --hyper-start 99999 --sweeps 60 --burn 59 --thin 1 $flags --out $O > experiments/logs/49/G${n}_c$C.txt 2>&1"
  done
done < /tmp/g2list.txt > /tmp/g2cmds.txt
xargs -P 3 -I{} bash -c "{}" < /tmp/g2cmds.txt
echo grid2 box $BOX done
