#!/bin/bash
# Spillover test: K=3 (and K=4) supervised only on slots j >= K+2 (i.e. predictions of c_{j+1} for j > K), 3 seeds.
cd "$(dirname "$0")"
while pgrep -f queue3.sh > /dev/null; do sleep 60; done
{
for s in 0 1 2; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:3 --Lmin 12 --layers 2 --seed $s --steps 3000 --loss_from 5 --out runs/lossfrom/K3_L2_s${s}.pt > logs/extra/lossfrom_K3_s${s}.log 2>&1"; done
echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:4 --Lmin 12 --layers 2 --seed 0 --steps 3000 --loss_from 6 --out runs/lossfrom/K4_L2_s0.pt > logs/extra/lossfrom_K4_s0.log 2>&1"
} | xargs -P 4 -I{} sh -c "{}"
echo queue4-done
