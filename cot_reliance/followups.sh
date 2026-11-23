#!/bin/bash
# Runs after the main sweep: depth-4 arm, p=17 arm, training-dynamics run. Gate-and-continue applied afterwards.
cd "$(dirname "$0")"
until [ $(ls runs/sweep/K*_L2_s*.pt 2>/dev/null | grep -v _it | wc -l) -ge 24 ] && ! pgrep -f "xargs -P 4 -L 1 sh -c OMP" > /dev/null; do sleep 60; done
{
for s in 0 1; do for K in 2 3 4 5; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:$K --Lmin 12 --layers 4 --seed $s --steps 3000 --out runs/depth4/K${K}_L4_s${s}.pt > logs/extra/K${K}_L4_s${s}.log 2>&1"; done; done
for s in 0 1; do for K in 2 3 4; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:17:$K --Lmin 12 --layers 2 --seed $s --steps 3000 --out runs/p17/K${K}_L2_s${s}.pt > logs/extra/p17_K${K}_L2_s${s}.log 2>&1"; done; done
echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:3 --Lmin 12 --layers 2 --seed 7 --steps 3000 --ckpt_every 250 --out runs/dyn/K3_L2_s7.pt > logs/extra/dyn_K3_s7.log 2>&1"
} | xargs -P 4 -I{} sh -c "{}"
echo followups-done
