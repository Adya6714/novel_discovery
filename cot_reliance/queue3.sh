#!/bin/bash
# Fine-resolution forced-strategy learnability (eval every 20 steps), then position-offset control and K=3 seed census.
cd "$(dirname "$0")"
{
for s in 0 1 2; do for K in 2 3 4 6; do for f in read recompute; do
  echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:$K --Lmin 12 --layers 2 --seed $s --steps 800 --log_every 20 --force $f --out runs/forced/K${K}_${f}_s${s}.pt > logs/extra/forced_K${K}_${f}_s${s}.log 2>&1"
done; done; done
for s in 0 1 2; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:3 --Lmin 12 --layers 2 --seed $s --steps 3000 --posoff 16 --out runs/posoff/K3_L2_s${s}.pt > logs/extra/posoff_K3_s${s}.log 2>&1"; done
for K in 2 4; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:$K --Lmin 12 --layers 2 --seed 0 --steps 3000 --posoff 16 --out runs/posoff/K${K}_L2_s0.pt > logs/extra/posoff_K${K}_s0.log 2>&1"; done
for s in 3 4 5 6 8; do echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:3 --Lmin 12 --layers 2 --seed $s --steps 3000 --out runs/k3seeds/K3_L2_s${s}.pt > logs/extra/k3_s${s}.log 2>&1"; done
echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:3 --Lmin 12 --layers 2 --seed 7 --steps 3000 --ckpt_every 250 --out runs/k3seeds/K3_L2_s7.pt > logs/extra/k3_s7_dyn.log 2>&1"
} | xargs -P 4 -I{} sh -c "{}"
echo queue3-done
