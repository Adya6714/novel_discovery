#!/bin/bash
# Forced-strategy learnability under random absolute-position shifts (like-for-like with the position-offset free models).
cd "$(dirname "$0")"
while pgrep -f "queue3.sh|queue4.sh" > /dev/null; do sleep 30; done
{
for s in 0 1; do for K in 3 4; do for f in read recompute; do
 echo "OMP_NUM_THREADS=1 python3 train.py --group iwin:11:$K --Lmin 12 --layers 2 --seed $s --steps 800 --log_every 20 --force $f --posoff 16 --out runs/forced_posoff/K${K}_${f}_s${s}.pt > logs/extra/forcedpos_K${K}_${f}_s${s}.log 2>&1"
done; done; done
} | xargs -P 4 -I{} sh -c "{}"
echo queue5-done
