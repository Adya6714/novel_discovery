#!/bin/bash
# For one model path: while clean teacher-forced accuracy (L=12, every slot) < 0.98 and total steps < CAP,
# continue training for BLOCK more steps from the latest checkpoint. Writes logs/continue.log lines.
path=$1; CAP=${CAP:-18000}; BLOCK=${BLOCK:-3000}
while true; do
  read gate steps <<< $(OMP_NUM_THREADS=1 python3 -c "
import torch; from evaluate import load, clean_acc
m,t,c=load('$path'); a=clean_acc(m,t,12,'chain',n=2000,g=torch.Generator().manual_seed(5)); print(int(min(a)>=0.98), c['steps'])")
  echo "$(date +%T) $path gate=$gate steps=$steps" >> logs/continue.log
  if [ "$gate" = "1" ] || [ "$steps" -ge "$CAP" ]; then break; fi
  spec=$(basename $path .pt); K=$(echo $spec | sed 's/K\([0-9]*\)_.*/\1/'); Ly=$(echo $spec | sed 's/.*_L\([0-9]*\)_.*/\1/'); s=$(echo $spec | sed 's/.*_s\([0-9]*\)/\1/')
  OMP_NUM_THREADS=1 python3 train.py --group iwin:11:$K --Lmin 12 --layers $Ly --seed $s --steps $BLOCK --init $path --out $path >> logs/sweep/${spec}.log 2>&1
done
