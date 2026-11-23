#!/bin/bash
# Usage: run_search.sh n V [parallel]
# Splits the search by the smallest speed v1 into chunks of roughly equal work (#tuples = C(V-v1, n-1)) and runs them in parallel.
n=$1; V=$2; P=${3:-4}
mkdir -p data/chunks_n${n}_V${V}
python3 - "$n" "$V" > data/chunks_n${n}_V${V}/jobs.txt <<'PY'
import sys; from math import comb
n, V = map(int, sys.argv[1:]); tot = comb(V, n); target = tot / 64
lo = 1; acc = 0
for x in range(1, V - n + 2):
    acc += comb(V - x, n - 1)
    if acc >= target or x == V - n + 1:
        print(lo, x); lo = x + 1; acc = 0
PY
cat data/chunks_n${n}_V${V}/jobs.txt | xargs -P $P -L 1 sh -c './neartight '"$n $V"' $0 $1 > data/chunks_n'"${n}_V${V}"'/out_$0_$1.txt 2> data/chunks_n'"${n}_V${V}"'/log_$0_$1.txt'
cat data/chunks_n${n}_V${V}/out_*.txt | sort -n -k1,1 -k2,2 -k3,3 -k4,4 -k5,5 -k6,6 -k7,7 -k8,8 > data/nt_n${n}_V${V}.txt
python3 -c "
import glob,re;L=0;F=0
for f in glob.glob('data/chunks_n${n}_V${V}/log_*.txt'):
    m=re.search(r'leaves=(\d+) primitive_near_tight=(\d+)',open(f).read()); L+=int(m[1]); F+=int(m[2])
from math import comb; print('n=${n} V=${V} leaves',L,'expected',comb(${V},${n}),'found',F)"
