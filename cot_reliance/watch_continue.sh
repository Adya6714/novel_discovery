#!/bin/bash
# process each sweep model once its .pt exists; at most $P concurrent continuations
P=${P:-1}
for spec in $(awk '{print "K"$1"_L"$3"_s"$2}' ${1:-sweep_jobs_L2.txt}); do
  f=runs/sweep/$spec.pt
  until [ -f $f ]; do sleep 30; done
  while [ $(pgrep -fc "continue_until_gate.sh") -ge $P ]; do sleep 30; done
  nohup ./continue_until_gate.sh $f > /dev/null 2>&1 &
  sleep 5
done
wait
