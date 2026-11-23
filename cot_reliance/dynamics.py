"""Training dynamics of route selection: per-slot census (READ R / RECOMPUTE C / mixed m) and clean accuracy at each saved checkpoint."""
import glob, re, json, torch
from evaluate import load, clean_acc, intervene
def label(r): return 'R' if r['follow'] >= 0.9 else ('C' if r['recompute'] >= 0.9 else 'm')
rows = []
paths = sorted(glob.glob('runs/k3seeds/K3_L2_s7_it*.pt'), key=lambda p: int(re.search(r'_it(\d+)', p)[1])) + ['runs/k3seeds/K3_L2_s7.pt']
for p in paths:
    m, t, c = load(p); g = torch.Generator().manual_seed(11)
    acc = clean_acc(m, t, 12, n=600, g=g)
    rs = [intervene(m, t, 12, j, n=600, g=g, freerun=False) for j in range(1, 12)]
    labs = ''.join(label(r) for r in rs)
    rows.append(dict(step=c['steps'], minacc=min(acc), labels=labs, follow=[r['follow'] for r in rs], recompute=[r['recompute'] for r in rs]))
    print(f"step {c['steps']:5d} minacc {min(acc):.3f} slots1..11 {labs}", flush=True)
json.dump(rows, open('results/dynamics_K3_s7.json', 'w'), indent=1)
