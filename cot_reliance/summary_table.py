"""Headline table for the depth-2 sweep: per K, each model's dominant route at slots j > K (K=12: all slots), three intervention types,
and counts with exact Clopper-Pearson 95% intervals for P(model READs)."""
import json, numpy as np
from scipy.stats import beta
R = json.load(open('results/sweep_results.json'))
def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi
rows = sorted([r for k, r in R.items() if k.startswith('runs/sweep/')], key=lambda r: (r['K'], r['cfg']['seed']))
out = []
print('| K | seed | gate | FOLLOW (chain corr.) | RECOMPUTE | FOLLOW (coherent cf.) | READ-consistent (input corr.) | free-run final follows corrupted |')
print('|---|---|---|---|---|---|---|---|')
for r in rows:
    K = r['K']; reg = 'jgtK' if K < 12 else 'jltK'
    inp = r.get(f'input_{reg}_read', float('nan')) if K < 12 else float(np.mean([q['read_consistent'] for q in r['input']]))
    print(f"| {K} | {r['cfg']['seed']} | {'pass' if r['gate'] else 'FAIL'} | {r[f'single_{reg}_follow']:.3f} | {r[f'single_{reg}_recompute']:.3f} | {r[f'consistent_{reg}_follow']:.3f} | {inp:.3f} | {r[f'single_{reg}_final_follow']:.3f} |")
print()
print('| K | models READ-dominant (FOLLOW > 0.5) / n | 95% CI for P(READ) |')
print('|---|---|---|')
for K in sorted({r['K'] for r in rows}):
    rs = [r for r in rows if r['K'] == K and r['gate']]; reg = 'jgtK' if K < 12 else 'jltK'
    k = sum(r[f'single_{reg}_follow'] > 0.5 for r in rs); lo, hi = cp(k, len(rs))
    print(f'| {K} | {k}/{len(rs)} | [{lo:.2f}, {hi:.2f}] |')
