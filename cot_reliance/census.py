"""Per-slot strategy census (response to review objections 2.1 / 4.1).
For each model: per slot j = 1..11, label = READ if single-corruption FOLLOW >= 0.9, RECOMPUTE if RECOMPUTE >= 0.9, else MIXED.
Position-offset models (trained with random absolute-position shifts 0..16) are evaluated at offsets 0, 8, 16."""
import sys, glob, json, torch
from evaluate import load, clean_acc, intervene

class Shift(torch.nn.Module):
    def __init__(s, m, o): super().__init__(); s.m, s.o = m, o
    def forward(s, x, **kw): return s.m(x, pos_offset=s.o, **kw)

def label(r):
    return 'R' if r['follow'] >= 0.9 else ('C' if r['recompute'] >= 0.9 else 'm')   # R = reads chain, C = recomputes, m = mixed

out = {}
for pattern in sys.argv[1:]:
    for p in sorted(glob.glob(pattern)):
        if '_it' in p: continue
        m, t, c = load(p); offs = [0, 8, 16] if c.get('posoff') else [0]
        for o in offs:
            mm = Shift(m, o) if o else m; g = torch.Generator().manual_seed(11)
            acc = min(clean_acc(mm, t, 12, n=1000, g=g))
            rows = [intervene(mm, t, 12, j, n=1000, g=g, freerun=False) for j in range(1, 12)]
            labs = ''.join(label(r) for r in rows)
            key = f'{p}@off{o}'
            out[key] = dict(K=t.K, posoff=c.get('posoff', 0), offset=o, seed=c['seed'], minacc=acc, labels=labs,
                            follow=[r['follow'] for r in rows], recompute=[r['recompute'] for r in rows])
            print(f'{key:45s} K={t.K} minacc={acc:.3f} slots1..11: {labs}', flush=True)
json.dump(out, open('results/census.json', 'w'), indent=1)
