"""Positive control for the intervention metric: hand-coded exact READ and RECOMPUTE 'models' must score
FOLLOW = 1 and RECOMPUTE = 1 respectively, for single and consistent corruption and free-running continuation."""
import torch
from train import make_task
from window_task import InterWindowTask
from evaluate import intervene, clean_acc, intervene_input

class Oracle(torch.nn.Module):
    def __init__(s, task, strat): super().__init__(); s.t = task; s.strat = strat
    def forward(s, seq, **kw):
        t = s.t; B, T = seq.shape; out = torch.full((B, T, t.vocab), -1e9)
        for q in range(0, T, 2):                 # x_j positions predict c_j
            j = q // 2 + 1; xs = seq[:, 0:q + 1:2]
            if s.strat == 'read' and j > 1:
                v = (seq[:, q - 1] - t.p) + xs[:, -1] - (xs[:, -1 - t.K] if j - t.K >= 1 else 0)
            else:
                v = xs[:, max(0, j - t.K):].sum(1)
            out[torch.arange(B), q, (v % t.p) + t.p] = 0
        return out

if __name__ == '__main__':
    for spec in ['iwin:11:2', 'iwin:11:3', 'iwin:11:6', 'iwin:11:12']:
        t = make_task(spec, seed=1)
        for strat, key in [('read', 'follow'), ('recompute', 'recompute')]:
            o = Oracle(t, strat); g = torch.Generator().manual_seed(0)
            assert min(clean_acc(o, t, 12, n=200, g=g)) == 1.0
            for j in range(1, 12):
                for kind in ['single', 'consistent']:
                    r = intervene(o, t, 12, j, n=200, g=g, kind=kind)
                    assert r[key] == 1.0, (spec, strat, j, kind, r)
                    assert r['final_' + ('follow' if strat == 'read' else 'true')] == 1.0
        if t.K >= 2:
            for strat, key in [('read', 'read_consistent'), ('recompute', 'recompute_consistent')]:
                o = Oracle(t, strat)
                for j in range(1, 12):
                    r = intervene_input(o, t, 12, j, n=200, g=torch.Generator().manual_seed(j))
                    assert r[key] == 1.0, (spec, strat, j, r)
    print('oracle metric validation OK')
