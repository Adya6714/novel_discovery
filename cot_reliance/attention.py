"""Mechanistic corroboration via attention, on CLEAN sequences (no corruption).
At chain slot j (query = position of c_j, predicting c_{j+1}):
  READ needs inputs x_{j+1} (new) and x_{j+1-K} (leaving the window) + the current token c_j.
  RECOMPUTE needs x_{j+2-K} .. x_{j+1} (the window interior + new element), and NOT x_{j+1-K}.
We report, summed over heads and layers (max over layers), attention mass on: new element, leaving element,
window interior (positions used only by RECOMPUTE), and chain positions.  Slots j >= K (K < L) or all slots (prefix chain)."""
import sys, json, torch
from evaluate import load

@torch.no_grad()
def attn_profile(m, task, L=12, n=500, g=None):
    K = task.K
    x, c, seq, _ = task.batch(n, L, 'chain', g=g)
    ctx = seq[:, :-1]
    # recompute attention weights per layer by hooking into MultiheadAttention with need_weights
    T = ctx.shape[1]; mask = torch.triu(torch.full((T, T), float('-inf')), 1)
    h = m.tok(ctx) + m.pos(torch.arange(T))
    per_layer = []
    for b in m.blocks:
        hn = b.ln1(h); a, w = b.attn(hn, hn, hn, attn_mask=mask, need_weights=True, average_attn_weights=True)
        per_layer.append(w)  # (n, T, T) averaged over heads
        h = h + a; h = h + b.mlp(b.ln2(h))
    out = []
    slots = range(K, L) if K < L else range(1, L)
    for j in slots:  # 1-indexed slot j, query index L + j
        q = L + j
        new = j          # x_{j+1} at index j (0-indexed)
        leave = j - K if K < L else None    # x_{j+1-K} at index j-K (absent for prefix chains)
        interior = list(range(max(0, j - K + 1), j))  # x_{j+2-K}..x_j
        chain = list(range(L + 1, q + 1))
        row = dict(j=j)
        for li, w in enumerate(per_layer):
            wq = w[:, q]
            row[f'L{li}_new'] = wq[:, new].mean().item(); row[f'L{li}_leave'] = wq[:, leave].mean().item() if leave is not None else float('nan')
            row[f'L{li}_interior_per_pos'] = wq[:, interior].mean().item() if interior else float('nan')
            row[f'L{li}_chain'] = wq[:, chain].sum(1).mean().item()
        out.append(row)
    return out

if __name__ == '__main__':
    for path in sys.argv[1:]:
        m, task, c = load(path)
        prof = attn_profile(m, task, g=torch.Generator().manual_seed(7))
        keys = [k for k in prof[0] if k != 'j']
        avg = {k: sum(r[k] for r in prof) / len(prof) for k in keys}
        print(path, json.dumps({k: round(v, 3) for k, v in avg.items()}))
