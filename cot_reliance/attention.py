"""Mechanistic corroboration via attention on CLEAN sequences (no corruption); layout-agnostic.
Query = the position whose logits predict c_{j+1}. Operands:
  new      x_{j+1}               (used by both strategies)
  leave    x_{j+1-K}             (used only by READ; exists for j >= K, K < L)
  interior x_{j+2-K}..x_j        (used only by RECOMPUTE)
  chain    written c_1..c_j      (READ uses c_j)
Reports head-averaged attention per layer, averaged over slots j >= K (or all slots for prefix chains)."""
import sys, json, torch
from evaluate import load

@torch.no_grad()
def attn_profile(m, task, L=12, n=500, g=None):
    K = task.K
    x, c, seq, _ = task.batch(n, L, 'chain', g=g)
    ctx = seq[:, :-1]; T = ctx.shape[1]
    mask = torch.triu(torch.full((T, T), float('-inf')), 1)
    h = m.tok(ctx) + m.pos(torch.arange(T)); per_layer = []
    for b in m.blocks:
        hn = b.ln1(h); a, w = b.attn(hn, hn, hn, attn_mask=mask, need_weights=True, average_attn_weights=True)
        per_layer.append(w); h = h + a; h = h + b.mlp(b.ln2(h))
    out = []
    slots = range(K, L) if K < L else range(1, L)
    for j in slots:
        q = task.pred_index(j + 1, L)
        new = task.x_index(j + 1, L)
        leave = task.x_index(j + 1 - K, L) if K < L else None
        interior = [task.x_index(i, L) for i in range(max(1, j + 2 - K), j + 1)]
        prevc = task.slot_index(j, L)
        chain = [task.slot_index(i, L) for i in range(1, j + 1)]
        row = dict(j=j)
        for li, w in enumerate(per_layer):
            wq = w[:, q]
            row[f'L{li}_new'] = wq[:, new].mean().item()
            row[f'L{li}_leave'] = wq[:, leave].mean().item() if leave is not None else float('nan')
            row[f'L{li}_interior_per_pos'] = wq[:, interior].mean().item() if interior else float('nan')
            row[f'L{li}_prev_chain'] = wq[:, prevc].mean().item()
            row[f'L{li}_chain_total'] = wq[:, chain].sum(1).mean().item()
        out.append(row)
    return out

if __name__ == '__main__':
    for path in sys.argv[1:]:
        m, task, c = load(path)
        prof = attn_profile(m, task, g=torch.Generator().manual_seed(7))
        keys = [k for k in prof[0] if k != 'j']
        print(path, json.dumps({k: round(sum(r[k] for r in prof) / len(prof), 3) for k in keys}))
