"""Causal interventions on written chain tokens (layout-agnostic via task index maps).
At chain slot j (1 <= j < L) we overwrite the written c_j with r != c_j and read the model's greedy prediction for c_{j+1}:
  FOLLOW    : prediction == READ(r)  = r + x_{j+1} - x_{j+1-K}   (the model uses its written chain)
  RECOMPUTE : prediction == true c_{j+1}                         (the model ignores the overwritten value)
  OTHER     : anything else.
kind='single'     : only c_j is overwritten (uniform r != c_j).
kind='consistent' : c_1..c_j are replaced by the chain of independently redrawn inputs x'_1..x'_j (coherent but contradicting the
                    visible inputs); rows with c'_j == c_j are dropped. Reading *any* chain token then implies FOLLOW."""
import json, sys, torch
from train import make_task; from model import GPT

def load(path):
    ck = torch.load(path, weights_only=False); c = ck['cfg']
    task = make_task(c['group'], seed=12345)
    m = GPT(task.vocab, d=c['d'], layers=c['layers'], maxlen=2*c['Lmax']+2); m.load_state_dict(ck['state']); m.eval()
    return m, task, c

@torch.no_grad()
def clean_acc(m, task, L, mode='chain', n=2000, g=None):
    x, c, seq, tgt = task.batch(n, L, mode, g=g)
    pred = m(seq[:, :-1]).argmax(-1)
    if mode != 'chain':
        return [(pred[:, L] == c[:, -1]).float().mean().item()]
    return [(task.dec(pred[:, task.pred_index(j, L)]) == c[:, j-1]).float().mean().item() for j in range(1, L+1)]

@torch.no_grad()
def intervene(m, task, L, j, n=2000, g=None, freerun=True, kind='single'):
    x, c, seq, _ = task.batch(n, L, 'chain', g=g)
    G = task.G; s = seq.clone()
    if kind == 'single':
        r = (c[:, j-1] + torch.randint(1, G, (n,), generator=g)) % G
        s[:, task.slot_index(j, L)] = task.enc(r)
    else:
        xc = x.clone(); xc[:, :j] = torch.randint(0, G, (n, j), generator=g)
        cc = task.prefix(xc) if hasattr(task, 'prefix') else task.chain(xc)
        for k in range(1, j+1): s[:, task.slot_index(k, L)] = task.enc(cc[:, k-1])
        r = cc[:, j-1]
    keep = r != c[:, j-1]
    x, c, s, r = x[keep], c[keep], s[keep], r[keep]
    q = task.pred_index(j+1, L)                  # position whose logits predict c_{j+1}
    ctx = s[:, :q+1]
    probs = m(ctx)[:, -1].softmax(-1); pred = task.dec(probs.argmax(-1))
    fol = task.follow_token(r, x, j); rec = c[:, j]
    out = dict(L=L, j=j, kind=kind, n=int(keep.sum()),
               follow=(pred == fol).float().mean().item(), recompute=(pred == rec).float().mean().item(),
               p_follow=probs.gather(1, task.enc(fol)[:, None]).mean().item(),
               p_recompute=probs.gather(1, task.enc(rec)[:, None]).mean().item())
    out['other'] = 1 - out['follow'] - out['recompute']
    if freerun:   # greedy continuation: write predicted chain values, insert the true inputs, until c_L
        cur = s.clone(); f = r.clone()
        for k in range(j+1, L+1):
            qk = task.pred_index(k, L)
            nxt = m(cur[:, :qk+1])[:, -1].argmax(-1)
            cur[:, task.slot_index(k, L)] = nxt
            f = task.follow_token(f, x, k-1)
        final = task.dec(cur[:, task.slot_index(L, L)])
        out['final_follow'] = (final == f).float().mean().item()
        out['final_true'] = (final == c[:, -1]).float().mean().item()
    return out
