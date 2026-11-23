"""Causal intervention on written chain tokens.
For a chain model and sequence x_1..x_L SEP p_1..p_L, we overwrite p_j (1<=j<L) with r != p_j and read the
model's prediction for the next chain slot. Outcomes:
  follow    : argmax == r * x_{j+1}     (model reads its written chain)
  recompute : argmax == p_{j+1}         (model ignores the overwritten token, recomputes from the input)
  other
Also reports probability mass on the follow / recompute tokens, and free-running continuation to the end."""
import json, sys, torch, numpy as np
from train import make_task; from model import GPT

def load(path):
    ck = torch.load(path, weights_only=False); c = ck['cfg']
    task = make_task(c['group'], seed=12345)
    m = GPT(task.vocab, d=c['d'], layers=c['layers'], maxlen=2*c['Lmax']+2); m.load_state_dict(ck['state']); m.eval()
    return m, task, c

@torch.no_grad()
def clean_acc(m, task, L, mode, n=2000, g=None):
    x, p, seq, tgt = task.batch(n, L, mode, g=g)
    pred = m(seq[:, :-1]).argmax(-1); t = tgt[:, 1:]
    if mode == 'chain':
        per_pos = (pred[:, L:2*L-0-0][:, :L] == p).float().mean(0)  # teacher-forced accuracy at each chain slot
        return per_pos.tolist()
    return [(pred[:, L] == p[:, -1]).float().mean().item()]

@torch.no_grad()
def intervene(m, task, L, j, n=2000, g=None, freerun=True, kind='single'):
    """j is 1-indexed chain position to corrupt (1..L-1).
    kind='single'     : overwrite only p_j with r != p_j.
    kind="consistent" : overwrite p_1..p_j with the chain of counterfactual inputs x_1..x_j (redrawn),
                        so the written chain is internally coherent but disagrees with the input; r = p_j'."""
    x, p, seq, _ = task.batch(n, L, 'chain', g=g)
    G = task.G
    s = seq.clone()
    if kind == 'single':
        r = (p[:, j-1] + torch.randint(1, G, (n,), generator=g)) % G  # uniform r != p_j
        s[:, L + j] = r                          # chain slot j lives at index L+j (SEP at index L)
    else:
        # counterfactual inputs x' (first input changed) -> its chain written in slots 1..j; inputs shown are the real ones
        xc = x.clone(); xc[:, :j] = torch.randint(0, G, (n, j), generator=g)
        if hasattr(task, 'prefix'): cc = task.prefix(xc)
        else: cc = task.chain(xc)
        s[:, L + 1:L + 1 + j] = cc[:, :j]; r = cc[:, j - 1]
    keep = r != p[:, j - 1]
    x, p, s, r = x[keep], p[keep], s[keep], r[keep]
    ctx = s[:, :L + j + 1]
    probs = m(ctx)[:, -1].softmax(-1)
    pred = probs.argmax(-1)
    fol_tok = task.follow_token(r, x, j)         # READ prediction given written c_j = r
    rec_tok = p[:, j]                            # true p_{j+1}
    out = dict(L=L, j=j, kind=kind,
               follow=(pred == fol_tok).float().mean().item(),
               recompute=(pred == rec_tok).float().mean().item(),
               p_follow=probs.gather(1, fol_tok[:, None]).mean().item(),
               p_recompute=probs.gather(1, rec_tok[:, None]).mean().item())
    out['other'] = 1 - out['follow'] - out['recompute']
    if freerun:  # generate remaining chain greedily, compare final token
        cur = ctx
        for _ in range(L - j):
            nxt = m(cur)[:, -1].argmax(-1, keepdim=True); cur = torch.cat([cur, nxt], 1)
        final = cur[:, -1]
        # final value implied by following corrupted r: r * x_{j+1} * ... * x_L
        f = r.clone()
        for k in range(j, L): f = task.follow_token(f, x, k)
        out['final_follow'] = (final == f).float().mean().item()
        out['final_true'] = (final == p[:, -1]).float().mean().item()
    return out

if __name__ == '__main__':
    res = {}
    for path in sys.argv[1:]:
        m, task, c = load(path); g = torch.Generator().manual_seed(999)
        r = dict(cfg=c)
        if c['mode'] == 'direct':
            r['direct_acc'] = {L: clean_acc(m, task, L, 'direct', g=g)[0] for L in [2, 4, 6, 8, 10, 12]}
        else:
            r['clean'] = {L: clean_acc(m, task, L, 'chain', g=g) for L in [4, 8, 12]}
            r['interv'] = [intervene(m, task, L, j, g=g, kind=k) for k in ['single', 'consistent'] for L in [8, 12] for j in range(1, L)]
        res[path] = r
        print(json.dumps(r)[:3000])
    if len(sys.argv) > 1:
        json.dump(res, open('results_last_eval.json', 'w'), indent=1)
