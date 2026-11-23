"""At slot j = K the leaving element x_1 also appears in the written chain as c_1 (= x_1 for window sums).
Under the coherent counterfactual, a model that takes the leaving element from the WRITTEN c'_1 predicts r + x_{K+1} - c'_1
('chain-leave'), whereas one that takes it from the input predicts r + x_{K+1} - x_1 (FOLLOW). Measures both."""
import sys, torch
from evaluate import load
@torch.no_grad()
def run(path, n=2000):
    m, t, c = load(path); K = t.K; L = 12; j = K; g = torch.Generator().manual_seed(3)
    x, cc0, seq, _ = t.batch(n, L, g=g); s = seq.clone()
    xc = x.clone(); xc[:, :j] = torch.randint(0, t.p, (n, j), generator=g); cc = t.chain(xc)
    for k in range(1, j + 1): s[:, t.slot_index(k, L)] = t.enc(cc[:, k - 1])
    r = cc[:, j - 1]; keep = (r != cc0[:, j - 1]) & (xc[:, 0] != x[:, 0])
    pred = t.dec(m(s[:, :t.pred_index(j + 1, L) + 1])[:, -1].argmax(-1))
    fol = (r + x[:, j] - x[:, 0]) % t.p; alt = (r + x[:, j] - xc[:, 0]) % t.p
    k = keep
    print(f'{path} K={K} j=K: FOLLOW(input x1)={(pred[k]==fol[k]).float().mean():.3f} chain-leave(written c1)={(pred[k]==alt[k]).float().mean():.3f} true={(pred[k]==cc0[k, j]).float().mean():.3f}')
for p in sys.argv[1:]: run(p)
