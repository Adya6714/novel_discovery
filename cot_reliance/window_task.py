"""Window-sum chain task (the main experimental task).
Inputs x_1..x_L uniform in Z_p.  Written chain c_j = sum_{i=max(1,j-K+1)}^{j} x_i  (mod p).
K >= L gives the ordinary prefix-sum chain-of-thought.  Sequence: x_1..x_L SEP c_1..c_L.
Two zero-loss strategies exist for predicting c_{j+1} at chain position j:
  READ      : c_{j+1} = c_j + x_{j+1} - x_{j+1-K}     (3 operands, uses the written chain)
  RECOMPUTE : c_{j+1} = x_{j+2-K} + ... + x_{j+1}      (min(K, j+1) operands, ignores the chain)"""
import torch

class WindowTask:
    def __init__(s, p, K, seed=0):
        s.p, s.K = p, K; s.G = p; s.SEP = p; s.vocab = p + 1
        s.g = torch.Generator().manual_seed(seed)
    def chain(s, x):
        cs = torch.cumsum(x, 1)
        c = cs.clone()
        if s.K < x.shape[1]:
            c[:, s.K:] = cs[:, s.K:] - cs[:, :-s.K]
        return c % s.p
    def batch(s, B, L, mode='chain', g=None):
        g = g or s.g
        x = torch.randint(0, s.p, (B, L), generator=g)
        c = s.chain(x); sep = torch.full((B, 1), s.SEP)
        if mode == 'chain':
            seq = torch.cat([x, sep, c], 1); tgt = torch.full_like(seq, -100); tgt[:, L+1:2*L+1] = c
        else:  # direct: predict c_L straight after SEP
            seq = torch.cat([x, sep, c[:, -1:]], 1); tgt = torch.full_like(seq, -100); tgt[:, L+1] = c[:, -1]
        return x, c, seq, tgt
    def follow_token(s, r, x, j):
        """value of c_{j+1} implied by READ if the written c_j were r (j 1-indexed, x 0-indexed tensor)."""
        f = r + x[:, j]
        if j + 1 - s.K >= 1:
            f = f - x[:, j - s.K]
        return f % s.p

if __name__ == '__main__':
    t = WindowTask(11, 3); x, c, seq, tgt = t.batch(2, 6)
    print(x); print(c)
    # check READ identity holds on clean data
    for j in range(1, 6):
        assert (t.follow_token(c[:, j-1], x, j) == c[:, j]).all()
    print('READ identity OK')
