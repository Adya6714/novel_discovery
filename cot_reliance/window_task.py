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
    # index maps for the original layout x_1..x_L SEP c_1..c_L (L needed)
    def slot_index(s, j, L): return L + j
    def pred_index(s, j, L): return L + j - 1
    def enc(s, v): return v
    def dec(s, tok): return tok
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

class InterWindowTask(WindowTask):
    """Interleaved layout (main design after D14): x_1 c_1 x_2 c_2 ... x_L c_L.
    Inputs use tokens 0..p-1, chain values tokens p..2p-1. The prediction of c_j is made at the position of x_j.
    Every operand of both strategies sits at a fixed relative offset from the query (current input 0, previous chain token -1,
    leaving element x_{j+1-K} at -2K), independent of L."""
    def __init__(s, p, K, seed=0):
        super().__init__(p, K, seed); s.vocab = 2 * p
    def batch(s, B, L, mode='chain', g=None):
        assert mode == 'chain'
        g = g or s.g
        x = torch.randint(0, s.p, (B, L), generator=g); c = s.chain(x)
        seq = torch.stack([x, c + s.p], 2).reshape(B, 2 * L)
        tgt = torch.full_like(seq, -100); tgt[:, 1::2] = c + s.p
        return x, c, seq, tgt
    # index maps (j is 1-indexed)
    def slot_index(s, j, L=None): return 2 * j - 1          # where written c_j sits
    def pred_index(s, j, L=None): return 2 * (j - 1)        # logits here predict c_j
    def enc(s, v): return v + s.p
    def dec(s, tok): return tok - s.p
