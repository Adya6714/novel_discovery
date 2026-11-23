"""Word problem with chain of thought.
Sequence: x_1..x_L  SEP  p_1..p_L   where p_j = x_1 * ... * x_j (left-to-right product).
mode='chain': loss on p_1..p_L. mode='direct': sequence x_1..x_L SEP p_L, loss on p_L only."""
import numpy as np, torch

class WordTask:
    def __init__(s, T, seed=0):
        s.T = torch.as_tensor(T); s.G = len(T); s.SEP = s.G; s.vocab = s.G + 1
        s.g = torch.Generator().manual_seed(seed)
    def prefix(s, x):
        p = torch.empty_like(x); cur = x[:, 0]; p[:, 0] = cur
        for j in range(1, x.shape[1]):
            cur = s.T[cur, x[:, j]]; p[:, j] = cur
        return p
    def batch(s, B, L, mode='chain', g=None):
        g = g or s.g
        x = torch.randint(0, s.G, (B, L), generator=g)
        p = s.prefix(x); sep = torch.full((B, 1), s.SEP)
        if mode == 'chain':
            seq = torch.cat([x, sep, p], 1)
            tgt = torch.full_like(seq, -100); tgt[:, L+1:2*L+1] = p  # token p_j sits at index L+j; logits at index i are scored against tgt[i+1]
        else:
            seq = torch.cat([x, sep, p[:, -1:]], 1)
            tgt = torch.full_like(seq, -100); tgt[:, L+1] = p[:, -1]
        return x, p, seq, tgt

    def follow_token(s, r, x, j):
        return s.T[r, x[:, j]]
