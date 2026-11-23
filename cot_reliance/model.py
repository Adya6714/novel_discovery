import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1 = nn.LayerNorm(d); s.attn = nn.MultiheadAttention(d, h, batch_first=True)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, mask):
        h = s.ln1(x); a, _ = s.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, vocab, d=128, h=4, layers=2, maxlen=64):
        super().__init__()
        s.tok = nn.Embedding(vocab, d); s.pos = nn.Embedding(maxlen, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, vocab)
    def forward(s, idx, return_resid=False):
        T = idx.shape[1]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        resid = [x]
        for b in s.blocks:
            x = b(x, mask); resid.append(x)
        out = s.head(s.lnf(x))
        return (out, resid) if return_resid else out
