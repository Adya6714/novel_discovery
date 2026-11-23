"""Unit test: the token at index i+1 must equal tgt[i+1] wherever tgt is set (targets align with the tokens they score)."""
from train import make_task
for spec in ['win:11:3', 'win:11:12', 'Z60', 'A5', 'iwin:11:3', 'iwin:11:12']:
    t = make_task(spec)
    for mode in (['chain'] if spec.startswith('iwin') else ['chain', 'direct']):
        x, c, seq, tgt = t.batch(64, 7, mode)
        m = tgt != -100
        assert (seq[m] == tgt[m]).all(), (spec, mode)
        assert m.sum(1).eq(7 if mode == 'chain' else 1).all()
print('target alignment OK')
