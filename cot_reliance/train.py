import argparse, json, time, os, torch, torch.nn.functional as F, numpy as np
from groups import get_group; from task import WordTask; from window_task import WindowTask, InterWindowTask; from model import GPT

def make_task(spec, seed=0):
    """spec: 'Z60' / 'A5' (group word problem) or 'win:p:K' (window-sum chain)."""
    if spec.startswith('win:'):
        _, p, K = spec.split(':'); return WindowTask(int(p), int(K), seed=seed)
    if spec.startswith('iwin:'):
        _, p, K = spec.split(':'); return InterWindowTask(int(p), int(K), seed=seed)
    return WordTask(get_group(spec), seed=seed)

def train(group, mode='chain', layers=2, d=128, steps=6000, B=256, Lmin=2, Lmax=12, wd=0.01, lr=1e-3, seed=0, out=None, log_every=500, ckpt_every=0, init=None):
    torch.manual_seed(seed)
    task = make_task(group, seed=seed)
    m = GPT(task.vocab, d=d, layers=layers, maxlen=2*Lmax+2)
    prev_steps = 0
    if init:
        ck = torch.load(init, weights_only=False); m.load_state_dict(ck['state']); prev_steps = ck['cfg']['steps']
        task = make_task(group, seed=seed + 100003 * prev_steps)   # fresh data stream for the continuation
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(seed + prev_steps); t0 = time.time(); hist = []
    for it in range(steps):
        L = int(rng.integers(Lmin, Lmax+1))
        _, _, seq, tgt = task.batch(B, L, mode)
        logits = m(seq[:, :-1]); loss = F.cross_entropy(logits.reshape(-1, task.vocab), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (it+1) % log_every == 0:
            with torch.no_grad():
                _, _, sq, tg = task.batch(512, Lmax, mode, g=torch.Generator().manual_seed(it))
                lg = m(sq[:, :-1]); evl = F.cross_entropy(lg.reshape(-1, task.vocab), tg[:, 1:].reshape(-1)).item()
                ok = tg[:, 1:] != -100; evacc = (lg.argmax(-1)[ok] == tg[:, 1:][ok]).float().mean().item()
            hist.append((it+1, loss.item(), evl, evacc))
            if ckpt_every and (it+1) % ckpt_every == 0 and out:
                torch.save({'state': m.state_dict(), 'cfg': dict(wd=wd, lr=lr, group=group, mode=mode, layers=layers, d=d, Lmax=Lmax, Lmin=Lmin, seed=seed, steps=it+1)}, out.replace('.pt', f'_it{it+1}.pt'))
            print(f'{group} {mode} L{layers} it{it+1} loss {loss.item():.4f} evalL{Lmax} loss {evl:.4f} acc {evacc:.4f} {time.time()-t0:.0f}s', flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({'state': m.state_dict(), 'cfg': dict(wd=wd, lr=lr, group=group, mode=mode, layers=layers, d=d, Lmax=Lmax, Lmin=Lmin, seed=seed, steps=prev_steps+steps, init=init), 'hist': hist}, out)
    return m, task

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group'); ap.add_argument('--mode', default='chain'); ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--out'); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--B', type=int, default=256); ap.add_argument('--ckpt_every', type=int, default=0); ap.add_argument('--init'); ap.add_argument('--Lmin', type=int, default=2)
    a = ap.parse_args()
    train(a.group, a.mode, a.layers, a.d, a.steps, B=a.B, Lmin=a.Lmin, lr=a.lr, wd=a.wd, seed=a.seed, out=a.out, ckpt_every=a.ckpt_every, init=a.init)
