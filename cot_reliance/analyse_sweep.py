"""Evaluate every model in runs/sweep (or given glob) and write results/sweep_results.json + a CSV summary.
Per model: clean teacher-forced accuracy at L=12 (gate >= 0.98 at every slot), and for single / consistent corruption
at L=12, FOLLOW / RECOMPUTE / OTHER for each slot j=1..11 (n=2000 each), probability mass, free-run final outcomes,
and attention signature (mass on the leaving element vs window interior, clean sequences)."""
import glob, json, sys, os, torch, numpy as np
from evaluate import load, clean_acc, intervene
from attention import attn_profile
from model import GPT
from train import make_task

torch.set_num_threads(int(os.environ.get('NT', '1')))
pattern = sys.argv[1] if len(sys.argv) > 1 else 'runs/sweep/*.pt'
out = sys.argv[2] if len(sys.argv) > 2 else 'results/sweep_results.json'
os.makedirs('results', exist_ok=True)
res = json.load(open(out)) if os.path.exists(out) else {}

def summarise(m, task, c, g):
    K = task.K; L = 12
    r = dict(cfg=c, K=K)
    r['clean'] = clean_acc(m, task, L, 'chain', n=2000, g=g)
    r['gate'] = min(r['clean']) >= 0.98
    for kind in ['single', 'consistent']:
        rows = [intervene(m, task, L, j, n=2000, g=g, kind=kind, freerun=True) for j in range(1, L)]
        r[kind] = rows
        for name, sel in [('jgeK', [x for x in rows if x['j'] >= K]), ('jltK', [x for x in rows if x['j'] < K])]:
            if sel:
                for key in ['follow', 'recompute', 'other', 'p_follow', 'p_recompute', 'final_follow', 'final_true']:
                    r[f'{kind}_{name}_{key}'] = float(np.mean([x[key] for x in sel]))
    if K >= 2:
        prof = attn_profile(m, task, g=g)
        keys = [k for k in prof[0] if k != 'j']
        r['attn'] = {k: float(np.mean([x[k] for x in prof])) for k in keys}
    return r

for path in sorted(glob.glob(pattern)):
    if path in res: continue
    m, task, c = load(path); g = torch.Generator().manual_seed(2024)
    res[path] = summarise(m, task, c, g)
    s = res[path]
    print(f"{path} gate={s['gate']} minacc={min(s['clean']):.3f} follow(j>=K)={s.get('single_jgeK_follow', float('nan')):.3f} "
          f"recomp(j>=K)={s.get('single_jgeK_recompute', float('nan')):.3f} cons_follow={s.get('consistent_jgeK_follow', float('nan')):.3f}", flush=True)
    json.dump(res, open(out, 'w'))

if len(sys.argv) > 3 and sys.argv[3] == '--untrained':   # negative control
    for K in [3, 12]:
        task = make_task(f'win:11:{K}', seed=12345); torch.manual_seed(0)
        m = GPT(task.vocab, d=128, layers=2, maxlen=26).eval()
        s = summarise(m, task, dict(group=f'win:11:{K}', layers=2, untrained=True), torch.Generator().manual_seed(2024))
        res[f'untrained_K{K}'] = s
        print('untrained', K, s['single_jgeK_follow'], s['single_jgeK_recompute'])
    json.dump(res, open(out, 'w'))
