"""Independent cost measure (response to review objection 3.1): learnability of each strategy in isolation.
Forced-READ and forced-RECOMPUTE models (attention whitelists in train.strategy_mask) are trained with identical settings;
cost = steps until the worst chain slot reaches >= 0.99 eval accuracy at L = 12 (evaluated every 100 steps), plus the area under
the worst-slot accuracy curve (AUC, higher = learned sooner). Prediction of the cost account: the free model's strategy at
slots j > K is the strategy with lower steps-to-criterion."""
import glob, json, re, numpy as np, torch, collections
rows = []
import sys as _sys
for p in sorted(glob.glob(_sys.argv[1] if len(_sys.argv) > 1 else 'runs/forced/*.pt')):
    ck = torch.load(p, weights_only=False); c = ck['cfg']; h = ck['hist']
    K = int(c['group'].split(':')[2]); its = [x[0] for x in h]; ms = [x[4] for x in h]
    stc = next((it for it, v in zip(its, ms) if v >= 0.99), None)
    s50 = next((it for it, v in zip(its, ms) if v >= 0.5), None)
    early = [v for it, v in zip(its, ms) if it <= 400]
    rows.append(dict(path=p, K=K, force=c['force'], seed=c['seed'], steps_to_criterion=stc, steps_to_half=s50,
                     auc=float(np.mean(early)), final_minslot=ms[-1]))
by = collections.defaultdict(list)
for r in rows: by[(r['K'], r['force'])].append(r)
print(f"{'K':>3} {'strategy':>10} {'steps to 0.99 (per seed)':>26} {'steps to 0.5':>18} {'AUC<=400':>9} {'final':>18}")
for (K, f), rs in sorted(by.items()):
    print(f"{K:>3} {f:>10} {str([r['steps_to_criterion'] for r in rs]):>26} {str([r['steps_to_half'] for r in rs]):>18} {np.mean([r['auc'] for r in rs]):>9.3f} {str([round(r['final_minslot'],3) for r in rs]):>18}")
json.dump(rows, open(_sys.argv[2] if len(_sys.argv) > 2 else 'results/forced_results.json', 'w'), indent=1)
