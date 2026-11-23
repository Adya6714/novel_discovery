"""Figures for the final report. Palette: reference categorical slots 1-2 (blue #2a78d6 = READ / follows chain,
orange #eb6834 = RECOMPUTE / ignores chain), neutral gray for mixed/other. Usage: python3 make_figures.py [outdir]"""
import json, sys, glob, collections, numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
BLUE, ORANGE, GRAY, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#b9b7b0', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'legend.frameon': False, 'axes.titlesize': 9, 'axes.titlecolor': INK})
out = sys.argv[1] if len(sys.argv) > 1 else '../figures'
R = json.load(open('results/sweep_results.json'))
rows = [r for k, r in R.items() if k.startswith('runs/sweep/') and r['gate']]
Ks = sorted({r['K'] for r in rows})

# ---- Fig 1: free-model strategy vs K (slots j > K; K=12 uses all slots), three intervention types
def val(r, kind, key):
    if r['K'] >= 12: return float(np.mean([x[key] for x in r[kind]]))
    return r[f'{kind}_jgtK_{key}']
fig, ax = plt.subplots(figsize=(5.0, 2.7))
x = np.arange(len(Ks))
for kind, dx, mk in [('single', -0.18, 'o'), ('consistent', 0.0, 's'), ('input', 0.18, '^')]:
    for key, col in [('follow', BLUE), ('recompute', ORANGE)]:
        k2 = {'follow': 'read', 'recompute': 'recompute'}[key] if kind == 'input' else key
        for xi, K in zip(x, Ks):
            vs = []
            for r in rows:
                if r['K'] != K: continue
                if kind == 'input':
                    if K < 2: continue
                    vs.append(r[f'input_jgtK_{k2}'] if K < 12 else np.mean([q[f'{k2}_consistent'] for q in r['input']]))
                else: vs.append(val(r, kind, key))
            ax.scatter([xi + dx] * len(vs), vs, s=14, marker=mk, color=col, alpha=0.85, linewidths=0, zorder=3)
m = [np.mean([val(r, 'single', 'follow') for r in rows if r['K'] == K]) for K in Ks]
ax.plot(x - 0.18, m, color=BLUE, lw=1.6, zorder=2)
ax.set_xticks(x, [str(k) if k < 12 else '12\n(prefix)' for k in Ks]); ax.set_xlabel('window K')
ax.set_ylabel('fraction of predictions (slots j > K)'); ax.set_ylim(-0.04, 1.04)
ax.set_title('Free models: which route dominates? (each marker = one trained model)')
ax.legend(handles=[Patch(color=BLUE, label='READ (uses written chain)'), Patch(color=ORANGE, label='RECOMPUTE (ignores chain)'),
    plt.Line2D([], [], marker='o', ls='', color=MUTED, label='chain-token corruption'),
    plt.Line2D([], [], marker='s', ls='', color=MUTED, label='coherent counterfactual chain'),
    plt.Line2D([], [], marker='^', ls='', color=MUTED, label='input-side corruption')], fontsize=6.8, loc='center right')
fig.tight_layout(); fig.savefig(f'{out}/fig1_strategy_vs_K.png', dpi=220); plt.close(fig)

# ---- Fig 2: forced-strategy learning curves (independent cost)
FK = sorted({int(torch.load(p, weights_only=False)['cfg']['group'].split(':')[2]) for p in glob.glob('runs/forced/*.pt')})
fig, axes = plt.subplots(1, len(FK), figsize=(7.2, 2.2), sharey=True)
for ax, K in zip(np.atleast_1d(axes), FK):
    for f, col in [('read', BLUE), ('recompute', ORANGE)]:
        for p in sorted(glob.glob(f'runs/forced/K{K}_{f}_s*.pt')):
            h = torch.load(p, weights_only=False)['hist']
            ax.plot([q[0] for q in h], [q[4] for q in h], color=col, lw=1.3, alpha=0.9)
    ax.set_title(f'K = {K}'); ax.set_xlim(0, 800); ax.set_xlabel('training step'); ax.set_ylim(-0.03, 1.03)
np.atleast_1d(axes)[0].set_ylabel('worst-slot accuracy')
np.atleast_1d(axes)[0].legend(handles=[Patch(color=BLUE, label='READ-only'), Patch(color=ORANGE, label='RECOMPUTE-only')], fontsize=7, loc='lower right')
fig.suptitle('Independent cost: learnability of each strategy when the other is masked out (one line per seed)', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f'{out}/fig2_forced_learnability.png', dpi=220); plt.close(fig)

# ---- Fig 3: per-slot census grid
try:
    C = json.load(open('results/census.json'))
    items = sorted(C.items(), key=lambda kv: (kv[1]['posoff'], kv[1]['K'], kv[1]['seed'], kv[1]['offset']))
    items = [kv for kv in items if kv[1]['K'] in (2, 3, 4)]
    fig, ax = plt.subplots(figsize=(5.2, 0.22 * len(items) + 0.9))
    cmap = {'R': BLUE, 'C': ORANGE, 'm': GRAY}
    for i, (k, v) in enumerate(items):
        for j, ch in enumerate(v['labels']):
            ax.add_patch(plt.Rectangle((j + 1 - 0.45, i - 0.4), 0.9, 0.8, color=cmap[ch], lw=0))
    ax.set_yticks(range(len(items)), [f"K={v['K']} seed {v['seed']}" + (f" (pos-shift trained, eval +{v['offset']})" if v['posoff'] else '') for k, v in items], fontsize=6.5)
    ax.set_xticks(range(1, 12)); ax.set_xlabel('corrupted chain slot j'); ax.set_xlim(0.4, 11.6); ax.set_ylim(-0.6, len(items) - 0.4)
    ax.invert_yaxis(); ax.grid(False)
    ax.legend(handles=[Patch(color=BLUE, label='READ'), Patch(color=ORANGE, label='RECOMPUTE'), Patch(color=GRAY, label='mixed')],
              fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.03), ncol=3)
    ax.set_title('Per-slot dominant route (FOLLOW or RECOMPUTE ≥ 0.9)', pad=22)
    fig.tight_layout(); fig.savefig(f'{out}/fig3_census.png', dpi=220); plt.close(fig)
except FileNotFoundError:
    print('no census yet')
print('figures written')
