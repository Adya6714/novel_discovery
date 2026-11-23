"""Figures from results/sweep_results.json. Palette: categorical slots 1-2 of the reference palette
(blue #2a78d6 = FOLLOW / reads chain, orange #eb6834 = RECOMPUTE / ignores chain), neutral gray = OTHER."""
import json, sys, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE, ORANGE, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'legend.frameon': False})
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/sweep_results.json'))
out = sys.argv[2] if len(sys.argv) > 2 else '../figures'
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
rows = [r for k, r in res.items() if not k.startswith('untrained') and r['gate'] and r['cfg']['layers'] == depth]
Ks = sorted({r['K'] for r in rows})

def agg(kind, region, key):
    m, s, pts = [], [], []
    for K in Ks:
        v = [r.get(f'{kind}_{region}_{key}') for r in rows if r['K'] == K]
        v = [x for x in v if x is not None]
        pts.append(v); m.append(np.mean(v) if v else np.nan); s.append(np.std(v) if v else np.nan)
    return np.array(m), np.array(s), pts

# Figure 1: FOLLOW / RECOMPUTE vs K (slots j >= K), single and consistent
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
x = np.arange(len(Ks))
for ax, kind, title in zip(axes, ['single', 'consistent'], ['single-token corruption', 'coherent counterfactual chain']):
    for key, col, lab in [('follow', BLUE, 'FOLLOW (reads written chain)'), ('recompute', ORANGE, 'RECOMPUTE (ignores chain)')]:
        m, s, pts = agg(kind, 'jgeK', key)
        ax.plot(x, m, color=col, lw=2, marker='o', ms=5, label=lab, zorder=3)
        for xi, p in zip(x, pts):
            ax.scatter([xi] * len(p), p, s=10, color=col, alpha=0.45, linewidths=0, zorder=2)
    ax.set_xticks(x, [str(k) if k < 12 else '12\n(prefix)' for k in Ks]); ax.set_xlabel('window K'); ax.set_title(title, fontsize=9, color=INK)
    ax.axvspan(-0.5, Ks.index(2) + 0.5 if 2 in Ks else -0.5, color='#f3f2ee', zorder=0)
    ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel('fraction of predictions\n(slots j ≥ K, L = 12)')
axes[0].legend(loc='center right', fontsize=8)
fig.suptitle(f'Strategy selection tracks relative cost (depth {depth}; line = seed mean, dots = seeds)', fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(f'{out}/fig1_follow_vs_K_L{depth}.png', dpi=200); fig.savefig(f'{out}/fig1_follow_vs_K_L{depth}.pdf')

# Figure 2: per-slot FOLLOW for each K (seed mean), single corruption
fig, ax = plt.subplots(figsize=(4.6, 2.8))
cm = plt.get_cmap('Blues')
for i, K in enumerate(Ks):
    sel = [r for r in rows if r['K'] == K]
    f = np.mean([[x['follow'] for x in r['single']] for r in sel], 0)
    ax.plot(range(1, 12), f, color=cm(0.3 + 0.7 * i / max(1, len(Ks) - 1)), lw=1.6, marker='o', ms=3)
    ax.annotate(f'K={K}', (11, f[-1]), xytext=(3, 0), textcoords='offset points', fontsize=7, color=MUTED, va='center')
ax.set_xlabel('corrupted chain slot j'); ax.set_ylabel('FOLLOW rate'); ax.set_xlim(0.5, 12.3); ax.set_ylim(-0.03, 1.03)
ax.set_title('FOLLOW by slot (single corruption, seed mean)', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f'{out}/fig2_follow_by_slot_L{depth}.png', dpi=200); fig.savefig(f'{out}/fig2_follow_by_slot_L{depth}.pdf')

# Figure 3: attention signature: leaving element vs interior (last layer)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
KK = [K for K in Ks if 2 <= K < 12]
lay = depth - 1
lv = [np.mean([r['attn'][f'L{lay}_leave'] for r in rows if r['K'] == K]) for K in KK]
it = [np.mean([r['attn'][f'L{lay}_interior_per_pos'] for r in rows if r['K'] == K]) for K in KK]
xx = np.arange(len(KK)); w = 0.38
ax.bar(xx - w / 2 - 0.01, lv, w, color=BLUE, label='x_{j+1−K} (needed only by READ)')
ax.bar(xx + w / 2 + 0.01, it, w, color=ORANGE, label='window interior, per position (needed only by RECOMPUTE)')
ax.set_xticks(xx, [str(k) for k in KK]); ax.set_xlabel('window K'); ax.set_ylabel(f'attention (layer {lay}, head mean)')
ax.legend(fontsize=7, loc='upper left'); ax.set_title('Attention signature on clean inputs', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f'{out}/fig3_attention_L{depth}.png', dpi=200); fig.savefig(f'{out}/fig3_attention_L{depth}.pdf')
print('figures written for depth', depth, 'K', Ks, 'n models', len(rows))
