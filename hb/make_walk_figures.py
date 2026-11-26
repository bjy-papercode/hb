"""Figures for the walk / entropy-tilt section from results_w*.json and the saved fields.
  python experiments/make_walk_figures.py [out_dir]   (default ../numint/figures)
Writes fig_w1.pdf, fig_w3.pdf, fig_w4.pdf."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'numint', 'figures')
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MUTED = '#8a8880'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
J = lambda f: json.load(open(os.path.join(HERE, f)))

# ---------- W1 / W1b: error vs radius factor; frozen vs unbiased corrected walk ----------
if os.path.exists(os.path.join(HERE, 'results_w1.json')):
    r = J('results_w1.json')['rows']
    has_b = os.path.exists(os.path.join(HERE, 'results_w1b.json'))
    fig, ax = plt.subplots(1, 3 if has_b else 2, figsize=(8.6 if has_b else 7.6, 2.7))
    for i, (sch, mk) in enumerate([('direct', 'o'), ('gauge', 's')]):
        cs = sorted({x['c'] for x in r if x['scheme'] == sch})
        mu = [np.mean([abs(x['u_rel_err']) for x in r if x['scheme'] == sch and x['c'] == c]) for c in cs]
        ms = [np.mean([x['score_rel_err'] for x in r if x['scheme'] == sch and x['c'] == c]) for c in cs]
        ax[0].plot(cs, mu, marker=mk, c=C[i], ls='-', label=f'{sch}, $|\\hat u/u-1|$')
        ax[0].plot(cs, ms, marker=mk, c=C[i], ls='--', mfc='white', label=f'{sch}, score error')
    ax[0].set_xlabel('radius factor $c$'); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].grid(alpha=.3)
    ax[0].set_title('W1: mean error over anchors'); ax[0].legend(frameon=False, fontsize=6.3)
    if has_b:
        rb = J('results_w1b.json')['linear']
        cfg = [(s_, c_) for c_ in (1.0, 0.5) for s_ in ('direct', 'gauge')]
        xx = np.arange(len(cfg)); wdt = 0.36
        for j, (wk, col) in enumerate([('frozen', C[0]), ('corrected', C[1])]):
            zz = [np.mean([abs(x['z']) for x in rb if x['scheme'] == s_ and x['c'] == c_ and x['walk'] == wk]) for s_, c_ in cfg]
            wn = [np.mean([x['work_norm'] for x in rb if x['scheme'] == s_ and x['c'] == c_ and x['walk'] == wk]) for s_, c_ in cfg]
            ax[1].bar(xx + (j - .5) * wdt, zz, wdt, color=col, label=wk)
            ax[2].bar(xx + (j - .5) * wdt, wn, wdt, color=col, label=wk)
        for a_ in ax[1:]:
            a_.set_xticks(xx); a_.set_xticklabels([f'{s_}\n$c={c_:g}$' for s_, c_ in cfg], fontsize=7)
        ax[1].axhline(1, c=MUTED, lw=.8, ls=':'); ax[1].set_title('W1b: bias, mean $|z|$'); ax[1].legend(frameon=False)
        ax[2].set_yscale('log'); ax[2].set_title('W1b: (err$^2$+s.e.$^2$) $\\times$ seconds')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_w1.pdf')); plt.close(fig)
    print('fig_w1')

# ---------- W3: densities with parts switched off + potential ----------
z = {t: np.load(os.path.join(HERE, f)) for t, f in [('learned', 'w3_fields.npz'), ('dwrot', 'w3_fields_dwrot.npz')]
     if os.path.exists(os.path.join(HERE, f))}
if z:
    fig, ax = plt.subplots(len(z), 4, figsize=(8.8, 2.25 * len(z)), squeeze=False, gridspec_kw=dict(wspace=0.45))
    for i, (tag, F) in enumerate(z.items()):
        c = F['c']; ext = [c[0], c[-1], c[0], c[-1]]
        dx = c[1] - c[0]; vmax = max(np.sqrt(F[f'u_{nm}_0.5']).max() for nm in ['full', 'grad_only', 'rot_only'])
        for j, (nm, ttl) in enumerate([('full', 'full drift'), ('grad_only', 'curl-free part only'), ('rot_only', 'divergence-free part only')]):
            U = F[f'u_{nm}_0.5']
            imd = ax[i, j].imshow(np.sqrt(U.T), origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=vmax)
            ax[i, j].set_title(ttl if i == 0 else '')
            if nm != 'full':
                L1 = float(np.abs(U - F['u_full_0.5']).sum() * dx * dx)
                ax[i, j].text(0.04, 0.04, f'$L^1$ to full: {L1:.2g}', transform=ax[i, j].transAxes, fontsize=7,
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        cb = fig.colorbar(imd, cax=ax[i, 2].inset_axes([1.04, 0.0, 0.05, 1.0])); cb.ax.tick_params(labelsize=6.5)

        V = F['V']
        im = ax[i, 3].imshow(np.clip(V, -3, 3).T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-3, vmax=3)
        fr = F['frot']; s = slice(None, None, max(1, len(c) // 16)); nrm = np.linalg.norm(fr, axis=-1, keepdims=True) + 1e-12
        fu = fr / nrm                                                      # directions only (magnitudes in the text)
        ax[i, 3].quiver(c[s], c[s], fu[s, s, 0].T, fu[s, s, 1].T, color='k', alpha=.55, width=0.004, scale=22)
        ax[i, 3].set_title('$V$ (clipped to $\\pm3$), direction of $\\tilde f_1$' if i == 0 else ''); cb2 = fig.colorbar(im, cax=ax[i, 3].inset_axes([1.04, 0.0, 0.05, 1.0])); cb2.ax.tick_params(labelsize=6.5)
        ax[i, 0].set_ylabel({'learned': 'learned drift', 'dwrot': 'double well + rotation'}[tag])
        for j, a_ in enumerate(ax[i]):
            a_.set_xlim(-4.5, 4.5); a_.set_ylim(-4.5, 4.5); a_.set_xticks([-4, 0, 4]); a_.set_yticks([-4, 0, 4])
            a_.tick_params(labelsize=7, length=2)
            if j:
                a_.set_yticklabels([])
            if i < len(z) - 1:
                a_.set_xticklabels([])
    fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.08); fig.savefig(os.path.join(OUT, 'fig_w3.pdf')); plt.close(fig); print('fig_w3')

# ---------- W4: samples from p and q, and RMSE ----------
if os.path.exists(os.path.join(HERE, 'results_w4.json')):
    R = J('results_w4.json')['datasets']
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7))
    f = os.path.join(HERE, 'w4_samples_rare_mode.npz')
    if os.path.exists(f):
        S = np.load(f)
        for j, (k, ttl) in enumerate([('rare_mode_p', 'model trained on $p$'), ('rare_mode_q', 'model trained on $q$ (tilted)')]):
            x = S[k]; ax[j].scatter(x[:, 0], x[:, 1], s=1, alpha=.35, c=C[j]); ax[j].set_title(ttl)
            ax[j].set_xlim(-6, 6.5); ax[j].set_ylim(-3, 6); ax[j].set_xticks([]); ax[j].set_yticks([]); ax[j].set_aspect('equal')
    names = list(R); xx = np.arange(len(names)); wdt = 0.2
    for i, (key, lab) in enumerate([('low_direct', 'low-density mass, direct'), ('low_tilted', 'low-density mass, tilted'),
                                    ('rare_direct', 'rarest mode, direct'), ('rare_tilted', 'rarest mode, tilted')]):
        ax[2].bar(xx + (i - 1.5) * wdt, [R[n]['generated'][key]['rmse'] for n in names], wdt, color=C[i // 2], alpha=1.0 if i % 2 else 0.45, label=lab)
    ax[2].set_xticks(xx); ax[2].set_xticklabels([n.replace('_', ' ') for n in names]); ax[2].set_title('RMSE of $\\mathbb{P}_p$ estimates')
    ax[2].set_ylim(0, 1.6 * max(R[n]['generated'][k]['rmse'] for n in names for k in ['low_direct', 'rare_direct']))
    ax[2].legend(frameon=False, fontsize=6.3, ncol=2, loc='upper center')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_w4.pdf')); plt.close(fig); print('fig_w4')
