"""Paper figures from the experiment JSON files (no recomputation).
  python experiments/make_paper_figures.py [out_dir]      (default: ../numint/figures)
Writes fig_e1.pdf, fig_e2.pdf, fig_e3.pdf, fig_e7.pdf."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'numint', 'figures')
os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(os.path.join(HERE, f)))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']      # fixed categorical order
MK = ['o', 's', '^', 'D', 'v']
INK, MUTED = '#262624', '#8a8880'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'lines.markersize': 4.5, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); plt.close(fig); print('wrote', name)


# ---------------- E1 ----------------
r = J('results_e1.json'); ds = sorted({x['d'] for x in r})
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=False)
for i, d in enumerate(ds):
    sc = [x for x in r if x['d'] == d and x['radius'] == 'scaled']; fx = [x for x in r if x['d'] == d and x['radius'] == 'fixed']
    N = [x['N'] for x in sc]
    ax[0].loglog(N, [x['poly_rmse'] for x in sc], marker=MK[i], c=C[i], label=f'$d={d}$')
    ax[1].loglog(N, [x['heat_rmse'] for x in sc], marker=MK[i], c=C[i], label=f'$d={d}$')
    ax[1].loglog(N, [x['heat_rmse_antithetic'] for x in sc], marker=MK[i], c=C[i], ls='--', alpha=.6, mfc='white')
    ax[2].loglog([x['N'] for x in fx], [x['heat_rmse'] for x in fx], marker=MK[i], c=C[i], label=f'$d={d}$')
for a_ in ax:
    n = np.array([16, 16384]); a_.loglog(n, 0.5 * n ** -0.5, c=MUTED, ls=':', lw=1, label='$N^{-1/2}$'); a_.set_xlabel('$N$')
ax[0].set_ylabel('RMSE'); ax[1].set_ylabel('relative RMSE')
ax[0].set_title('(a) $|z|^2-2d\\sigma$ (mean 0)')
ax[1].set_title('(b) heat flow, $r=0.5\\sqrt{2/d}$\n(dashed: antithetic pairs)')
ax[2].set_title('(c) heat flow, fixed $r=0.5$')
ax[0].legend(frameon=False); save(fig, 'fig_e1.pdf')

# ---------------- E2 ----------------
r = J('results_e2.json')
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
a = r['a']; d = [x['d'] for x in a]
ax[0].loglog(d, [x['chi2p1_exact'] for x in a], c=C[0], label='exact $\\chi^2+1$ (closed form)')
ax[0].loglog(d, [x['chi2p1_mc'] for x in a], ls='none', marker='o', c=C[0], mfc='white', label='Monte Carlo')
ax[0].loglog(d, [x['lower_bound'] for x in a], c=C[1], ls='--', label='lower bound $|S|/|E_1|$')
ax[0].set_xlabel('$d$'); ax[0].set_title('(a) bounding-cylinder IS, constant integrand'); ax[0].legend(frameon=False)
c_ = r['c']
for i, rr in enumerate([0.25, 1.0]):
    rows = [x for x in c_ if x['r'] == rr]; dd = [x['d'] for x in rows]
    ax[1].loglog(dd, [x['heatball_rel_rmse'] for x in rows], marker=MK[0], c=C[0], ls='-' if i == 0 else '--',
                 label=f'heat ball, $r={rr}$')
    ax[1].loglog(dd, [x['semigroup_rel_rmse'] for x in rows], marker=MK[1], c=C[1], ls='-' if i == 0 else '--',
                 mfc='white', label=f'semigroup, $r={rr}$')
rows = [x for x in c_ if x['r'] == 0.25 and 'cylinder_IS_rmse_lower_bound_constant_part' in x]
ax[1].loglog([x['d'] for x in rows], [x['cylinder_IS_rmse_lower_bound_constant_part'] for x in rows], c=C[2],
             marker=MK[2], ls=':', label='cylinder IS (lower bound)')
ax[1].set_xlabel('$d$'); ax[1].set_ylabel('relative RMSE ($N=4096$)'); ax[1].set_title('(b) matched mean time lag')
ax[1].legend(frameon=False, ncol=1); save(fig, 'fig_e2.pdf')

# ---------------- E3 ----------------
r = J('results_e3.json')
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
from math import pi
for i, d in enumerate(sorted({x['d'] for x in r})):
    rows = [x for x in r if x['d'] == d]; rr = np.array([x['r'] for x in rows]); pw = rows[0]['pointwise']
    C1 = (d / (d + 2)) ** (d / 2 + 2) / pi
    ax[0].loglog(rr, np.abs([x['R'] for x in rows]), marker=MK[i], c=C[i], label=f'$d={d}$')
    ax[0].loglog(rr, C1 / 4 * abs(pw) * rr ** 2, c=MUTED, ls=':', lw=1)
    ax[1].semilogx(rr, [x['scaled'] / pw for x in rows], marker=MK[i], c=C[i], label=f'$d={d}$')
ax[1].axhline(1, c=MUTED, ls=':', lw=1)
_lo = min(x['r'] for x in r)
ax[1].axvspan(_lo * 0.8, 0.4, color='#2a78d6', alpha=0.07, lw=0)
ax[1].text(_lo * 0.9, 1.07, 'limit 1 (Theorem)', fontsize=7, color=MUTED, va='bottom')
ax[1].text(0.38, 0.05, 'within 1% for $r\\leq0.4$', fontsize=7, color=MUTED, ha='right', va='bottom', transform=ax[1].get_xaxis_transform())
ax[0].set_xlabel('$r$'); ax[0].set_ylabel('$|R_r|$'); ax[0].set_title('(a) residual (dotted: $C_1r^2/4\\times$ pointwise)')
ax[1].set_xlabel('$r$'); ax[1].set_ylabel('$4R_r/(C_1r^2)$ / pointwise'); ax[1].set_title('(b) scaled residual')
ax[0].legend(frameon=False); save(fig, 'fig_e3.pdf')

# ---------------- E7 ----------------
r = J('results_e7.json')
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1))
for j, d in enumerate([2, 16]):
    A = ax[j]
    floor = [abs(x['hb_res2']) for x in r['rows'] if x['d'] == d and x['eps'] == 0 and x['N'] == 2048 and x['kind'] != 'rot']
    for i, kind in enumerate(['global', 'low']):
        rows = [x for x in r['rows'] if x['d'] == d and x['kind'] == kind and x['eps'] > 0 and x['N'] == 2048]
        pw = np.array([x['pw_res2'] for x in rows]); hb = np.array([x['hb_res2'] for x in rows])
        pos = hb > 0
        A.loglog(pw[pos], hb[pos], ls='none', marker=MK[i], c=C[i], label={'global': '$\\sin(k\\cdot x)$ perturbation',
                                                                           'low': 'local bump'}[kind])
        if (~pos).any():
            A.loglog(pw[~pos], np.full((~pos).sum(), 1e-3), ls='none', marker=MK[i], c=C[i], mfc='white')
    lim = np.array([3e-4, 30]); A.loglog(lim, lim, c=MUTED, ls=':', lw=1)
    A.axhspan(1e-4, max(floor), color='#e6e5e0', alpha=.7, lw=0, label='noise floor ($\\varepsilon=0$)')
    A.set_xlim(*lim); A.set_ylim(5e-4, 50)
    A.set_xlabel('pointwise FP residual$^2$ (exact Laplacian)'); A.set_title(f'$d={d}$')
ax[0].set_ylabel('heat-ball residual$^2$ ($N=2048$)'); ax[0].legend(frameon=False, loc='upper left')
save(fig, 'fig_e7.pdf')
