"""Figure and LaTeX table for E10 from results_e10.json (+ results_e10c.json for the cost part).
  python experiments/make_e10_figure.py [figure_dir] [tex_out]"""
import os, sys, json
import numpy as np


def sci(x):
    m, e = f'{x:.1e}'.split('e'); return f'{m}\\times10^{{{int(e)}}}'

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'numint', 'figures')
TEX = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'table_e10.tex')
R = json.load(open(os.path.join(HERE, 'results_e10.json')))
if os.path.exists(os.path.join(HERE, 'results_e10c.json')):
    R['c'] = json.load(open(os.path.join(HERE, 'results_e10c.json')))['c']
C = ['#2a78d6', '#eb6834', '#1baf7a']
MUTED = '#8a8880'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
dims = sorted({x['d'] for x in R['b']})
fig, ax = plt.subplots(1, len(dims), figsize=(2.9 * len(dims), 2.6), squeeze=False)
for j, d in enumerate(dims):
    A = ax[0, j]
    for k, (meth, ls) in enumerate([('quadrature', '-'), ('monte_carlo', '--')]):
        for qty, mk in [('rel_err_score', 'o'), ('rel_err_logu', 's')]:
            rows = [x for x in R['b'] if x['d'] == d and x['method'] == meth and x['r'] == 1.0]
            ev = sorted({x['evals'] for x in rows})
            e = [np.mean([x[qty] for x in rows if x['evals'] == v]) for v in ev]
            A.plot(ev, e, ls=ls, marker=mk, ms=3, c=C[k], label=f"{'quadrature' if k == 0 else 'Monte Carlo'}, {'score' if qty == 'rel_err_score' else 'log-density'}")
    A.set_xscale('log'); A.set_yscale('log'); A.grid(alpha=.3); A.set_title(f'$d={d}$, $r=1$')
    A.set_xlabel('evaluations of $s$ (or $v$)')
    if j == 0:
        A.set_ylabel('relative error'); A.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_e10.pdf')); plt.close(fig)
print('fig_e10.pdf')

# ---- table: (a) MC, VJP form vs IBP form, N = N' = 4096; quadrature at a comparable budget; (c) cost
lines = ['\\begin{tabular}{rcccccc}', '\\toprule',
         ' & \\multicolumn{2}{c}{Monte Carlo, $N=N\'=4096$} & \\multicolumn{2}{c}{quadrature, form (IBP)} & \\multicolumn{2}{c}{ms/step, vector net} \\\\',
         '$d$ & form (VJP) & form (IBP) & evaluations & error & VJP & IBP \\\\', '\\midrule']
for d in sorted({x['d'] for x in R['a']}):
    a = [x for x in R['a'] if x['d'] == d and x['N'] == 4096]
    ev = lambda f: np.mean([x['rel_err_rms'] for x in a if x['form'] == f])
    q = [x for x in R['b'] if x['d'] == d and x['method'] == 'quadrature']
    if q:
        tgt = min(q, key=lambda x: abs(np.log(x['evals'] / 12288)))['evals']
        qe = np.mean([x['rel_err_score'] for x in q if x['evals'] == tgt]); qs = f'{tgt}', f'${sci(qe)}$'
    else:
        qs = '---', '---'
    cc = {x['form']: x['ms_per_step'] for x in R.get('c', []) if x['d'] == d and x['model'] == 'vector' and 'ms_per_step' in x}
    lines.append(f"{d} & ${sci(ev('vjp'))}$ & ${sci(ev('ibp'))}$ & {qs[0]} & {qs[1]} & "
                 f"{cc.get('vjp', float('nan')):.0f} & {cc.get('ibp', float('nan')):.0f} \\\\" if cc else
                 f"{d} & ${sci(ev('vjp'))}$ & ${sci(ev('ibp'))}$ & {qs[0]} & {qs[1]} & --- & --- \\\\")
for d in sorted({x['d'] for x in R.get('c', [])} - {x['d'] for x in R['a']}):
    cc = {x['form']: x['ms_per_step'] for x in R['c'] if x['d'] == d and x['model'] == 'vector' and 'ms_per_step' in x}
    lines.append(f"{d} & --- & --- & --- & --- & {cc.get('vjp', float('nan')):.0f} & {cc.get('ibp', float('nan')):.0f} \\\\")
lines += ['\\bottomrule', '\\end{tabular}']
open(TEX, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
