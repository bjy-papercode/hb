"""E0: verification of the Beta-Gamma sampler family mu_{a,b,c} (density ~ |z|^{2a} psi0^b sigma^{-c} on E1),
using the SAME torch sampler as the training code (hbmc.heatball.sample_family) and the closed-form mass
hbmc.heatball.Z_family, against 2D quadrature over E1.
Output: results_e0.json and table_e0.tex (paper table for the E0 TODO)."""
import sys, os, json, math
import numpy as np, torch
from scipy import integrate
from scipy.special import gammaln
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_family, Z_family
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(11)
NMC = int(os.environ.get('NMC', 1_500_000))


def R2(s, d):
    return -2 * d * s * np.log(4 * np.pi * s)


def quad(d, a, b, c, f):
    S = 2 * np.pi ** (d / 2) / np.exp(gammaln(d / 2))
    w = lambda r2, s: r2 ** a * max(-d / 2 * np.log(4 * np.pi * s) - r2 / (4 * s), 0.0) ** b * s ** (-c)
    def inner(s, g):
        R = np.sqrt(R2(s, d))
        return integrate.quad(lambda r: S * r ** (d - 1) * w(r * r, s) * g(r * r, s), 0, R, limit=200)[0]
    num = integrate.quad(lambda s: inner(s, f), 0, 1 / (4 * np.pi), limit=300)[0]
    den = integrate.quad(lambda s: inner(s, lambda r2, s: 1.0), 0, 1 / (4 * np.pi), limit=300)[0]
    return num / den, den


tests = [("|z|^2", lambda r2, s: r2, lambda z, s: (z ** 2).sum(1)),
         ("sigma", lambda r2, s: s, lambda z, s: s),
         ("e^{-|z|^2}(1+sigma)", lambda r2, s: np.exp(-r2) * (1 + s), lambda z, s: torch.exp(-(z ** 2).sum(1)) * (1 + s))]
cases = [(2, 1, 0, 2, "Watson kappa"), (2, 0, 1, 0, "source psi"), (2, 0.5, 0, 1, "IBP"), (2, 0, 0, 0, "uniform"),
         (3, 0.5, 1, 1, "mixed"), (5, 2, 2, 1.5, "mixed"), (1, 1, 0, 2, "Watson d=1"), (8, 0, 1, 0, "source d=8")]
rows, worst = [], 0.0
for d, a, b, c, name in cases:
    z, s, _ = sample_family(d, a, b, c, NMC, dtype=torch.float64)
    row = dict(d=d, a=a, b=b, c=c, name=name, Z_closed=float(Z_family(d, a, b, c)))
    for tn, fq, ft in tests:
        q, den = quad(d, a, b, c, fq)
        mc = float(ft(z, s).mean()); se = float(ft(z, s).std()) / math.sqrt(NMC)
        row[tn] = dict(mc=mc, se=se, quad=q, rel=abs(mc - q) / abs(q)); worst = max(worst, abs(mc - q) / abs(q))
    row['Z_quad'] = den; row['Z_rel'] = abs(den - row['Z_closed']) / den
    rows.append(row); print(row, flush=True)
json.dump(dict(rows=rows, worst_rel=worst, n_mc=NMC), open(os.path.join(HERE, 'results_e0.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'table_e0.tex'), 'w') as fh:
    fh.write('\\begin{tabular}{lrrrr|rr|rr|rr|cc}\n\\toprule\n')
    fh.write('case & $d$ & $a$ & $b$ & $c$ & \\multicolumn{2}{c|}{$\\mathbb E|z|^2$ (MC / quad)} & '
             '\\multicolumn{2}{c|}{$\\mathbb E\\sigma$} & \\multicolumn{2}{c|}{$\\mathbb E e^{-|z|^2}(1+\\sigma)$} & '
             '$Z$ closed & rel.\\ err.\\\\\n\\midrule\n')
    for r in rows:
        cells = ' & '.join(f"{r[t]['mc']:.4g} & {r[t]['quad']:.4g}" for t, _, _ in tests)
        fh.write(f"{r['name']} & {r['d']} & {r['a']:g} & {r['b']:g} & {r['c']:g} & {cells} & "
                 f"{r['Z_closed']:.6e} & {r['Z_rel']:.1e}\\\\\n")
    fh.write('\\bottomrule\n\\end{tabular}\n')
print('worst MC-vs-quadrature relative discrepancy', worst)
