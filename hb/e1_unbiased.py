"""E1: unbiasedness and N^{-1/2} convergence of the Watson mean, d in {2,16,128,1024}.
Test functions: (a) backward-caloric polynomial |z|^2 - 2 d sigma (exact mean 0);
(b) heat flow (zero drift) of a Gaussian mixture, M_r[u] = u(x0, tau0) exactly.
Plain and antithetic sampling.  Two radii: fixed r = 0.5 and dimension-scaled r = 0.5 sqrt(2/d) (the spatial
extent of the heat ball grows like r sqrt(d), so at fixed r the relative variance of u-values explodes with d;
the scaled radius keeps r |z| = O(1)).  Output: results_e1.json (+ figure e1.png)."""
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson
from hbmc.linear_gmm import LinearDrift, BlockGMM
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
Ns = [16, 64, 256, 1024, 4096, 16384]; reps = 200
out = []
for d in [2, 16, 128, 1024]:
    gen = torch.Generator().manual_seed(d)
    mu = torch.randn(3, d, generator=gen, dtype=torch.float64); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2])); D = LinearDrift.ou(d, a=0.0)
    x0 = G.sample_tau(1, 0.3, D)[0]; t0 = 0.3
    lu0 = float(G.logp_score(x0, torch.tensor([t0], dtype=torch.float64), D, need_score=False))  # work with u/u0 (no underflow in high d)
    for r, rlab in [(0.5, 'fixed'), (0.5 * math.sqrt(2 / d), 'scaled')]:
     for N in Ns:
        if d >= 128 and N > 4096:
            continue
        ep, eu, eua = [], [], []
        for _ in range(reps if d < 128 else 40):
            z, s = sample_watson(d, N, dtype=torch.float64)
            ep.append(float(((z ** 2).sum(1) - 2 * d * s).mean()))
            T = t0 - r * r * s
            up = torch.exp(G.logp_score(x0 + r * z, T, D, need_score=False) - lu0)
            um = torch.exp(G.logp_score(x0 - r * z, T, D, need_score=False) - lu0)
            eu.append(float(up.mean()) - 1); eua.append(float(0.5 * (up + um)[: N // 2].mean()) - 1)
        row = dict(d=d, r=r, radius=rlab, N=N, poly_mean=float(np.mean(ep)), poly_rmse=float(np.sqrt(np.mean(np.square(ep)))),
                   heat_bias=float(np.mean(eu)), heat_rmse=float(np.sqrt(np.mean(np.square(eu)))),
                   heat_rmse_antithetic=float(np.sqrt(np.mean(np.square(eua)))), reps=len(ep))
        out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e1.json'), 'w'), indent=1)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for d in sorted(set(r['d'] for r in out)):
        rr = [r for r in out if r['d'] == d and r['radius'] == 'scaled']
        rf = [r for r in out if r['d'] == d and r['radius'] == 'fixed']
        ax[0].loglog([r['N'] for r in rr], [r['poly_rmse'] for r in rr], 'o-', label=f'd={d}')
        l, = ax[1].loglog([r['N'] for r in rr], [r['heat_rmse'] for r in rr], 'o-', label=f'd={d}')
        ax[1].loglog([r['N'] for r in rr], [r['heat_rmse_antithetic'] for r in rr], 's--', alpha=.6, c=l.get_color())
        ax[1].loglog([r['N'] for r in rf], [r['heat_rmse'] for r in rf], ':', alpha=.5, c=l.get_color())
    for a_ in ax:
        n = np.array(Ns); a_.loglog(n, 2 * n ** -0.5, 'k:', label='$N^{-1/2}$'); a_.set_xlabel('N'); a_.legend(fontsize=7)
    ax[0].set_title('$|z|^2-2d\\sigma$ (exact mean 0)'); ax[1].set_title('heat flow, r=0.5(2/d)^{1/2} (dashed: antithetic; dotted: fixed r=0.5)', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'e1.png'), dpi=150)
except Exception as e:
    print('plot skipped', e)
