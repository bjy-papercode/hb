"""E3: small-radius rate of the heat-ball residual (Theorem: 4 R_r / (C1 r^2) -> pointwise FP residual).
Exact log-density v of a Gaussian mixture under an anisotropic OU drift; candidate F = v + eps sin(k.x).
R_r[F] is estimated as R_r[F] - R_r[v] with common random numbers (R_r[v] has mean 0), antithetic Watson pairs.
Output: results_e3.json, e3.png (log-log |R_r| vs r with slope-2 reference)."""
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, sample_source, C1
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import pointwise_residual_v
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
torch.manual_seed(0)
out = []
for d in ([2] if QUICK else [2, 16]):
    g = torch.Generator().manual_seed(5)
    mu = torch.randn(3, d, generator=g, dtype=torch.float64); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
    D = LinearDrift.ou(d, a=np.linspace(0.5, 2.0, d // 2).tolist(), a2=np.linspace(2.0, 0.5, d // 2).tolist())
    k = torch.randn(d, generator=g, dtype=torch.float64); k /= k.norm(); eps = 0.05
    v = lambda x, t: G.logp_score(x, t, D, need_score=False)
    F = lambda x, t: v(x, t) + eps * torch.sin(x @ k) * torch.exp(-2 * t)   # time-dependent perturbation
    def score(fn, x, t):
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True); y = fn(xr, t)
            return torch.autograd.grad(y.sum(), xr)[0]
    h = lambda fn, x, t: (score(fn, x, t) ** 2).sum(1) - (D(x) * score(fn, x, t)).sum(1) - D.div(x)
    t0 = 0.5; x0 = G.sample_tau(1, t0, D)[0]
    pw = float(pointwise_residual_v(F, D, D.div, x0, torch.tensor([t0], dtype=torch.float64), create_graph=False))
    for r in np.geomspace(0.02, math.sqrt(4 * math.pi * t0) * 0.95, 9):
        n = 20000 if QUICK else 200000; vals = []
        for _ in range(5):
            z, s = sample_watson(d, n, dtype=torch.float64); T = t0 - r * r * s
            dM = 0.5 * eps * (torch.sin((x0 + r * z) @ k) + torch.sin((x0 - r * z) @ k)) * torch.exp(-2 * T)       # M_r[F] - M_r[v], antithetic
            zp, sp = sample_source(d, n, dtype=torch.float64); U = torch.rand(n, dtype=torch.float64)
            Y = x0 + (r * U.sqrt())[:, None] * zp; TY = t0 - r * r * U * sp
            dS = C1(d) * r * r / 4 * (h(F, Y, TY) - h(v, Y, TY))
            vals.append(float(eps * torch.sin(x0 @ k)) * math.exp(-2 * t0) - float(dM.mean()) - float(dS.mean()))
        R = float(np.mean(vals)); se = float(np.std(vals) / math.sqrt(len(vals)))
        row = dict(d=d, r=float(r), R=R, se=se, scaled=4 * R / (C1(d) * r * r), pointwise=pw)
        out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e3.json'), 'w'), indent=1)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for d in sorted(set(o['d'] for o in out)):
        rr = [o for o in out if o['d'] == d]
        r = np.array([o['r'] for o in rr]); R = np.abs([o['R'] for o in rr])
        ax[0].loglog(r, R, 'o-', label=f'd={d}')
        ax[0].loglog(r, C1(d) / 4 * abs(rr[0]['pointwise']) * r ** 2, 'k:', alpha=.6)
        ax[1].semilogx(r, [o['scaled'] for o in rr], 'o-', label=f'd={d}')
        ax[1].axhline(rr[0]['pointwise'], ls=':', c='k')
    ax[0].set_xlabel('r'); ax[0].set_ylabel('|R_r|'); ax[0].set_title('heat-ball residual (dotted: $C_1 r^2/4$ x pointwise)')
    ax[1].set_xlabel('r'); ax[1].set_title('$4R_r/(C_1 r^2)$ -> pointwise residual'); ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'e3.png'), dpi=150)
except Exception as e:
    print('plot skipped', e)
