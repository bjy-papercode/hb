"""E4: evaluation of u, log u and the score by the heat-ball identities under a linear drift with NONZERO
source (anisotropic OU with rotation), d in {2, 16, 128}, against closed-form values.
  u     : u(x0) = M_r[u] + S_r[-div(f u)]              source by antithetic integration by parts (values only)
  log u : v(x0) = M_r[v] + S_r[h_v]                    h_v from the exact score
  score : s(x0) = M_r[s] + S_r[grad h_v],  grad h_v = H(2s - f) - B^T s   (B = grad f; div f constant)
Anchors in a high- and a low-density region; relative error vs N and r (mean over repetitions).
Output: results_e4.json"""
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, sample_source, C1
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import hb_residual_u
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
torch.manual_seed(0)
out = []
for d in ([2] if QUICK else [2, 16, 128]):
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(3, d, generator=g, dtype=torch.float64); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
    D = LinearDrift.ou(d, a=0.5, a2=1.5, omega=1.0)
    Bm = torch.block_diag(*D.B)
    v = lambda x, t: G.logp_score(x, t, D, need_score=False)
    sc = lambda x, t: G.logp_score(x, t, D)[1]
    def grad_h(x, t):
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True); s = sc(xr, t)
            w = (2 * s - D(xr)).detach()
            Hw = torch.autograd.grad((s * w).sum(), xr)[0]
        return Hw - s.detach() @ Bm
    t0 = 0.5
    xs, _ = G.sample_tau(2000, t0, D); lp = v(xs, torch.full((2000,), t0, dtype=torch.float64))
    anchors = {'high': xs[lp.argmax()][None], 'low': xs[lp.argsort()[100]][None]}
    for lab, x0 in anchors.items():
        tt = torch.tensor([t0], dtype=torch.float64)
        v0 = float(v(x0, tt)); s0 = sc(x0, tt)[0]
        for r in [0.3, 1.0]:
            for N in [256, 4096]:
                reps = 2 if QUICK else (10 if d < 128 else 4)
                eu, ev, es = [], [], []
                for _ in range(reps):
                    Ru = hb_residual_u(lambda x, t: torch.exp(v(x, t) - v0), D, x0, tt, r, N=N, Np=N)
                    eu.append(abs(float(Ru)))
                    nn_ = N // 2
                    z, s = sample_watson(d, nn_, dtype=torch.float64); T = t0 - r * r * s
                    Mv = 0.5 * (v(x0 + r * z, T) + v(x0 - r * z, T)).mean()
                    Ms = 0.5 * (sc(x0 + r * z, T) + sc(x0 - r * z, T)).mean(0)
                    zp, sp = sample_source(d, N, dtype=torch.float64); U = torch.rand(N, dtype=torch.float64)
                    Y = x0 + (r * U.sqrt())[:, None] * zp; TY = t0 - r * r * U * sp
                    sY = sc(Y, TY); hv = (sY ** 2).sum(1) - (D(Y) * sY).sum(1) - D.div(Y)
                    ev.append(abs(float(Mv + C1(d) * r * r / 4 * hv.mean()) - v0) / abs(v0))
                    es.append(float((Ms + C1(d) * r * r / 4 * grad_h(Y, TY).mean(0) - s0).norm() / s0.norm()))
                row = dict(d=d, anchor=lab, r=r, N=N, rel_err_u=float(np.mean(eu)), rel_err_logu=float(np.mean(ev)),
                           rel_err_score=float(np.mean(es)), reps=reps)
                out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e4.json'), 'w'), indent=1)
