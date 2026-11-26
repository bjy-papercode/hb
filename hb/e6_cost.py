"""E6: cost and accuracy against alternatives, at equal work.

(a) E4 problem (anisotropic OU + rotation, Gaussian-mixture data; u in closed form, NONZERO source h_u = -div(f u)).
    Estimators of u(x0, tau0), all derivative-free (values of u and f only):
      HB    heat ball:  M_r[u] (antithetic Watson) + S_r[-div(f u)] (antithetic IBP, mu_{1/2,0,1})
      SG    Gaussian semigroup / Duhamel at matched mean lag delta = C1 r^2 / 4:
              u = E u(x0 + sqrt(2 delta) G, tau0 - delta) + delta E[-div(fu)](x0 + sqrt(2 s) G, tau0 - s), s ~ U(0, delta),
            the divergence by antithetic Stein IBP  E div F(x + aG) = E[(F(x+aG) - F(x-aG)).G] / (2a)
      CYL   bounding-cylinder importance sampling for M_r (uniform on B_Rmax x [0, 1/(4 pi)]), same S_r as HB
    Reported: relative RMSE, wall-clock per estimate, and the work-normalised error  RMSE^2 x time  (lower = better).
(b) Bounded domain, Dirichlet: heat equation on the half-space {x_1 > 0}, u = 0 on x_1 = 0 (exact by images).
    The heat ball needs only interior values when r R_max < dist(x0, boundary): unbiased, no boundary treatment.
    The free-space semigroup estimator must extend u (here by 0) and is biased; we report its bias vs delta.
Output: results_e6.json.  QUICK=1 for a smoke test."""
import sys, os, json, math, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, R_max, C1, sample_family, sample_ibp, Z_family
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import hb_residual_u, _watson_mean
from scipy.special import gammaln
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
torch.manual_seed(0)
out = {'a': [], 'b': []}
dt64 = torch.float64


def sg_estimate(u, f, x0, t0, delta, N):
    d = x0.shape[1]; n = N // 2
    G = torch.randn(n, d, dtype=dt64)
    M = 0.5 * (u(x0 + math.sqrt(2 * delta) * G, torch.full((n,), t0 - delta, dtype=dt64)) +
               u(x0 - math.sqrt(2 * delta) * G, torch.full((n,), t0 - delta, dtype=dt64))).mean()
    s = torch.rand(n, dtype=dt64) * delta; a = torch.sqrt(2 * s)[:, None]; G2 = torch.randn(n, d, dtype=dt64)
    T = t0 - s
    F = lambda X: f(X) * u(X, T)[:, None]
    div = ((F(x0 + a * G2) - F(x0 - a * G2)) * G2).sum(1) / (2 * a[:, 0])
    return M + delta * (-div).mean()


def cyl_watson(u, x0, t0, r, N):
    d = x0.shape[1]; R = R_max(d)
    g = torch.randn(N, d, dtype=dt64); g = g / g.norm(dim=1, keepdim=True)
    z = g * (R * torch.rand(N, 1, dtype=dt64) ** (1 / d))
    s = torch.rand(N, dtype=dt64) / (4 * math.pi)
    inside = (z ** 2).sum(1) <= -2 * d * s * torch.log(4 * math.pi * s)
    logS = (d / 2) * math.log(math.pi) - gammaln(d / 2 + 1) + d * math.log(R) - math.log(4 * math.pi)
    kappa = (z ** 2).sum(1) / (4 * s ** 2)
    vals = torch.zeros(N, dtype=dt64)
    if inside.any():
        vals[inside] = kappa[inside] * u(x0 + r * z[inside], t0 - r * r * s[inside])
    return math.exp(logS) * vals.mean()


def s_ibp(u, f, x0, t0, r, Np):
    """S_r[-div(f u)] by antithetic IBP (identical to the source part of hbmc.residuals.hb_residual_u)."""
    d = x0.shape[1]
    zq, sq, zh = sample_ibp(d, Np, dtype=dt64); rho = r * torch.rand(Np, dtype=dt64)
    T = t0 - rho ** 2 * sq
    G = lambda X: f(X, T) * u(X, T)[:, None]
    g = 0.5 * ((G(x0 + rho[:, None] * zq) - G(x0 - rho[:, None] * zq)) * zh).sum(1)
    return (-d * r * Z_family(d, 0.5, 0.0, 1.0) / 2.0 * g).mean()


def timed(fn, reps):
    vals = []; t = time.perf_counter()
    for _ in range(reps):
        vals.append(float(fn()))
    return np.array(vals), (time.perf_counter() - t) / reps


# ---------------- (a) ----------------
for d in ([2, 16] if QUICK else [2, 16, 128]):
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(3, d, generator=g, dtype=dt64); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    Gm = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
    D = LinearDrift.ou(d, a=0.5, a2=1.5, omega=1.0)
    t0 = 0.5
    xs, _ = Gm.sample_tau(2000, t0, D); lp = Gm.logp_score(xs, torch.full((2000,), t0, dtype=dt64), D, need_score=False)
    for lab, x0 in {'high': xs[lp.argmax()][None], 'low': xs[lp.argsort()[100]][None]}.items():
        tt = torch.tensor([t0], dtype=dt64)
        lu0 = float(Gm.logp_score(x0, tt, D, need_score=False))
        u = lambda X, T: torch.exp(Gm.logp_score(X, T.expand(len(X)) if T.numel() == 1 else T, D, need_score=False) - lu0)
        for r in [0.3, 1.0]:
            delta = C1(d) * r * r / 4
            for N in ([1024] if QUICK else [1024, 8192]):
                reps = 5 if QUICK else (30 if d < 128 else 10)
                hb, t_hb = timed(lambda: 1 - hb_residual_u(u, D, x0, tt, r, N=N, Np=N, u0=torch.ones(1, dtype=dt64))[0], reps)
                sg, t_sg = timed(lambda: sg_estimate(u, D, x0, t0, delta, 2 * N), reps)
                cy, t_cy = timed(lambda: cyl_watson(u, x0, t0, r, N) + s_ibp(u, D, x0, t0, r, N), reps)
                row = dict(d=d, anchor=lab, r=r, delta=delta, N=N, reps=reps)
                for k, v, t in [('HB', hb, t_hb), ('SG', sg, t_sg), ('CYL', cy, t_cy)]:
                    rm = float(np.sqrt(np.mean((v - 1) ** 2)))
                    row[k] = dict(rel_rmse=rm, bias=float(np.mean(v - 1)), sec=t, work_norm=rm * rm * t)
                out['a'].append(row); print('(a)', json.dumps(row), flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e6.json'), 'w'), indent=1)
# note: CYL = cylinder IS for the Watson part + the same IBP source estimator as HB (only M_r differs).

# ---------------- (b) bounded domain ----------------
# Half-space {x_1 > 0}, Dirichlet u = 0; u = heat kernel started at y (image at y*), anchor close to the wall.
for d in [2, 16]:
    y = torch.zeros(d, dtype=dt64); y[0] = 0.5; ys = y.clone(); ys[0] = -0.5; ts = 0.05
    def u(X, T):
        T = T.expand(len(X)) if T.numel() == 1 else T
        v = 2 * (T + ts)
        k = lambda c: torch.exp(-((X - c) ** 2).sum(1) / (2 * v) - d / 2 * torch.log(2 * math.pi * v))
        return k(y) - k(ys)
    def u_ext(X, T):                     # what a free-space estimator must do: extend by zero outside
        return torch.where(X[:, 0] > 0, u(X, T), torch.zeros(len(X), dtype=dt64))
    dist = 0.3
    x0 = torch.zeros(1, d, dtype=dt64); x0[0, 0] = dist; t0 = 0.1
    u0 = float(u(x0, torch.tensor([t0], dtype=dt64)))
    r_ok = dist / R_max(d)                                   # heat ball inside the domain
    r_time = 0.999 * math.sqrt(4 * math.pi * (t0 + ts))      # heat ball inside the time range of u
    for frac in [0.25, 0.5, 0.95]:
        r = min(frac * r_ok, r_time)
        for lag_name, delta in [('matched_mean_lag', C1(d) * r * r / 4), ('matched_max_lag', r * r / (4 * math.pi))]:
            hb, sg = [], []
            for _ in range(5 if QUICK else 20):
                hb.append(float(_watson_mean(u_ext, x0, torch.tensor([t0], dtype=dt64), torch.tensor([r], dtype=dt64), 65536)) / u0 - 1)
                G = torch.randn(65536, d, dtype=dt64)
                sg.append(float(u_ext(x0 + math.sqrt(2 * delta) * G, torch.tensor([t0 - delta], dtype=dt64)).mean()) / u0 - 1)
            row = dict(d=d, dist=dist, r=r, r_over_r_inside=r / r_ok, sg_lag=lag_name, delta=delta,
                       HB_bias=float(np.mean(hb)), HB_se=float(np.std(hb) / math.sqrt(len(hb))),
                       SG_bias=float(np.mean(sg)), SG_se=float(np.std(sg) / math.sqrt(len(sg))),
                       SG_leak_prob=float(0.5 * math.erfc(dist / math.sqrt(4 * delta))))
            out['b'].append(row); print('(b)', row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e6.json'), 'w'), indent=1)
