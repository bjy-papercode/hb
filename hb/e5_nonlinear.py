"""E5: nonlinear drift (no closed form), Fokker-Planck grid references.

(a) 1D double well  dX = (X - X^3) dt + dW,  i.e. in cumulative-variance time tau = t/2:
        dx = f dtau + sqrt(2) dW,  f(x) = 2 (x - x^3)          (hbmc.drifts.DW1D)
    INITIAL LAW (stated explicitly, the earlier version did not):  p0 = 0.6 N(-1, 0.25^2) + 0.4 N(1.2, 0.4^2).
    Reference: Scharfetter-Gummel finite volumes on [-4, 4] (N = 3000 cells), Crank-Nicolson, dt = 1e-3;
    grid convergence is reported against N = 1500.
    Check of the score mean-value identity with a NUMERICALLY SUPPLIED source:
        s(x0) = M_r[s] + S_r[grad h_v],  h_v = |s|^2 - f.s - div f,
    where s, h_v, grad h_v are finite differences of log u on the grid, interpolated in (tau, x).
    Error metric (stated explicitly): relative RMSE over anchors x0 ~ p_tau0, i.e. the p_tau0-WEIGHTED
        sqrt(E|s_hat - s|^2 / E|s|^2);  also reported without the source term (S_r omitted) to show it matters.
(b) 2D nonlinear drift: double well + rotation (hbmc.drifts.DWRot, omega = 1.5), asymmetric_gmm initial law,
    grid on [-6, 6]^2 (N = 400; convergence vs N = 200); same check as (a).
(c) Trained-source comparison in 1D (referee request): the SAME scalar network trained by
    ssm | ssm+hb | hb_only | ssm+fp under the double-well forward process, 5 seeds; score error vs the grid
    reference at several tau (p_tau-weighted relative MSE, and on the lowest-10%-density subset).
Output: results_e5.json
QUICK=1 for a smoke test."""
import sys, os, json, math, time
import numpy as np, torch
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, sample_source, C1
from hbmc.fp_grid import FPGrid, GridReference
from hbmc import drifts, datasets2d, losses
from hbmc.linear_gmm import BlockGMM
from hbmc.models import VNet
from hbmc.metrics import score_errors
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
PARTS = os.environ.get('E5_PARTS', 'abc')                     # e.g. E5_PARTS=c: only the trained-source comparison
OUT = os.path.join(HERE, os.environ.get('E5_OUT', 'results_e5.json'))
torch.manual_seed(0); rng = np.random.default_rng(0)
res = {}


def logp0_1d(x):
    x = np.asarray(x, float)
    c = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)
    return np.log(0.6 * c(-1.0, 0.25) + 0.4 * c(1.2, 0.4))


class GridField:
    """log u and its derived fields on a grid, at uniformly spaced snapshot times, interpolated in (tau, x)."""

    def __init__(self, grid, sols, drift_np, div_np):
        self.grid = grid; d = grid.d; h = grid.h
        taus = np.array(sorted(t for t in sols if t > 0)); self.taus = taus
        pts = grid.points(); fx = drift_np(pts, 0.0); dv = div_np(pts)
        F = {k: [] for k in ['V', 'S', 'H', 'GH']}
        for t in taus:
            lu = np.log(np.maximum(sols[t], 1e-300))
            if d == 1:
                S = np.gradient(lu, h)[:, None]
            else:
                LU = lu.reshape(grid.N, grid.N); S = np.stack([g.ravel() for g in np.gradient(LU, h)], 1)
            H = (S ** 2).sum(1) - (fx * S).sum(1) - dv
            GH = np.gradient(H, h)[:, None] if d == 1 else \
                np.stack([g.ravel() for g in np.gradient(H.reshape(grid.N, grid.N), h)], 1)
            F['V'].append(lu); F['S'].append(S); F['H'].append(H); F['GH'].append(GH)
        axes = (taus, grid.c) if d == 1 else (taus, grid.c, grid.c)
        shp = (len(taus),) + (grid.N,) * d
        mk = lambda A: RegularGridInterpolator(axes, A.reshape(shp), bounds_error=False, fill_value=None)
        self.f = {k: (mk(np.stack(v)) if k in ('V', 'H') else [mk(np.stack(v)[..., j]) for j in range(d)])
                  for k, v in F.items()}

    def __call__(self, key, x, t):
        q = np.concatenate([np.asarray(t, float).reshape(-1, 1) * np.ones((len(x), 1)), np.asarray(x, float)], 1)
        f = self.f[key]
        return f(q) if not isinstance(f, list) else np.stack([g(q) for g in f], 1)


def identity_check(field, anchors, t0, r, n_mc):
    d = anchors.shape[1]; out = dict(err_full=[], err_nosrc=[], ref=[], v_resid=[])
    for x0 in anchors:
        z, s = sample_watson(d, n_mc // 2, dtype=torch.float64); z, s = z.numpy(), s.numpy()
        T = t0 - r * r * s
        Ms = 0.5 * (field('S', x0 + r * z, T) + field('S', x0 - r * z, T)).mean(0)
        Mv = 0.5 * (field('V', x0 + r * z, T) + field('V', x0 - r * z, T)).mean()
        zp, sp = sample_source(d, n_mc, dtype=torch.float64); zp, sp = zp.numpy(), sp.numpy()
        U = rng.random(n_mc); Y = x0 + (r * np.sqrt(U))[:, None] * zp; TY = t0 - r * r * U * sp
        c = C1(d) * r * r / 4
        Sg = c * field('GH', Y, TY).mean(0); Sh = c * field('H', Y, TY).mean()
        s_ref = field('S', x0[None], t0)[0]; v_ref = field('V', x0[None], t0)[0]
        out['err_full'].append(((Ms + Sg - s_ref) ** 2).sum()); out['err_nosrc'].append(((Ms - s_ref) ** 2).sum())
        out['ref'].append((s_ref ** 2).sum()); out['v_resid'].append(v_ref - Mv - Sh)
    E = lambda k: float(math.sqrt(np.mean(out[k]) / np.mean(out['ref'])))
    return dict(rel_rmse_with_source=E('err_full'), rel_rmse_without_source=E('err_nosrc'),
                v_identity_residual_mean=float(np.mean(out['v_resid'])),
                v_identity_residual_sd=float(np.std(out['v_resid'])), n_anchor=len(anchors), n_mc=n_mc, r=r, tau0=t0)


# ---------------- (a) 1D double well ----------------
D1 = drifts.make('dw1d')
f1 = lambda x, t: 2 * (x - x ** 3)
div1 = lambda x: (2 - 6 * x[:, 0] ** 2)
res['a'] = dict(initial_law='0.6 N(-1,0.25^2) + 0.4 N(1.2,0.4^2)', drift='f = 2(x-x^3) in tau = t/2', checks=[])
dt = 1e-3
for t0, r in ([] if 'a' not in PARTS else [(0.5, 0.5)] if QUICK else [(0.1, 0.5), (0.5, 0.5), (0.5, 1.0)]):
    win = [round(k * dt, 10) for k in range(int(round((t0 - r * r / (4 * math.pi)) / dt)) - 1, int(round(t0 / dt)) + 1)]
    fields = {}
    for N in ([1500] if QUICK else [3000, 1500]):
        g = FPGrid(1, 4.0, N, f1)
        sols = g.solve(np.exp(logp0_1d(g.c)), win, dt=dt)
        fields[N] = (g, sols, GridField(g, sols, f1, div1))
    Nf = max(fields); g, sols, F = fields[Nf]
    anchors = g.sample(sols[min(sols, key=lambda t: abs(t - t0))], 10 if QUICK else 40, rng)
    row = identity_check(F, anchors, t0, r, 8192 if QUICK else 65536)
    if len(fields) > 1:
        Fc = fields[min(fields)][2]
        sf, sc = F('S', anchors, t0), Fc('S', anchors, t0)
        row['grid_convergence_rel_score_diff_N1500_vs_N3000'] = float(np.sqrt(((sf - sc) ** 2).sum() / (sf ** 2).sum()))
    res['a']['checks'].append(row); print('(a)', row, flush=True)

# ---------------- (b) 2D double well + rotation ----------------
D2 = drifts.make('dw_rot')
f2 = lambda x, t: D2(torch.as_tensor(x)).numpy()
div2 = lambda x: D2.div(torch.as_tensor(x)).numpy()
G2 = datasets2d.get('asymmetric_gmm')
lp0_2 = lambda x: G2.logp_score(torch.as_tensor(x), torch.zeros(len(x), dtype=torch.float64), drifts.make('ou'),
                                need_score=False).numpy()
res['b'] = dict(drift='DWRot omega=1.5: f = -(|x|^2/4-1)x + omega J x/(1+|x|^2/4)', initial='asymmetric_gmm', checks=[])
for t0, r in ([] if 'b' not in PARTS else [(0.5, 0.5)] if QUICK else [(0.2, 0.5), (0.5, 0.5)]):
    win = [round(k * dt, 10) for k in range(int(round((t0 - r * r / (4 * math.pi)) / dt)) - 1, int(round(t0 / dt)) + 1)]
    fields = {}
    for N in ([150] if QUICK else [400, 200]):
        g = FPGrid(2, 6.0, N, f2)
        sols = g.solve(np.exp(lp0_2(g.points())), win, dt=dt)
        fields[N] = (g, sols, GridField(g, sols, f2, div2))
    Nf = max(fields); g, sols, F = fields[Nf]
    anchors = g.sample(sols[min(sols, key=lambda t: abs(t - t0))], 10 if QUICK else 40, rng)
    row = identity_check(F, anchors, t0, r, 8192 if QUICK else 32768)
    if len(fields) > 1:
        Fc = fields[min(fields)][2]
        sf, sc = F('S', anchors, t0), Fc('S', anchors, t0)
        row['grid_convergence_rel_score_diff_N200_vs_N400'] = float(np.sqrt(((sf - sc) ** 2).sum() / (sf ** 2).sum()))
    res['b']['checks'].append(row); print('(b)', row, flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

# ---------------- (c) trained-source comparison in 1D ----------------
tau_min, tau_max = 1e-3, 1.0
eval_taus = [0.05, 0.2, 0.5, 1.0]
snap = sorted(set(np.round(np.geomspace(tau_min, tau_max, 80), 3).tolist()) | set(eval_taus))
g = FPGrid(1, 4.0, 3000 if not QUICK else 1500, f1)
sols = g.solve(np.exp(logp0_1d(g.c)), snap, dt=dt)
REF = GridReference(g, sols)
snaps = np.array(sorted(t for t in sols if t > 0))
bank = {t: torch.as_tensor(g.sample(sols[t], 20000, rng)).float() for t in snaps}
logp0_t = lambda x: torch.as_tensor(logp0_1d(x.double().numpy()[:, 0])).float()


def batch(n, tau_lo=None):
    ok = snaps[snaps >= (tau_lo or 0)]
    ti = rng.integers(0, len(ok), n); taus = ok[ti]
    x = torch.stack([bank[t][rng.integers(0, 20000)] for t in taus])
    return x, torch.as_tensor(taus).float()


class D1t:                                     # torch drift for the losses
    linear = False
    def __call__(self, x, tau=None): return 2 * (x - x ** 3)
    def div(self, x, tau=None): return (2 - 6 * x ** 2)[:, 0]


Dt = D1t()
# 'ssm*3.3' = ssm with 3.3x the steps (equal wall clock with ssm+hb):  E5_PARTS=c E5C_METHODS='ssm*3.3' E5_OUT=...
methods = os.environ.get('E5C_METHODS', 'ssm,ssm+hb,hb_only,ssm+fp').split(',')
seeds = [0] if QUICK else [0, 1, 2, 3, 4]
steps = 600 if QUICK else 4000
r_hb, lam = 0.3, 0.1
res['c'] = dict(config=dict(steps=steps, batch=256, anchors=64, r=r_hb, lam=lam, hb_N=16, hb_Np=16,
                            model='VNet(1, hidden=64, ted=32, blocks=2)', weighting='p_tau-weighted rel. MSE'),
                runs=[])
for m in methods:
    mult = float(m.split('*')[1]) if '*' in m else 1.0; mb = m.split('*')[0]; nsteps = int(round(steps * mult))
    for seed in seeds:
        torch.manual_seed(seed)
        net = VNet(1, 64, 32, 2); opt = torch.optim.Adam(net.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps, eta_min=5e-5)
        t_start = time.time()
        for it in range(1, nsteps + 1):
            L = 0.0
            if mb.startswith('ssm'):
                x, t = batch(256); L = losses.ssm(net, x, t)
            else:
                xd = torch.as_tensor(np.concatenate([rng.normal(-1, .25, 192), rng.normal(1.2, .4, 128)])).float()[:, None]
                xd = torch.cat([xd, xd + 0.5 * torch.randn_like(xd)])
                L = losses.bnd(net, xd, logp0_t(xd))
            w = lam * min(1.0, it / (0.1 * nsteps)) if mb.startswith('ssm') else 1.0
            if 'hb' in mb:
                xa, ta = batch(64, tau_lo=r_hb ** 2 / (4 * math.pi)); L = L + w * losses.hb(net, Dt, xa, ta, r_hb)
            if 'fp' in mb:
                xa, ta = batch(64); L = L + w * losses.fp(net, Dt, xa, ta)
            opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step(); sch.step()
        el = time.time() - t_start
        row = dict(method=m, seed=seed, steps=nsteps, train_s=el, ms_per_step=1e3 * el / nsteps, score={})
        for t in eval_taus:
            x = torch.as_tensor(g.sample(sols[min(sols, key=lambda s_: abs(s_ - t))], 4000, np.random.default_rng(7))).float()
            s_ref = torch.as_tensor(REF.score(x.double().numpy(), t)).float(); lp = torch.as_tensor(REF.logp(x.double().numpy(), t))
            row['score'][f'{t:g}'] = score_errors(net.score(x, torch.full((len(x),), t)), s_ref, lp)
        res['c']['runs'].append(row)
        print('(c)', m, seed, {k: round(v['rel_mse'], 4) for k, v in row['score'].items()}, f"{row['ms_per_step']:.1f} ms", flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
summ = {}
for m in methods:
    rr = [r_ for r_ in res['c']['runs'] if r_['method'] == m]
    summ[m] = {t: dict(rel_mse=f"{np.mean([r_['score'][t]['rel_mse'] for r_ in rr]):.4f} +- {np.std([r_['score'][t]['rel_mse'] for r_ in rr]):.4f}",
                       low10=f"{np.mean([r_['score'][t]['rel_mse_low'] for r_ in rr]):.4f} +- {np.std([r_['score'][t]['rel_mse_low'] for r_ in rr]):.4f}")
               for t in rr[0]['score']}
    summ[m]['ms_per_step'] = float(np.mean([r_['ms_per_step'] for r_ in rr]))
res['c']['summary'] = summ
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(summ, indent=1))
