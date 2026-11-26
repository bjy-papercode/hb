"""E7: the heat-ball residual as a DIAGNOSTIC for score models, with the true score known (E4 setting:
linear drift with rotation, Gaussian-mixture data, closed-form score).

Imperfect models
  scalar  v = v* + eps * phi,  phi in {global: sin(k.x)(1+tau),  low: bump centred in a low-density region}
  vector  s = s* + eps * J x bump(x)   (rotational, non-gradient; uses the vector/score identity)
  trained DSM networks at several training budgets (d = 2)
For each model: true score error  E|s - s*|^2  (p_tau-weighted over anchors, and on the lowest-10% density anchors)
versus the estimated mean squared heat-ball residual (unbiased two-replica product, scaled to pointwise units
4/(C1 r^2)) and the pointwise FP residual (exact Laplacian, and Hutchinson with one probe).
Cost side: seconds per anchor and the variance of each residual estimator per unit cost.
Output: results_e7.json (+ e7.png).  QUICK=1 for a smoke test."""
import sys, os, json, math, time
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, sample_source, C1, admissible_r
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import hb_residual_v, pointwise_residual_v, _rep
from hbmc.models import VNet
from hbmc import losses
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
torch.manual_seed(0); dt = torch.float64
R_HB, N_HB = 0.3, 16                         # N_HB: the training setting; diagnostics also use larger N
N_DIAG = [16, 256] if QUICK else [16, 256, 2048]
out = dict(config=dict(r=R_HB, N_train=N_HB, N_diag=N_DIAG), rows=[], cost=[], trained=[])


def vec_residual(s_fn, D, x0, tau0, r, N, Np):
    """Vector identity s(x0) = M_r[s] + S_r[grad h(s)], h(s) = |s|^2 - f.s - div f (s need not be a gradient)."""
    B, d = x0.shape
    rr = admissible_r(tau0, torch.full((B,), r, dtype=dt))
    n = N // 2
    z, sg = sample_watson(d, B * n, dtype=dt); R_ = _rep(rr, n); X0 = _rep(x0, n); T = _rep(tau0, n) - R_ ** 2 * sg
    M = (0.5 * (s_fn(X0 + R_[:, None] * z, T) + s_fn(X0 - R_[:, None] * z, T))).view(B, n, d).mean(1)
    zp, sp = sample_source(d, B * Np, dtype=dt); U = torch.rand(B * Np, dtype=dt); R2 = _rep(rr, Np)
    Y = (_rep(x0, Np) + (R2 * U.sqrt())[:, None] * zp).requires_grad_(True); TY = _rep(tau0, Np) - R2 ** 2 * U * sp
    with torch.enable_grad():
        s = s_fn(Y, TY); h = (s * s).sum(1) - (D(Y) * s).sum(1) - D.div(Y)
        gh, = torch.autograd.grad(h.sum(), Y)
    S = (C1(d) * R2 ** 2 / 4)[:, None] * gh
    return (s_fn(x0, tau0) - M - S.view(B, Np, d).mean(1)) * (4 / (C1(d) * rr ** 2))[:, None]


def chunked(f, x, t, k=16):
    """evaluate a residual estimator anchor-chunk by anchor-chunk (memory: N samples per anchor)."""
    return torch.cat([f(x[i:i + k], t[i:i + k]) for i in range(0, len(x), k)])


def score_of(v_fn, x, t):
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        return torch.autograd.grad(v_fn(xr, t).sum(), xr, create_graph=False)[0]


for d in ([2] if QUICK else [2, 16]):
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(3, d, generator=g, dtype=dt); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
    D = LinearDrift.ou(d, a=0.5, a2=1.5, omega=1.0)
    vstar = lambda x, t: G.logp_score(x, t, D, need_score=False)
    sstar = lambda x, t: G.logp_score(x, t, D)[1]
    B = 64 if QUICK else (256 if d == 2 else 128)
    tau = torch.exp(torch.empty(B, dtype=dt).uniform_(math.log(R_HB ** 2 / (4 * math.pi) * 1.05), 0.0))
    x = torch.stack([G.sample_tau(1, float(t), D)[0][0] for t in tau])
    lp = vstar(x, tau); low = lp <= torch.quantile(lp, 0.10)
    c_low = x[lp.argsort()[B // 20]]                                  # a low-density centre
    k = torch.randn(d, generator=g, dtype=dt); k /= k.norm()
    J = torch.zeros(d, d, dtype=dt)
    for j in range(d // 2):
        J[2 * j, 2 * j + 1], J[2 * j + 1, 2 * j] = -1.0, 1.0
    bump = lambda X, c, w=0.6: torch.exp(-((X - c) ** 2).sum(1) / (2 * w * w))
    phis = {'global': lambda X, T: torch.sin(X @ k) * (1 + T), 'low': lambda X, T: 3.0 * bump(X, c_low)}
    s_ref = sstar(x, tau); nrm = (s_ref ** 2).sum(1)
    for kind in ['global', 'low', 'rot']:
        for eps, Nd in [(e_, n_) for e_ in ([0.0, 0.03, 0.3] if QUICK else [0.0, 0.01, 0.03, 0.1, 0.3]) for n_ in N_DIAG]:
            if kind == 'rot':
                s_fn = lambda X, T: sstar(X, T) + eps * (X @ J.T) * bump(X, c_low)[:, None]
                s_m = s_fn(x, tau)
                Rs = [chunked(lambda xx, tt: vec_residual(s_fn, D, xx, tt, R_HB, Nd, Nd), x, tau) for _ in range(2)]
                hb2 = (Rs[0] * Rs[1]).sum(1); pw2 = torch.full((B,), float('nan'), dtype=dt); hu2 = pw2
            else:
                v_fn = (lambda ph: (lambda X, T: vstar(X, T) + eps * ph(X, T)))(phis[kind])
                s_m = score_of(v_fn, x, tau)
                sc = 4 / (C1(d) * admissible_r(tau, torch.full((B,), R_HB, dtype=dt)) ** 2)
                Rs = [sc * chunked(lambda xx, tt: hb_residual_v(v_fn, D, D.div, xx, tt, R_HB, N=Nd, Np=Nd,
                                                                 create_graph=False).detach(), x, tau) for _ in range(2)]
                hb2 = Rs[0] * Rs[1]
                pw2 = pointwise_residual_v(v_fn, D, D.div, x, tau, lap='exact', create_graph=False).detach() ** 2
                hu2 = pointwise_residual_v(v_fn, D, D.div, x, tau, lap='hutch', create_graph=False).detach() ** 2
            err = ((s_m - s_ref) ** 2).sum(1)
            row = dict(d=d, kind=kind, eps=eps, N=Nd,
                       score_rel_mse=float(err.mean() / nrm.mean()), score_rel_mse_low=float(err[low].mean() / nrm[low].mean()),
                       hb_res2=float(hb2.mean()), hb_res2_low=float(hb2[low].mean()),
                       pw_res2=float(pw2.mean()), pw_res2_low=float(pw2[low].mean()),
                       hutch_res2=float(hu2.mean()), hutch_res2_low=float(hu2[low].mean()))
            out['rows'].append(row); print(row, flush=True)
    # ---- cost / variance per unit cost (scalar global perturbation, eps = 0.1) ----
    v_fn = lambda X, T: vstar(X, T) + 0.1 * phis['global'](X, T)
    sc = 4 / (C1(d) * admissible_r(tau, torch.full((B,), R_HB, dtype=dt)) ** 2)
    est = {'heat_ball': lambda: (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=N_HB, Np=N_HB, create_graph=False)).detach() ** 2,
           'heat_ball_ustat': lambda: (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=N_HB, Np=N_HB, create_graph=False)).detach() *
                                      (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=N_HB, Np=N_HB, create_graph=False)).detach(),
           'heat_ball_fd': lambda: (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=N_HB, Np=N_HB, mode='fd', create_graph=False)).detach() ** 2,
           'heat_ball_ustat_N256': lambda: (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=256, Np=256, create_graph=False)).detach() *
                                           (sc * hb_residual_v(v_fn, D, D.div, x, tau, R_HB, N=256, Np=256, create_graph=False)).detach(),
           'pointwise_exact': lambda: pointwise_residual_v(v_fn, D, D.div, x, tau, lap='exact', create_graph=False).detach() ** 2,
           'pointwise_hutch': lambda: pointwise_residual_v(v_fn, D, D.div, x, tau, lap='hutch', create_graph=False).detach() ** 2}
    for name, f in est.items():
        vals = []; t0 = time.perf_counter()
        for _ in range(5 if QUICK else 20):
            vals.append(float(f().mean()))
        sec = (time.perf_counter() - t0) / len(vals)
        row = dict(d=d, estimator=name, mean_sq_residual=float(np.mean(vals)), sd=float(np.std(vals)),
                   sec_per_anchor=sec / B, var_x_cost=float(np.var(vals)) * sec)
        out['cost'].append(row); print('cost', row, flush=True)
    # ---- trained DSM networks at several budgets (d = 2 only) ----
    if d == 2:
        torch.manual_seed(1)
        net = VNet(2, 128, 64, 3); opt = torch.optim.Adam(net.parameters(), 1e-3)
        done = 0
        for budget in ([50, 400] if QUICK else [50, 200, 800, 3200, 8000]):
            for _ in range(budget - done):
                x0 = G.sample0(512).float(); tt = torch.exp(torch.empty(512).uniform_(math.log(1e-3), 0.0))
                L = losses.dsm(net, G, D, x0, tt); opt.zero_grad(); L.backward(); opt.step()
            done = budget
            vn = lambda X, T: net(X.float(), T.float()).double()
            sm = net.score(x.float(), tau.float()).double(); err = ((sm - s_ref) ** 2).sum(1)
            Rs = [sc * chunked(lambda xx, tt: hb_residual_v(vn, D, D.div, xx, tt, R_HB, N=N_DIAG[-1], Np=N_DIAG[-1], create_graph=False).detach(), x, tau, 4) for _ in range(2)]
            hb2 = Rs[0] * Rs[1]
            pw2 = pointwise_residual_v(vn, D, D.div, x, tau, lap='exact', create_graph=False).detach() ** 2
            row = dict(d=d, dsm_steps=budget, score_rel_mse=float(err.mean() / nrm.mean()),
                       score_rel_mse_low=float(err[low].mean() / nrm[low].mean()),
                       hb_res2=float(hb2.mean()), hb_res2_low=float(hb2[low].mean()),
                       pw_res2=float(pw2.mean()), pw_res2_low=float(pw2[low].mean()))
            out['trained'].append(row); print('trained', row, flush=True)
    json.dump(out, open(os.path.join(HERE, 'results_e7.json'), 'w'), indent=1)

rows = [r for r in out['rows'] if r['eps'] > 0 and r['N'] == N_DIAG[-1]]
if len(rows) > 3:
    out['spearman_hb_vs_error'] = float(spearmanr([r['hb_res2'] for r in rows], [r['score_rel_mse'] for r in rows])[0])
    out['spearman_hb_vs_error_low'] = float(spearmanr([r['hb_res2_low'] for r in rows], [r['score_rel_mse_low'] for r in rows])[0])
json.dump(out, open(os.path.join(HERE, 'results_e7.json'), 'w'), indent=1)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind, mk in [('global', 'o'), ('low', 's'), ('rot', '^')]:
        rr = [r for r in rows if r['kind'] == kind]
        ax[0].loglog([r['score_rel_mse'] for r in rr], [max(r['hb_res2'], 1e-12) for r in rr], mk, label=kind)
        ax[1].loglog([r['score_rel_mse_low'] for r in rr], [max(r['hb_res2_low'], 1e-12) for r in rr], mk, label=kind)
    tr = out['trained']
    ax[0].loglog([r['score_rel_mse'] for r in tr], [max(r['hb_res2'], 1e-12) for r in tr], 'k*-', label='DSM nets')
    ax[1].loglog([r['score_rel_mse_low'] for r in tr], [max(r['hb_res2_low'], 1e-12) for r in tr], 'k*-')
    ax[0].set_xlabel('true rel. score MSE'); ax[0].set_ylabel('heat-ball mean sq. residual'); ax[0].legend(fontsize=7)
    ax[1].set_xlabel('true rel. score MSE (lowest 10% density)'); ax[0].set_title('all anchors'); ax[1].set_title('low density')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'e7.png'), dpi=150)
except Exception as e:
    print('plot skipped', e)
