"""E10: the score identity with the source integrated by parts (no derivative of the score), and deterministic
quadrature of the heat-ball integrals.

Score identity (Corollary score-mean):  s(x0) = M_r[s] + S_r[grad h_v],  h_v = |s|^2 - f.s - div f.
Integration by parts in z against psi_0 (psi_0 = 0 on the boundary of E_1, grad_z psi_0 = -z/(2 sigma)):
    S_r[grad h] = d int_0^r int_E1 z/(2 sigma) h(x0 + rho z, tau0 - rho^2 sigma) dz dsigma drho
               = (d r Z_{1/2,0,1}/2) E[ (h(x0+rho z) - h(x0-rho z))/2 * zhat ],  (z,sigma) ~ mu_{1/2,0,1}, rho = r U,
so the score identity needs values of s (and of f, div f) only.

(a) Monte Carlo, exact score of the E4 problem (anisotropic OU with rotation, 3-component Gaussian mixture; exact u, s):
    form "vjp" (Hessian of log u along 2s - f, as in E4) versus form "ibp"; d in {2, 16, 64}, r in {0.3, 1}, N = N' in
    {256, 4096}; relative error, its spread over repetitions, and function evaluations.
(b) Deterministic quadrature instead of sampling.  Every member of the Beta-Gamma family is the law of
        z = sqrt(2 d sigma Y t) zhat,   sigma = e^{-Y}/(4 pi),   Y ~ Gamma(d/2+a+b+1, rate d/2+a-c+1),  t ~ Beta(d/2+a, b+1),
    so E[F] is a product integral over (Y, t, zhat) (and rho):  generalised Gauss-Laguerre in Y, Gauss-Jacobi in t,
    Gauss-Legendre in rho, and for the direction: the trapezoidal rule on the circle (d = 2: fully deterministic) or a
    randomly rotated cross-polytope {+-R e_i} (2d points, exact for polynomials of degree 3 on the sphere; random
    rotations keep it unbiased) for d > 2.  Error of the log-density and score identities vs number of evaluations,
    against Monte Carlo with the exact samplers.
(c) Cost of the vector residual for networks: a scalar network v_theta used through s = grad v_theta (form vjp needs a
    Hessian-vector product of v_theta, form ibp only grad v_theta) and a vector network s_theta (vjp: one VJP; ibp: none);
    ms per training step (loss + backward in theta), d in {16, 64, 256, 1024}.
Output: results_e10.json.   QUICK=1 for a smoke test."""
import sys, os, json, math, time
import numpy as np, torch
from scipy.special import roots_genlaguerre, roots_jacobi, roots_legendre
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import C1, Z_family
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import hb_residual_s
from hbmc.models import VNet
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
PARTS = os.environ.get('E10_PARTS', 'abc')
OUTF = os.path.join(HERE, 'results_e10.json')
DT = torch.float64
torch.manual_seed(0)
out = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for k in PARTS:
    if not (os.environ.get('E10_APPEND') and k in out):
        out[k] = []


def problem(d):
    g = torch.Generator().manual_seed(7)
    mu = torch.randn(3, d, generator=g, dtype=DT); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
    G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
    D = LinearDrift.ou(d, a=0.5, a2=1.5, omega=1.0)
    Bm = torch.block_diag(*D.B)
    v = lambda x, t: G.logp_score(x, t.expand(len(x)) if t.dim() else torch.full((len(x),), float(t), dtype=DT), D, need_score=False)
    s = lambda x, t: G.logp_score(x, t.expand(len(x)) if t.dim() else torch.full((len(x),), float(t), dtype=DT), D)[1]
    t0 = 0.5
    xs, _ = G.sample_tau(2000, t0, D); lp = G.logp_score(xs, torch.full((2000,), t0, dtype=DT), D, need_score=False)
    anchors = {'high': xs[lp.argmax()][None], 'low': xs[lp.argsort()[100]][None]}
    return G, D, Bm, v, s, t0, anchors


# ------------------------------------------------------------------------------------------- (a) Monte Carlo
for d in ([] if 'a' not in PARTS else [2] if QUICK else [2, 16, 64]):
    G, D, Bm, v, s, t0, anchors = problem(d)
    divf = lambda x, t: D.div(x)
    gfs = lambda x, t, sv: sv @ Bm                   # (grad f)^T s for f = B x
    gdf = lambda x, t: torch.zeros_like(x)           # div f is constant
    for lab, x0 in anchors.items():
        tt = torch.tensor([t0], dtype=DT); s0 = s(x0, tt)
        for r in [0.3, 1.0]:
            for N in ([256] if QUICK else [256, 4096]):
                for form in ['vjp', 'ibp']:
                    errs, t_ = [], time.time()
                    for rep in range(4 if QUICK else 20):
                        R = hb_residual_s(lambda x, t: s(x, t), lambda x, t: D(x), divf, x0, tt, r, N=N, Np=N, form=form,
                                          create_graph=False, s0=s0, gradf_T_s=gfs, grad_div_f=gdf)
                        errs.append((R[0].norm() / s0[0].norm()).item())
                    row = dict(d=d, anchor=lab, r=r, N=N, form=form, rel_err_mean=float(np.mean(errs)),
                               rel_err_rms=float(np.sqrt(np.mean(np.square(errs)))), sec=(time.time() - t_) / len(errs),
                               evals_s=(N + 2 * N) if form == 'ibp' else (N + N), vjps=0 if form == 'ibp' else N)
                    out['a'].append(row); print('(a)', row, flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)


# ------------------------------------------------------------------------------------------- (b) quadrature
def family_nodes(d, a, b, c, nY, nt, half=False):
    """Product Gauss rule for the radial / time part of mu_{a,b,c}: returns sigma (m,), R (m,), w (m,), sum w = 1.
    half=True: for integrands that are ODD in |z| (R = sqrt(2 d sigma Y t) times a smooth function), the factor
    sqrt(Y t) is moved into the Gauss weights (Laguerre exponent + 1/2, Jacobi exponent + 1/2); the returned weights
    then already include E[sqrt(Y t)] / sqrt(Y t), so that sum_i w_i F_i is again the expectation of F."""
    k = d / 2.0 + a + b + 1.0; lam = d / 2.0 + a - c + 1.0; al = d / 2.0 + a; be = b + 1.0
    h = 0.5 if half else 0.0
    xY, wY = roots_genlaguerre(nY, k - 1.0 + h); Y = xY / lam; wY = wY / wY.sum()
    xt, wt = roots_jacobi(nt, b, al - 1.0 + h); t = (1 + xt) / 2; wt = wt / wt.sum()   # weight t^{al-1+h}(1-t)^b
    Yg, tg = np.meshgrid(Y, t, indexing='ij'); W = np.outer(wY, wt)
    if half:
        lg = math.lgamma
        EsqY = math.exp(lg(k + 0.5) - lg(k)) / math.sqrt(lam)
        Esqt = math.exp(lg(al + 0.5) + lg(al + be) - lg(al) - lg(al + be + 0.5))
        W = W * EsqY * Esqt / np.sqrt(Yg * tg)
    sig = np.exp(-Yg) / (4 * math.pi); R = np.sqrt(2 * d * sig * Yg * tg)
    return torch.as_tensor(sig.ravel()), torch.as_tensor(R.ravel()), torch.as_tensor(W.ravel())


def directions(d, nd, gen):
    """symmetric direction set (rows) with equal weights: circle trapezoid (d = 2) or rotated cross-polytopes."""
    if d == 2:
        th = 2 * math.pi * (torch.arange(nd, dtype=DT) + torch.rand(1, generator=gen, dtype=DT)) / nd
        return torch.stack([th.cos(), th.sin()], 1)
    out_ = []
    for _ in range(max(1, nd // (2 * d))):
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=DT))
        out_ += [Q.T, -Q.T]
    return torch.cat(out_)


def quad_identities(v, s, D, x0, t0, r, nY, nt, nr, nd, gen):
    """log-density and score identities by product quadrature; returns (v_hat, s_hat, evaluations)."""
    d = x0.shape[1]
    U = directions(d, nd, gen)                                      # (m_d, d)
    # Watson mean (a=1, b=0, c=2)
    sig, R, W = family_nodes(d, 1.0, 0.0, 2.0, nY, nt)
    Z = (R[:, None, None] * U[None]).reshape(-1, d); T = (t0 - r * r * sig)[:, None].expand(-1, len(U)).reshape(-1)
    X = x0 + r * Z
    w = (W[:, None] / len(U)).expand(-1, len(U)).reshape(-1)
    Mv = (w * v(X, T)).sum(); Ms = (w[:, None] * s(X, T)).sum(0)
    ev = len(X)
    # log-density source: S_r[h] = (C1 r^2/4) E_{mu_psi, rho ~ 2 rho/r^2}[h]  (a=0, b=1, c=0), rho by Gauss-Jacobi (weight rho)
    sig, R, W = family_nodes(d, 0.0, 1.0, 0.0, nY, nt)
    xr, wr = roots_jacobi(nr, 0.0, 1.0); rho = torch.as_tensor(r * (1 + xr) / 2); wr = torch.as_tensor(wr / wr.sum())
    Zs = (R[:, None, None] * U[None]).reshape(-1, d); Ts = sig[:, None].expand(-1, len(U)).reshape(-1)
    ws = (W[:, None] / len(U)).expand(-1, len(U)).reshape(-1)
    X = (x0 + rho[:, None, None] * Zs[None]).reshape(-1, d); T = (t0 - rho[:, None] ** 2 * Ts[None]).reshape(-1)
    wtot = (wr[:, None] * ws[None]).reshape(-1)
    sv = s(X, T); h = (sv * sv).sum(1) - (D(X) * sv).sum(1) - D.div(X)
    Sv = C1(d) * r * r / 4 * (wtot * h).sum(); ev += len(X)
    # score source by integration by parts (a=1/2, b=0, c=1), rho uniform (Gauss-Legendre); directions symmetric
    sig, R, W = family_nodes(d, 0.5, 0.0, 1.0, nY, nt, half=True)      # integrand odd in |z|
    xl, wl = roots_legendre(nr); rho = torch.as_tensor(r * (1 + xl) / 2); wl = torch.as_tensor(wl / wl.sum())
    Zs = (R[:, None, None] * U[None]).reshape(-1, d); Zh = U[None].expand(len(R), -1, -1).reshape(-1, d)
    Ts = sig[:, None].expand(-1, len(U)).reshape(-1); ws = (W[:, None] / len(U)).expand(-1, len(U)).reshape(-1)
    X = (x0 + rho[:, None, None] * Zs[None]).reshape(-1, d); T = (t0 - rho[:, None] ** 2 * Ts[None]).reshape(-1)
    wtot = (wl[:, None] * ws[None]).reshape(-1); zh = Zh[None].expand(len(rho), -1, -1).reshape(-1, d)
    sv = s(X, T); h = (sv * sv).sum(1) - (D(X) * sv).sum(1) - D.div(X)
    Ss = d * r * Z_family(d, 0.5, 0.0, 1.0) / 2 * (wtot[:, None] * h[:, None] * zh).sum(0); ev += len(X)
    return float(Mv + Sv), Ms + Ss, ev


def mc_identities(v, s, D, x0, t0, r, N):
    """the same three integrals by Monte Carlo with the exact samplers (antithetic), N samples each."""
    d = x0.shape[1]; tt = torch.tensor([t0], dtype=DT)
    from hbmc.heatball import sample_watson, sample_source
    z, sg = sample_watson(d, N // 2, dtype=DT); T = t0 - r * r * sg
    Mv = 0.5 * (v(x0 + r * z, T) + v(x0 - r * z, T)).mean(); Ms = 0.5 * (s(x0 + r * z, T) + s(x0 - r * z, T)).mean(0)
    zp, sp = sample_source(d, N, dtype=DT); U = torch.rand(N, dtype=DT)
    Y = x0 + (r * U.sqrt())[:, None] * zp; TY = t0 - r * r * U * sp
    sv = s(Y, TY); h = (sv * sv).sum(1) - (D(Y) * sv).sum(1) - D.div(Y)
    Sv = C1(d) * r * r / 4 * h.mean()
    from hbmc.heatball import sample_ibp
    zq, sq, zh = sample_ibp(d, N // 2, dtype=DT); Uq = torch.rand(N // 2, dtype=DT); rho = r * Uq; Tq = t0 - rho ** 2 * sq
    hf = lambda X: (lambda sv_: (sv_ * sv_).sum(1) - (D(X) * sv_).sum(1) - D.div(X))(s(X, Tq))
    Ss = (d * r * Z_family(d, 0.5, 0.0, 1.0) / 2 * 0.5 * (hf(x0 + rho[:, None] * zq) - hf(x0 - rho[:, None] * zq))[:, None] * zh).mean(0)
    return float(Mv + Sv), Ms + Ss, N + N + N


gen = torch.Generator().manual_seed(3)
for d in ([] if 'b' not in PARTS else [2] if QUICK else [int(x) for x in os.environ.get('E10_DIMS_B', '2,16,64').split(',')]):
    G, D, Bm, v, s, t0, anchors = problem(d)
    for lab, x0 in anchors.items():
        tt = torch.tensor([t0], dtype=DT); v0 = float(v(x0, tt)); s0 = s(x0, tt)[0]
        for r in [0.3, 1.0]:
            levels = [(2, 2, 2, 8), (4, 4, 4, 8), (6, 6, 6, 8), (8, 8, 8, 8), (12, 12, 12, 8), (16, 16, 16, 8)] if d == 2 else \
                     ([(4, 4, 4, 2 * d), (8, 8, 8, 2 * d), (8, 8, 8, 8 * d), (8, 8, 8, 32 * d)] if d <= 16 else
                      [(4, 4, 4, 2 * d), (8, 8, 8, 2 * d), (4, 4, 4, 8 * d)])
            if QUICK:
                levels = levels[:2]
            for (nY, nt, nr, nd) in levels:
                reps = 1 if d == 2 else (2 if QUICK else 5)
                ev_, es_ = [], []
                for _ in range(reps):
                    vh, sh, ev = quad_identities(v, s, D, x0, t0, r, nY, nt, nr, nd, gen)
                    ev_.append(abs(vh - v0) / abs(v0)); es_.append(float((sh - s0).norm() / s0.norm()))
                row = dict(d=d, anchor=lab, r=r, method='quadrature', nodes=(nY, nt, nr, nd), evals=ev,
                           rel_err_logu=float(np.sqrt(np.mean(np.square(ev_)))), rel_err_score=float(np.sqrt(np.mean(np.square(es_)))), reps=reps)
                out['b'].append(row); print('(b)', row, flush=True)
                # Monte Carlo with the same number of evaluations
                N = max(4, 2 * (ev // 6))
                ev_, es_ = [], []
                for _ in range(2 if QUICK else 10):
                    vh, sh, evm = mc_identities(v, s, D, x0, t0, r, N)
                    ev_.append(abs(vh - v0) / abs(v0)); es_.append(float((sh - s0).norm() / s0.norm()))
                row = dict(d=d, anchor=lab, r=r, method='monte_carlo', evals=evm,
                           rel_err_logu=float(np.sqrt(np.mean(np.square(ev_)))), rel_err_score=float(np.sqrt(np.mean(np.square(es_)))))
                out['b'].append(row); print('(b)', row, flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)


# ------------------------------------------------------------------------------------------- (c) training cost
class VecNet(torch.nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d + 1, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, d))

    def forward(self, x, t):
        return self.f(torch.cat([x, t.reshape(-1, 1).expand(len(x), 1).to(x.dtype)], 1)) - x


for d in ([] if 'c' not in PARTS else [16] if QUICK else [16, 64, 256, 1024]):
    D = LinearDrift.ou(d, a=1.0, omega=0.5); D.B = D.B.float(); D.dtype = torch.float32
    Bm = torch.block_diag(*D.B)
    x = torch.randn(64, d); tau = torch.rand(64) * 0.9 + 0.1
    for kind in ['scalar', 'vector']:
        torch.manual_seed(0)
        net = VNet(d, 256, 128, 4) if kind == 'scalar' else VecNet(d)
        if kind == 'scalar':
            def s_fn(X, T):
                with torch.enable_grad():
                    Xr = X if X.requires_grad else X.detach().requires_grad_(True)
                    g, = torch.autograd.grad(net(Xr, T).sum(), Xr, create_graph=True)
                return g
        else:
            s_fn = lambda X, T: net(X, T)
        for form in ['vjp', 'ibp']:
            def step():
                net.zero_grad(set_to_none=True)
                R = hb_residual_s(s_fn, lambda X, T: D(X), lambda X, T: D.div(X), x, tau, 0.3, N=16, Np=16, form=form,
                                  gradf_T_s=lambda X, T, sv: sv @ Bm, grad_div_f=lambda X, T: torch.zeros_like(X))
                (R ** 2).sum(1).mean().backward()
            try:
                step(); t_ = time.perf_counter(); reps = 2 if QUICK else 5
                for _ in range(reps):
                    step()
                ms = 1e3 * (time.perf_counter() - t_) / reps
                row = dict(d=d, model=kind, form=form, ms_per_step=ms)
            except RuntimeError as e:
                row = dict(d=d, model=kind, form=form, error=str(e)[:200])
            out['c'].append(row); print('(c)', row, flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)
