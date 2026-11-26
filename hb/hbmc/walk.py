"""
hbmc.walk -- heat-ball random walks for the (linear) Fokker--Planck density, with a gauge transform.

Equation (cumulative-variance time tau):  d_tau u = Lap u - div(f u),  f = -grad Phi + f_rot,  div f_rot = 0.

Gauge transform  u = exp(-Phi/2) phi  turns it into
    d_tau phi = Lap phi - div(f_rot phi) - V phi,
    V = |grad Phi|^2/4 - Lap Phi/2 - f_rot.grad Phi/2 - d_tau Phi/2 :
the curl-free part of the drift becomes a zero-order potential V (absorption V>0 / creation V<0), the
divergence-free part a pure transport in divergence form.  Every term is linear in phi, so the heat-ball
identity  phi(x0,t0) = M_r[phi] + S_r[-V phi] + S_r[-div(f_rot phi)]  can be applied recursively with one sample
per level: a random walk with multiplicative weights.  At each step, with radius r = c sqrt(4 pi tau):
    Watson branch (prob pM):  (x, tau) -> (x + r z, tau - r^2 sigma),         (z,sigma) ~ kappa,     weight 1/pM
    potential branch (pV):    (x, tau) -> (x + r sqrt(U) z', tau - r^2 U s'), (z',s') ~ mu_psi,     weight -(C1 r^2/4) V / pV
    transport branch (pG):    (x, tau) -> (x + s rho z, tau - rho^2 sigma),  (z,sigma) ~ mu_{1/2,0,1}, rho = rU, s = +-1,
                              weight -(d r Z_{1/2,0,1}/2) s (f_rot . zhat) / pG    (Proposition "antithetic IBP")
The walk stops when tau <= eps and returns the terminal value  phi = exp(Phi/2) u_term(x, tau).
Estimator:  u(x0,t0) = exp(-Phi(x0,t0)/2) E[ W * phi_term ],  unbiased up to the terminal approximation.
Without the gauge (mode="direct"): Phi = 0, f_rot := f, V = 0 (the full drift goes through the transport branch).

Gradients (score) by common random numbers: the random stream (branch choices, offsets, termination) does not
depend on x0, so u_hat(x0) is a smooth function of x0 for fixed randomness and central differences with the same
seed give an unbiased-up-to-O(h^2) estimate of grad u; the score is the ratio of means.
"""
import math
import numpy as np
import torch
from .heatball import sample_watson, sample_source, sample_ibp, C1, Z_family

DT = torch.float64


class Gauge:
    """Phi(x,t), grad Phi, Lap Phi, d_t Phi and f_rot(x,t) as callables on (n,d) float64 tensors and (n,) tau."""

    def __init__(self, Phi, gradPhi, lapPhi, dtPhi, frot, d=2, name=''):
        self.Phi, self.gradPhi, self.lapPhi, self.dtPhi, self.frot, self.d, self.name = Phi, gradPhi, lapPhi, dtPhi, frot, d, name

    def V(self, x, t):
        g = self.gradPhi(x, t)
        return (g * g).sum(1) / 4 - self.lapPhi(x, t) / 2 - (self.frot(x, t) * g).sum(1) / 2 - self.dtPhi(x, t) / 2

    def drift(self, x, t):
        return -self.gradPhi(x, t) + self.frot(x, t)

    def direct(self):
        """the same drift without gauge transform: Phi = 0, all of f in the transport branch."""
        z = lambda x, t: torch.zeros(len(x), dtype=x.dtype)
        return Gauge(z, lambda x, t: torch.zeros_like(x), z, z, self.drift, self.d, self.name + '/direct')

    def parts(self, which):
        """'grad': keep only -grad Phi;  'rot': keep only f_rot."""
        z = lambda x, t: torch.zeros(len(x), dtype=x.dtype)
        if which == 'grad':
            return Gauge(self.Phi, self.gradPhi, self.lapPhi, self.dtPhi, lambda x, t: torch.zeros_like(x), self.d, self.name + '/grad')
        if which == 'rot':
            return Gauge(z, lambda x, t: torch.zeros_like(x), z, z, self.frot, self.d, self.name + '/rot')
        return self


def J(x):
    return torch.stack([-x[:, 1], x[:, 0]], 1)


def gauge_linear(a1, a2, omega):
    """f = -diag(a1,a2) x + omega J x  (block of hbmc.linear_gmm.LinearDrift.ou(2, a1, omega, a2))."""
    A = torch.tensor([a1, a2], dtype=DT)
    return Gauge(lambda x, t: 0.5 * (A * x * x).sum(1), lambda x, t: A * x, lambda x, t: torch.full((len(x),), float(A.sum()), dtype=DT),
                 lambda x, t: torch.zeros(len(x), dtype=DT), lambda x, t: omega * J(x), 2, f'linear({a1},{a2},{omega})')


def gauge_dwrot(omega=1.5):
    """hbmc.drifts.DWRot:  f = -(|x|^2/4-1) x + omega J x/(1+|x|^2/4),  Phi = |x|^4/16 - |x|^2/2."""
    r2 = lambda x: (x * x).sum(1)
    return Gauge(lambda x, t: r2(x) ** 2 / 16 - r2(x) / 2, lambda x, t: (r2(x) / 4 - 1)[:, None] * x,
                 lambda x, t: r2(x) - 2.0, lambda x, t: torch.zeros(len(x), dtype=DT),
                 lambda x, t: omega * J(x) / (1 + r2(x) / 4)[:, None], 2, f'dwrot({omega})')


def gauge_networks(phi_net, psi_net):
    """Drift of the original 2D forward model, f = grad phi_net + perp-grad psi_net (networks.PhiNet / PsiNet):
    Phi = -phi_net, f_rot = (-d_y psi, d_x psi); derivatives by automatic differentiation (float32 networks)."""
    def _phi(x, t):
        return phi_net(x.float(), t.float().reshape(-1).expand(len(x)))

    def _grads(x, t, order):
        with torch.enable_grad():
            xr = x.detach().float().requires_grad_(True); tr = t.detach().float().reshape(-1).expand(len(x)).clone().requires_grad_(True)
            p = phi_net(xr, tr)
            gx, gt = torch.autograd.grad(p.sum(), (xr, tr), create_graph=(order == 2))
            if order == 1:
                return gx.double(), gt.double()
            lap = sum(torch.autograd.grad(gx[:, i].sum(), xr, retain_graph=True)[0][:, i] for i in range(x.shape[1]))
            return lap.double()

    def frot(x, t):
        with torch.enable_grad():
            xr = x.detach().float().requires_grad_(True)
            g = torch.autograd.grad(psi_net(xr, t.float().reshape(-1).expand(len(x))).sum(), xr)[0]
        return torch.stack([-g[:, 1], g[:, 0]], 1).double()
    return Gauge(lambda x, t: -_phi(x, t).double().detach(), lambda x, t: -_grads(x, t, 1)[0],
                 lambda x, t: -_grads(x, t, 2), lambda x, t: -_grads(x, t, 1)[1], frot, 2, 'networks')


def walk_exact(x0, tau0, G, terminal_u, P=4096, c=0.6, eps=1e-3, pV=0.15, pG=0.15, max_steps=2000, seed=None, stats=False):
    """Estimate u(x0, tau0) with P independent walks.  x0: (d,) tensor; terminal_u(X (n,d), T (n,)) -> u values.
    Returns the per-path estimates (P,) of u(x0,tau0) (their mean is the estimator) and optional statistics."""
    if seed is not None:
        state = torch.random.get_rng_state(); torch.manual_seed(seed)
    d = x0.shape[0]
    pM = 1.0 - pV - pG
    X = x0.to(DT).expand(P, d).clone(); T = torch.full((P,), float(tau0), dtype=DT); W = torch.ones(P, dtype=DT)
    alive = T > eps; steps = torch.zeros(P); c1 = C1(d); zI = Z_family(d, 0.5, 0.0, 1.0)
    nbr = dict(M=0, V=0, G=0)
    for k in range(max_steps):
        if not alive.any():
            break
        r = c * torch.sqrt(4 * math.pi * T)
        u = torch.rand(P, dtype=DT)
        zw, sw = sample_watson(d, P, dtype=DT)
        zp, sp = sample_source(d, P, dtype=DT); U = torch.rand(P, dtype=DT)
        zi, si, zh = sample_ibp(d, P, dtype=DT); U2 = torch.rand(P, dtype=DT)
        sg = (torch.rand(P, dtype=DT) < 0.5).to(DT) * 2 - 1
        bM = alive & (u < pM); bV = alive & (u >= pM) & (u < pM + pV); bG = alive & (u >= pM + pV)
        Xn, Tn, Wn = X.clone(), T.clone(), W.clone()
        if bM.any():
            Xn[bM] = X[bM] + r[bM, None] * zw[bM]; Tn[bM] = T[bM] - r[bM] ** 2 * sw[bM]; Wn[bM] = W[bM] / pM
        if bV.any():
            rv = r[bV]; Y = X[bV] + (rv * U[bV].sqrt())[:, None] * zp[bV]; TY = T[bV] - rv ** 2 * U[bV] * sp[bV]
            Xn[bV] = Y; Tn[bV] = TY; Wn[bV] = W[bV] * (-(c1 * rv ** 2 / 4) * G.V(Y, TY) / pV)
        if bG.any():
            rg = r[bG]; rho = rg * U2[bG]; Y = X[bG] + (sg[bG] * rho)[:, None] * zi[bG]; TY = T[bG] - rho ** 2 * si[bG]
            Xn[bG] = Y; Tn[bG] = TY
            Wn[bG] = W[bG] * (-(d * rg * zI / 2) * sg[bG] * (G.frot(Y, TY) * zh[bG]).sum(1) / pG)
        X, T, W = Xn, Tn, Wn
        steps += alive.float()
        nbr['M'] += int(bM.sum()); nbr['V'] += int(bV.sum()); nbr['G'] += int(bG.sum())
        alive = T > eps
    phi_term = torch.exp(G.Phi(X, T) / 2) * terminal_u(X, T)
    est = torch.exp(-G.Phi(x0.to(DT)[None], torch.tensor([tau0], dtype=DT)) / 2) * W * phi_term
    if seed is not None:
        torch.random.set_rng_state(state)
    if stats:
        return est, dict(mean_steps=float(steps.mean()), max_steps=float(steps.max()), w_abs_mean=float(W.abs().mean()),
                         w_sd=float(W.std()), frac_negative=float((W < 0).double().mean()), branches=nbr)
    return est


def walk_frozen(x0, tau0, G, terminal_u, P=4096, c=0.6, eps=1e-3, scheme='direct', max_steps=5000, seed=None,
                stats=False, antithetic=True):
    """Moving heat-ball walk with frozen coefficients: positive weights, no branching.
    Over one heat ball the transport velocity v and the potential V are frozen (Heun / trapezoid), which makes
    the step exact for constant coefficients: in the frame moving with v the equation is the heat equation (Watson's
    identity applies), and a constant zero-order term contributes exp(-V dtau).  Step:
        (z, sigma) ~ kappa,  dtau = r^2 sigma,  X' = X + r z - v_avg dtau,  W' = W exp(-(V(X)+V(X'))/2 dtau)
    scheme 'direct':  v = f,      V = div f          (u itself;  Feynman--Kac form of the density)
    scheme 'gauge' :  v = f_rot,  V = gauge potential (phi = exp(Phi/2) u; the curl-free part enters only via V)
    antithetic=True: the second half of the paths uses -z at every step (same sigma).
    x0 may require grad: the estimate is then differentiable in x0 (pathwise gradient, common random numbers).
    Local error O(r^3)-O(r^4) per step for smooth coefficients, i.e. a global bias O(r^2) = O(c^2 tau); no
    sign problem."""
    if seed is not None:
        state = torch.random.get_rng_state(); torch.manual_seed(seed)
    d = x0.shape[0]
    if scheme == 'gauge':
        vel, pot = G.frot, G.V
    else:
        vel = G.drift
        pot = lambda x, t: -G.lapPhi(x, t)            # div f = -Lap Phi + div f_rot (= 0)
    X = x0.to(DT).expand(P, d) + torch.zeros(P, d, dtype=DT); T = torch.full((P,), float(tau0), dtype=DT)
    logW = torch.zeros(P, dtype=DT)
    alive = T > eps; steps = torch.zeros(P)
    for k in range(max_steps):
        if not alive.any():
            break
        r = c * torch.sqrt(4 * math.pi * T)
        if antithetic:
            zh, sh = sample_watson(d, P // 2, dtype=DT); z = torch.cat([zh, -zh]); s = torch.cat([sh, sh])
        else:
            z, s = sample_watson(d, P, dtype=DT)
        dt = torch.where(alive, r ** 2 * s, torch.zeros_like(T))
        Tn = T - dt
        v0 = vel(X, T); Xp = X + r[:, None] * z - v0 * dt[:, None]
        Xn = X + r[:, None] * z - 0.5 * (v0 + vel(Xp, Tn)) * dt[:, None]
        Xn = torch.where(alive[:, None], Xn, X)
        logW = logW - 0.5 * (pot(X, T) + pot(Xn, Tn)) * dt
        bad = ~torch.isfinite(Xn).all(1) | ~torch.isfinite(logW)
        if bad.any():                                   # explosive paths (e.g. backward characteristics of a
            nbad = int(bad.sum())                       # super-linear confining drift): reported, not hidden
            raise FloatingPointError(f'{nbad} of {P} paths became non-finite at step {k}')
        X, T = Xn, Tn; steps += alive.float(); alive = T > eps
    if scheme == 'gauge':
        lt = logW + G.Phi(X, T) / 2 - G.Phi(x0.to(DT)[None], torch.tensor([tau0], dtype=DT)) / 2
    else:
        lt = logW
    est = torch.exp(lt) * terminal_u(X, T)
    if seed is not None:
        torch.random.set_rng_state(state)
    if stats:
        return est, dict(mean_steps=float(steps.mean()), max_steps=float(steps.max()), logw_sd=float(lt.detach().std()))
    return est


def walk_score(x0, tau0, G, terminal_u, P=100000, batches=10, seed=0, **kw):
    """u, grad u and score at x0 from the frozen walk with pathwise (autograd) gradients; standard errors from
    `batches` independent batches.  Score = mean(grad u) / mean(u) (ratio estimator, bias O(1/P))."""
    us, gs, st = [], [], None
    for b in range(batches):
        xr = x0.to(DT).clone().requires_grad_(True)
        est, st = walk_frozen(xr, tau0, G, terminal_u, P=P // batches, seed=seed * 1000 + b, stats=True, **kw)
        m = est.mean(); g, = torch.autograd.grad(m, xr)
        us.append(float(m)); gs.append(g.detach())
    us = torch.tensor(us, dtype=DT); gs = torch.stack(gs)
    u = us.mean(); gu = gs.mean(0)
    # delta-method standard error of the score (ratio of means over batches)
    sc_b = gs / u - us[:, None] * gu / u ** 2
    return dict(u=float(u), u_se=float(us.std() / math.sqrt(batches)), grad=gu, score=gu / u,
                score_se=sc_b.std(0) / math.sqrt(batches), stats=st)


def walk(x0, tau0, G, terminal_u, method='frozen', **kw):
    return walk_frozen(x0, tau0, G, terminal_u, **kw) if method == 'frozen' else walk_exact(x0, tau0, G, terminal_u, **kw)


def walk_u_grad(x0, tau0, G, terminal_u, P=4096, h=1e-3, seed=0, **kw):
    """u and grad u at x0 by common-random-number central differences (same seed for all 2d+1 walks)."""
    est, st = walk(x0, tau0, G, terminal_u, P=P, seed=seed, stats=True, **kw)
    d = x0.shape[0]; g = torch.zeros(d, dtype=DT); g_paths = []
    for i in range(d):
        e = torch.zeros(d, dtype=DT); e[i] = h
        ep = walk(x0 + e, tau0, G, terminal_u, P=P, seed=seed, **kw)
        em = walk(x0 - e, tau0, G, terminal_u, P=P, seed=seed, **kw)
        g_paths.append((ep - em) / (2 * h))
    gp = torch.stack(g_paths, 1)                               # (P,d) per-path gradient estimates
    u = est.mean(); gu = gp.mean(0)
    return dict(u=float(u), u_se=float(est.std() / math.sqrt(P)), grad=gu, grad_se=gp.std(0) / math.sqrt(P),
                score=gu / u, stats=st, paths_u=est, paths_g=gp)


# ---------------- terminal conditions ----------------
def terminal_gmm_heat(G0):
    """u(x, tau) ~ (p0 * N(0, 2 tau I))(x): heat flow of the initial mixture over the last (tiny) interval."""
    from .linear_gmm import LinearDrift
    D0 = LinearDrift.ou(G0.d, a=0.0)
    return lambda X, T: torch.exp(G0.logp_score(X, T, D0, need_score=False))


def terminal_knn(data, k=64):
    """Same, with p0 replaced by the empirical measure of `data` (n,d): Gaussian kernel of variance 2 tau, summed
    over the k nearest data points only (cKDTree) -- cost independent of n up to the tree query."""
    from scipy.spatial import cKDTree
    data = np.asarray(data, float); tree = cKDTree(data); n, d = data.shape

    def f(X, T):
        Xn = X.detach().numpy(); dist, idx = tree.query(Xn, k=min(k, n))
        var = 2 * T.detach().numpy()[:, None]
        val = np.exp(-dist ** 2 / (2 * var)).sum(1) / (n * (2 * math.pi * var[:, 0]) ** (d / 2))
        return torch.as_tensor(val, dtype=DT)
    return f


def terminal_gmm_short(G0, drift):
    """Short-time Gaussian approximation of u(x, T) from the initial mixture: each component N(m, S) is moved by
    the linearised drift over [0, T]:  m' = m + f(m) T,  S' = S + 2 T I + T (B S + S B^T),  B = grad f(m).
    Local error O(T^2); used to stop the walk at a moderate T_stop (variance/bias trade-off)."""
    m = G0.mu.to(DT); S = G0.S0[:, 0].to(DT); w = G0.w.to(DT)          # (K,2), (K,2,2), (K,)
    z0 = torch.zeros(len(m), dtype=DT); hh = 1e-4
    fm = drift(m, z0).detach()
    cols = []                                                         # Jacobian by central differences
    for j in range(2):                                                # (works for black-box / network drifts)
        e = torch.zeros(2, dtype=DT); e[j] = hh
        cols.append(((drift(m + e, z0) - drift(m - e, z0)) / (2 * hh)).detach())
    B = torch.stack(cols, 2)                                          # B[k, i, j] = d f_i / d x_j
    I2 = torch.eye(2, dtype=DT)

    def f(X, T):
        T3 = T[:, None, None, None]                                    # (P,1,1,1)
        mm = m[None] + fm[None] * T[:, None, None]                     # (P,K,2)
        SS = S[None] + 2 * T3 * I2 + T3 * (B @ S + S @ B.transpose(1, 2))[None]     # (P,K,2,2)
        diff = X[:, None, :] - mm
        a, b, c_ = SS[..., 0, 0], SS[..., 0, 1], SS[..., 1, 1]
        det = a * c_ - b * b
        q = (c_ * diff[..., 0] ** 2 - 2 * b * diff[..., 0] * diff[..., 1] + a * diff[..., 1] ** 2) / det
        return (w[None] * torch.exp(-0.5 * q) / (2 * math.pi * det.sqrt())).sum(1)
    return f


def terminal_knn_short(data, drift, k=1024, m_sub=None, seed=0):
    """Empirical-data version: kernel N(x; y + f(y) T, 2 T I) summed over the k nearest data points of x.
    m_sub: if given, every call (one batch of paths) uses a fresh random subset of m_sub data points, which is an
    UNBIASED estimate of the full empirical kernel sum and keeps the cost independent of the data size n.
    f.trunc records, per call, the fraction of paths whose k-th neighbour lies within 3 kernel standard deviations
    (where truncation of the kernel sum may matter)."""
    from scipy.spatial import cKDTree
    data_t = torch.as_tensor(np.asarray(data, float), dtype=DT); n, d = data_t.shape
    fy_all = drift(data_t, torch.zeros(n, dtype=DT)).detach()
    rng = np.random.default_rng(seed)
    full_tree = cKDTree(data_t.numpy()) if m_sub is None or m_sub >= n else None

    def f(X, T):
        if full_tree is not None:
            tree, dat, fy, nn = full_tree, data_t, fy_all, n
        else:
            sel = torch.as_tensor(rng.choice(n, m_sub, replace=False))
            dat, fy, nn = data_t[sel], fy_all[sel], m_sub; tree = cKDTree(dat.numpy())
        kk = min(k, nn)
        dist, idx = tree.query(X.detach().numpy(), k=kk)
        if kk < nn:
            f.trunc.append(float((dist[:, -1] < 3 * np.sqrt(2 * T.detach().numpy())).mean()))
        idx = torch.as_tensor(idx)
        mu = dat[idx] + fy[idx] * T[:, None, None]                    # (P,k,d)
        var = 2 * T[:, None]
        q = ((X[:, None, :] - mu) ** 2).sum(-1) / var
        return torch.exp(-0.5 * q).sum(1) / (nn * (2 * math.pi * var[:, 0]) ** (d / 2))
    f.trunc = []
    return f


def _local_lip(fn, X, T, h=1e-3, vector=True):
    """cheap local Lipschitz estimate of fn at X by central differences along the coordinate axes."""
    out = torch.zeros(len(X), dtype=DT)
    for j in range(X.shape[1]):
        e = torch.zeros_like(X); e[:, j] = h
        dfx = (fn(X + e, T) - fn(X - e, T)) / (2 * h)
        out = out + ((dfx ** 2).sum(1) if vector else dfx ** 2)
    return out.sqrt().detach()


def walk_corrected(x0, tau0, G, terminal_u, P=4096, c=0.5, eps=1e-3, scheme='gauge', aG=0.02, aV=0.005, pmax=0.3,
                   max_steps=5000, seed=None, stats=False, antithetic=True):
    """UNBIASED moving heat-ball walk: frozen coefficients + exact integration-by-parts corrections.
    At the walker's current point x_k the velocity v = vel(x_k) and the potential V0 = V(x_k) are frozen; with
    G = vel - v and dV = V - V0 the equation is, exactly,
        d_tau w = Lap w - v.grad w - V0 w  - div(G w) - dV w .
    In the frame moving with v and with the factor exp(-V0 dtau) the first three terms are the heat equation, so
    Watson's identity holds with the source  -div(G w) - dV w.  The divergence term is integrated by parts against
    psi_0 (no boundary term: psi_0 = 0 on the heat-ball boundary) -> mu_{1/2,0,1}, values of G w only; the potential
    term uses the source measure mu_psi.  On the heat ball G = O(L_f r) and dV = O(L_V r), so the correction
    contributions are O(L_f r^2) and O(L_V r^3).  The branch probabilities are chosen proportional to these sizes,
        pG = min(pmax, aG L_f r^2),  pV = min(pmax, aV L_V r^3)   (L from local finite differences),
    so that a correction branch, when taken, carries a weight of order one; the second moment per step is then
    1 + O(r^2) and, over the O(1/r^2) steps of a walk, stays bounded as r -> 0 (exp of a constant).
    scheme 'direct': w = u, vel = f, no potential (div f is inside div(G u)).
    scheme 'gauge' : w = phi = exp(Phi/2) u, vel = f_rot, potential V of Proposition (gauge)."""
    if seed is not None:
        state = torch.random.get_rng_state(); torch.manual_seed(seed)
    d = x0.shape[0]; c1 = C1(d); zI = Z_family(d, 0.5, 0.0, 1.0)
    gauge = scheme == 'gauge'
    vel = G.frot if gauge else G.drift
    X = x0.to(DT).expand(P, d) + torch.zeros(P, d, dtype=DT); T = torch.full((P,), float(tau0), dtype=DT)
    W = torch.ones(P, dtype=DT); alive = T > eps; steps = torch.zeros(P); nbr = dict(M=0, G=0, V=0)
    for k in range(max_steps):
        if not alive.any():
            break
        r = c * torch.sqrt(4 * math.pi * T)
        Lf = _local_lip(vel, X.detach(), T)
        pG = torch.clamp(aG * Lf * r ** 2, max=pmax)
        if gauge:
            LV = _local_lip(lambda x, t: G.V(x, t)[:, None], X.detach(), T)
            pV = torch.clamp(aV * LV * r ** 3, max=pmax)
        else:
            pV = torch.zeros_like(pG)
        pM = 1 - pG - pV
        u = torch.rand(P, dtype=DT)
        if antithetic:
            zh_, sh_ = sample_watson(d, P // 2, dtype=DT); zw = torch.cat([zh_, -zh_]); sw = torch.cat([sh_, sh_])
        else:
            zw, sw = sample_watson(d, P, dtype=DT)
        zi, si, zhat = sample_ibp(d, P, dtype=DT); U2 = torch.rand(P, dtype=DT); sg = (torch.rand(P, dtype=DT) < 0.5).to(DT) * 2 - 1
        zp, sp = sample_source(d, P, dtype=DT); U3 = torch.rand(P, dtype=DT)
        bM = alive & (u < pM); bG = alive & (u >= pM) & (u < pM + pG); bV = alive & (u >= pM + pG)
        v0 = vel(X, T); V0 = G.V(X, T) if gauge else torch.zeros_like(T)
        rho = r * U2
        dtau = torch.where(bM, r ** 2 * sw, torch.where(bG, rho ** 2 * si, torch.where(bV, r ** 2 * U3 * sp, torch.zeros_like(T))))
        off = torch.where(bM[:, None], r[:, None] * zw,
                          torch.where(bG[:, None], (sg * rho)[:, None] * zi, (r * U3.sqrt())[:, None] * zp))
        Xn = torch.where(alive[:, None], X + off - v0 * dtau[:, None], X); Tn = T - dtau
        fac = torch.exp(-V0 * dtau)
        mult = torch.where(bM, fac / pM, torch.ones_like(T))
        if bG.any():
            Gv = vel(Xn, Tn) - v0
            mG = fac * (-(d * r * zI / 2) * sg * (Gv * zhat).sum(1)) / pG.clamp_min(1e-300)
            mult = torch.where(bG, mG, mult)
        if gauge and bV.any():
            mV = fac * (-(c1 * r ** 2 / 4) * (G.V(Xn, Tn) - V0)) / pV.clamp_min(1e-300)
            mult = torch.where(bV, mV, mult)
        W = W * torch.where(alive, mult, torch.ones_like(T))
        if not (torch.isfinite(Xn).all() and torch.isfinite(W).all()):
            raise FloatingPointError(f'non-finite paths at step {k}')
        nbr['M'] += int(bM.sum()); nbr['G'] += int(bG.sum()); nbr['V'] += int(bV.sum())
        X, T = Xn, Tn; steps += alive.float(); alive = T > eps
    if gauge:
        W = W * torch.exp(G.Phi(X, T) / 2 - G.Phi(x0.to(DT)[None], torch.tensor([tau0], dtype=DT)) / 2)
    est = W * terminal_u(X, T)
    if seed is not None:
        torch.random.set_rng_state(state)
    if stats:
        return est, dict(mean_steps=float(steps.mean()), w_sd=float(W.detach().std()),
                         frac_negative=float((W < 0).double().mean()), branches=nbr)
    return est


def walk_score_any(x0, tau0, G, terminal_u, walker, P=100000, batches=10, seed=0, **kw):
    """walk_score for any walker (walk_frozen or walk_corrected): pathwise gradients, batch standard errors."""
    us, gs, st = [], [], None
    for b in range(batches):
        xr = x0.to(DT).clone().requires_grad_(True)
        est, st = walker(xr, tau0, G, terminal_u, P=P // batches, seed=seed * 1000 + b, stats=True, **kw)
        m = est.mean(); g, = torch.autograd.grad(m, xr)
        us.append(float(m.detach())); gs.append(g.detach())
    us = torch.tensor(us, dtype=DT); gs = torch.stack(gs); u = us.mean(); gu = gs.mean(0)
    sc_b = gs / u - us[:, None] * gu / u ** 2
    return dict(u=float(u), u_se=float(us.std() / math.sqrt(batches)), grad=gu, score=gu / u,
                score_se=sc_b.std(0) / math.sqrt(batches), stats=st)
