"""
hbmc.residuals -- heat-ball and pointwise Fokker-Planck residuals for learned density models.

Conventions (tau = cumulative-variance time, d_tau u = Laplace u - div(f u)):
  * log-density model  v(x, tau)   (scalar, any normalisation constant independent of tau)
  * density model      u(x, tau) > 0
  * drift              f(x, tau) -> (n, d);   div_f(x, tau) -> (n,)  (or None: use integration by parts)

All heat-ball integrals use the exact samplers of hbmc.heatball (no sigma truncation, no
self-normalised weights, rho sampled exactly).  Squared residuals are made unbiased with the
two-replica product  R^(1) R^(2)  (Section "squared residuals without variance bias").
"""
import math
import torch
from .heatball import (C1, Z_family, sample_watson, sample_source, sample_ibp, admissible_r)

__all__ = ["score_of", "hb_residual_v", "hb_residual_u", "hb_loss", "pointwise_residual_v", "hb_residual_s", "hb_residual_s_mass"]


def score_of(v_fn, x, tau, create_graph=False):
    with torch.enable_grad():
        xr = x if x.requires_grad else x.detach().requires_grad_(True)
        v = v_fn(xr, tau)
        s, = torch.autograd.grad(v.sum(), xr, create_graph=create_graph)
    return v, s


def _rep(t, k):
    return t.repeat_interleave(k, 0)


def _div_ibp(f_fn, x0, tau0, r, n, d):
    """Unbiased estimate of S_r[-div f] at each anchor (antithetic IBP, mu_{1/2,0,1})."""
    B = x0.shape[0]
    z, s, zh = sample_ibp(d, B * n, device=x0.device, dtype=x0.dtype)
    U = torch.rand(B * n, device=x0.device, dtype=x0.dtype)
    rr = _rep(r, n); rho = rr * U
    T = _rep(tau0, n) - rho ** 2 * s
    xp = _rep(x0, n) + rho[:, None] * z; xm = _rep(x0, n) - rho[:, None] * z
    g = 0.5 * ((f_fn(xp, T) - f_fn(xm, T)) * zh).sum(1)
    Z = Z_family(d, 0.5, 0.0, 1.0)
    return (-d * rr * Z / 2.0 * g).view(B, n).mean(1)


def _watson_mean(fn, x0, tau0, r, N, antithetic=True):
    """M_r[fn] with exact Watson samples.  antithetic=True pairs z with -z (kappa is symmetric):
    the O(r) term cancels, so the Monte Carlo noise is O(r^2) like the residual itself."""
    B, d = x0.shape
    n = N // 2 if antithetic else N
    z, s = sample_watson(d, B * n, device=x0.device, dtype=x0.dtype)
    rr = _rep(r, n); X0 = _rep(x0, n); T = _rep(tau0, n) - rr ** 2 * s
    if antithetic:
        vals = 0.5 * (fn(X0 + rr[:, None] * z, T) + fn(X0 - rr[:, None] * z, T))
    else:
        vals = fn(X0 + rr[:, None] * z, T)
    return vals.view(B, n).mean(1)


def hb_residual_v(v_fn, f_fn, div_f, x0, tau0, r, N=16, Np=16, mode="ad", eps=1e-2,
                  n_dirs=1, div_mode="analytic", create_graph=True, v0=None, n_div=None, antithetic=True):
    """Heat-ball residual R_r[v] = v(x0,t0) - M_r[v] - S_r[h_v],  h_v = |grad v|^2 - f.grad v - div f.

    mode = "ad": grad v by automatic differentiation (create_graph for training)
    mode = "fd": |grad v|^2 by d * (central difference along a random unit direction)^2 (n_dirs
                 directions), f.grad v by a central difference along f/|f|  -> values of v only.
    div_mode = "analytic": div_f(x,tau) is used; "ibp": antithetic integration by parts (no derivative
                 of the drift).
    r: scalar or (B,) radius; clipped to the admissible range r^2 <= 4 pi tau0.
    Returns R (B,).
    """
    B, d = x0.shape
    dev, dt = x0.device, x0.dtype
    r = admissible_r(tau0, r)
    if v0 is None:
        v0 = v_fn(x0, tau0)
    M = _watson_mean(v_fn, x0, tau0, r, N, antithetic)
    # source term
    zp, sp = sample_source(d, B * Np, device=dev, dtype=dt)
    U = torch.rand(B * Np, device=dev, dtype=dt)
    rr = _rep(r, Np)
    Y = _rep(x0, Np) + (rr * U.sqrt())[:, None] * zp
    TY = _rep(tau0, Np) - rr ** 2 * U * sp
    f = f_fn(Y, TY)
    if mode == "ad":
        _, sc = score_of(v_fn, Y, TY, create_graph=create_graph)
        h = (sc * sc).sum(1) - (f * sc).sum(1)
    elif mode == "fd":
        sq = 0.0
        for _ in range(n_dirs):
            xi = torch.randn(B * Np, d, device=dev, dtype=dt); xi = xi / xi.norm(dim=1, keepdim=True)
            dv = (v_fn(Y + eps * xi, TY) - v_fn(Y - eps * xi, TY)) / (2 * eps)
            sq = sq + d * dv ** 2
        sq = sq / n_dirs
        fn = f.norm(dim=1, keepdim=True).clamp_min(1e-12); fh = (f / fn).detach()
        tr = fn[:, 0] * (v_fn(Y + eps * fh, TY) - v_fn(Y - eps * fh, TY)) / (2 * eps)
        h = sq - tr
    else:
        raise ValueError(mode)
    c = C1(d) * r ** 2 / 4.0
    S = c * h.view(B, Np).mean(1)
    if div_mode == "analytic":
        S = S + c * (-div_f(Y, TY)).view(B, Np).mean(1)
    elif div_mode == "ibp":
        S = S + _div_ibp(f_fn, x0, tau0, r, n_div or Np, d)
    else:
        raise ValueError(div_mode)
    return v0 - M - S


def hb_residual_u(u_fn, f_fn, x0, tau0, r, N=16, Np=16, u0=None, normalise=True, antithetic=True):
    """Density identity, derivative-free: R_r[u] = u(x0) - M_r[u] - S_r[-div(f u)],
    S_r[-div(f u)] by antithetic integration by parts.  Uses values of u and f only.
    normalise=True returns R / u(x0) (relative residual, gradient flows through u0 too)."""
    B, d = x0.shape
    dev, dt = x0.device, x0.dtype
    r = admissible_r(tau0, r)
    if u0 is None:
        u0 = u_fn(x0, tau0)
    M = _watson_mean(u_fn, x0, tau0, r, N, antithetic)
    zq, sq, zh = sample_ibp(d, B * Np, device=dev, dtype=dt)
    U = torch.rand(B * Np, device=dev, dtype=dt)
    rr = _rep(r, Np); rho = rr * U
    T = _rep(tau0, Np) - rho ** 2 * sq
    xp = _rep(x0, Np) + rho[:, None] * zq; xm = _rep(x0, Np) - rho[:, None] * zq
    G = lambda X: f_fn(X, T) * u_fn(X, T)[:, None]
    g = 0.5 * ((G(xp) - G(xm)) * zh).sum(1)
    S = (-d * rr * Z_family(d, 0.5, 0.0, 1.0) / 2.0 * g).view(B, Np).mean(1)
    R = u0 - M - S
    return R / u0.detach().clamp_min(1e-30) if normalise else R


def hb_loss(residual_fn, ustat=True, **kw):
    """Squared heat-ball residual. ustat=True: unbiased two-replica product mean(R1*R2)."""
    if ustat:
        R1 = residual_fn(**kw); R2 = residual_fn(**kw)
        return (R1 * R2).mean(), R1.detach()
    R = residual_fn(**kw)
    return (R * R).mean(), R.detach()


def pointwise_residual_v(v_fn, f_fn, div_f, x, tau, lap="exact", create_graph=True):
    """Pointwise FP residual of a log-density model (FP-Diffusion-type regulariser):
       d_tau v - Lap v - (|grad v|^2 - f.grad v - div f).
    lap = "exact" (d Hessian-vector products) or "hutch" (one Rademacher probe)."""
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True); tr = tau.detach().requires_grad_(True)
        v = v_fn(xr, tr)
        gx, gt = torch.autograd.grad(v.sum(), (xr, tr), create_graph=True)
        if lap == "exact":
            L = 0.0
            for i in range(x.shape[1]):
                L = L + torch.autograd.grad(gx[:, i].sum(), xr, create_graph=create_graph,
                                            retain_graph=True)[0][:, i]
        else:
            e = torch.randint(0, 2, x.shape, device=x.device).to(x.dtype) * 2 - 1
            L = (torch.autograd.grad((gx * e).sum(), xr, create_graph=create_graph,
                                     retain_graph=True)[0] * e).sum(1)
        f = f_fn(xr, tr)
        h = (gx * gx).sum(1) - (f * gx).sum(1) - div_f(xr, tr)
        return gt - L - h


def hb_residual_v_cv(v_fn, vref_fn, sref_fn, f_fn, div_f, x0, tau0, r, N=8, Np=8, n_ref=256, create_graph=True):
    """Heat-ball residual of the FULL v with a control variate for an analytic part v_ref of v (e.g. the Gaussian
    term -|x|^2/(2 g(tau)) of networks_mnist_v4.LogPNetV2):
        v = v_ref + v_rest,  s = s_ref + s_rest,
        M_r[v]   = M_r[v_ref] (n_ref samples) + M_r[v_rest] (N samples),
        S_r[h_v] = S_r[|s_ref|^2 - f.s_ref - div f] (n_ref) + S_r[2 s_ref.s_rest + |s_rest|^2 - f.s_rest] (Np).
    Every part is an expectation under the exact measures, so the estimator is unbiased for any split; the cheap
    analytic parts are oversampled.  Returns R (B,) in the units of v (multiply by 4/(C1 r^2) for pointwise units)."""
    B, d = x0.shape
    dev, dt = x0.device, x0.dtype
    rr = admissible_r(tau0, r)
    v0 = v_fn(x0, tau0)
    vrest = lambda X, T: v_fn(X, T) - vref_fn(X, T)
    M = _watson_mean(vrest, x0, tau0, rr, N) + _watson_mean(vref_fn, x0, tau0, rr, n_ref)

    def pts(n):
        zp, sp = sample_source(d, B * n, device=dev, dtype=dt)
        U = torch.rand(B * n, device=dev, dtype=dt); R_ = _rep(rr, n)
        return _rep(x0, n) + (R_ * U.sqrt())[:, None] * zp, _rep(tau0, n) - R_ ** 2 * U * sp
    Y, TY = pts(n_ref)
    sr = sref_fn(Y, TY)
    Sref = ((sr * sr).sum(1) - (f_fn(Y, TY) * sr).sum(1) - div_f(Y, TY)).view(B, n_ref).mean(1)
    Y, TY = pts(Np)
    with torch.enable_grad():
        Yr = Y.detach().requires_grad_(True)
        srest, = torch.autograd.grad(vrest(Yr, TY).sum(), Yr, create_graph=create_graph)
    sr = sref_fn(Y, TY)
    Snet = (2 * (sr * srest).sum(1) + (srest * srest).sum(1) - (f_fn(Y, TY) * srest).sum(1)).view(B, Np).mean(1)
    return v0 - M - C1(d) * rr ** 2 / 4.0 * (Sref + Snet)


def hb_residual_s(s_fn, f_fn, div_f, x0, tau0, r, N=16, Np=16, form="ibp", create_graph=True, s0=None,
                  antithetic=True, gradf_T_s=None, grad_div_f=None):
    """Vector (score) heat-ball residual  R_r[s] = s(x0) - M_r[s] - S_r[grad h_v],  h_v = |s|^2 - f.s - div f.

    form = "vjp": the source as in the score identity, grad h_v = (grad s)^T (2 s - f) - (grad f)^T s - grad div f,
                  one vector-Jacobian product of s per source point (and derivatives of the drift);
    form = "ibp": integration by parts in z against psi_0 (no boundary term, psi_0 = 0 on the heat-ball boundary):
                  S_r[grad h] = d int_0^r int_E1 (z / (2 sigma)) h  =  (d r Z_{1/2,0,1} / 2) E[ (h(x0+rho z) - h(x0-rho z))/2 zhat ],
                  (z, sigma) ~ mu_{1/2,0,1}, rho = r U:  values of s, f and div f only -- no derivative of the model.
    gradf_T_s(x, tau, s) and grad_div_f(x, tau) are needed for form="vjp" only (default: autograd through f_fn, div_f).
    Returns R (B, d) in the units of s (multiply by 4 / (C1 r^2) for pointwise units)."""
    B, d = x0.shape
    dev, dt = x0.device, x0.dtype
    r = admissible_r(tau0, r)
    if s0 is None:
        s0 = s_fn(x0, tau0)
    # Watson mean of the vector field (antithetic)
    n = N // 2 if antithetic else N
    z, sg = sample_watson(d, B * n, device=dev, dtype=dt)
    rr = _rep(r, n); X0 = _rep(x0, n); T = _rep(tau0, n) - rr ** 2 * sg
    vals = 0.5 * (s_fn(X0 + rr[:, None] * z, T) + s_fn(X0 - rr[:, None] * z, T)) if antithetic else s_fn(X0 + rr[:, None] * z, T)
    M = vals.view(B, n, d).mean(1)
    if form == "ibp":
        zq, sq, zh = sample_ibp(d, B * Np, device=dev, dtype=dt)
        U = torch.rand(B * Np, device=dev, dtype=dt)
        rr = _rep(r, Np); rho = rr * U
        T = _rep(tau0, Np) - rho ** 2 * sq
        xp = _rep(x0, Np) + rho[:, None] * zq; xm = _rep(x0, Np) - rho[:, None] * zq

        def h(X):
            s = s_fn(X, T); return (s * s).sum(1) - (f_fn(X, T) * s).sum(1) - div_f(X, T)
        g = 0.5 * (h(xp) - h(xm))[:, None] * zh
        S = (d * rr * Z_family(d, 0.5, 0.0, 1.0) / 2.0)[:, None] * g
        S = S.view(B, Np, d).mean(1)
    elif form == "vjp":
        zp, sp = sample_source(d, B * Np, device=dev, dtype=dt)
        U = torch.rand(B * Np, device=dev, dtype=dt)
        rr = _rep(r, Np)
        Y = _rep(x0, Np) + (rr * U.sqrt())[:, None] * zp
        TY = _rep(tau0, Np) - rr ** 2 * U * sp
        with torch.enable_grad():
            Yr = Y.detach().requires_grad_(True)
            s = s_fn(Yr, TY); f = f_fn(Yr, TY)
            w = (2 * s - f).detach()
            Jw, = torch.autograd.grad((s * w).sum(), Yr, create_graph=create_graph)    # (grad s)^T w (symmetric for gradients)
            if gradf_T_s is None:
                fs, = torch.autograd.grad((f * s.detach()).sum(), Yr, retain_graph=True)
            if grad_div_f is None:
                Yd = Y.detach().requires_grad_(True)
                gdf, = torch.autograd.grad(div_f(Yd, TY).sum(), Yd)
        if gradf_T_s is not None:
            fs = gradf_T_s(Y, TY, s.detach())
        if grad_div_f is not None:
            gdf = grad_div_f(Y, TY)
        gh = Jw - fs.detach() - gdf.detach()
        S = (C1(d) * rr ** 2 / 4.0)[:, None] * gh
        S = S.view(B, Np, d).mean(1)
    else:
        raise ValueError(form)
    return s0 - M - S


def hb_residual_s_mass(v_fn, f_fn, div_f, x0, tau0, r, N=16, Np=16, create_graph=True, antithetic=True, detach_weights=False):
    """Score residual in MASS form: the heat-ball identity of grad u (linear in u, no |s|^2 term), divided by u(x0):
         R = s0 - M_r[s e^{v - v0}] - S_r[grad H] / u0,     H = -div(f u),   H / u0 = -(div f + f.s) e^{v - v0},
       S_r[grad H] by integration by parts (antithetic, mu_{1/2,0,1}): values of v, s, f, div f only.
       Ratios e^{v - v0} make every term a ratio of masses (normalization constant cancels).  For any model, R / (C1 r^2/4)
       tends to grad(u P)/u = grad P + P s, P the pointwise Fokker-Planck residual of v; R = 0 for the true solution.
       detach_weights=True: the mass ratios e^{v - v0} are treated as fixed weights in the gradient (semi-gradient).
       Returns (B, d), units of s (multiply by 4/(C1 r^2) for pointwise units)."""
    B, d = x0.shape
    dev, dt = x0.device, x0.dtype
    r = admissible_r(tau0, r)
    v0, s0 = score_of(v_fn, x0, tau0, create_graph=create_graph)
    v0d = v0 if (create_graph and not detach_weights) else v0.detach()   # ratios e^{v - v0} depend on the model through v0 as well
    ex = (lambda a: torch.exp(a).detach()) if detach_weights else torch.exp
    n = N // 2 if antithetic else N
    z, sg = sample_watson(d, B * n, device=dev, dtype=dt)
    rr = _rep(r, n); X0 = _rep(x0, n); T = _rep(tau0, n) - rr ** 2 * sg; V0 = _rep(v0d, n)
    vp, sp_ = score_of(v_fn, X0 + rr[:, None] * z, T, create_graph=create_graph)
    if antithetic:
        vm, sm_ = score_of(v_fn, X0 - rr[:, None] * z, T, create_graph=create_graph)
        vals = 0.5 * (sp_ * ex(vp - V0)[:, None] + sm_ * ex(vm - V0)[:, None])
    else:
        vals = sp_ * ex(vp - V0)[:, None]
    M = vals.view(B, n, d).mean(1)
    zq, sq, zh = sample_ibp(d, B * Np, device=dev, dtype=dt)
    U = torch.rand(B * Np, device=dev, dtype=dt)
    rr = _rep(r, Np); rho = rr * U
    T = _rep(tau0, Np) - rho ** 2 * sq; V0 = _rep(v0d, Np); X0 = _rep(x0, Np)

    def Hu(X):
        v_, s_ = score_of(v_fn, X, T, create_graph=create_graph)
        return -(div_f(X, T) + (f_fn(X, T) * s_).sum(1)) * ex(v_ - V0)
    g = 0.5 * (Hu(X0 + rho[:, None] * zq) - Hu(X0 - rho[:, None] * zq))[:, None] * zh
    S = ((d * rr * Z_family(d, 0.5, 0.0, 1.0) / 2.0)[:, None] * g).view(B, Np, d).mean(1)
    return s0 - M - S
