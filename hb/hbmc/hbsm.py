"""
hbmc.hbsm -- heat-ball score matching: a data-driven, transition-kernel-free score-matching loss from the
radius-averaged heat-ball identity of the (gauge-transformed) density.

Setting (tau = cumulative variance):  d_tau u = Lap u - div(f u),  f = -grad Phi + f_rot,  div f_rot = 0.
Gauge (Proposition gauge):  omega = e^{Phi/2} u  solves  d_tau omega - Lap omega = -div(f_rot omega) - V omega.
Coordinates of a point (y, tau) relative to the apex (x0, tau0):  xi = x0 - y,  delta = tau0 - tau > 0.

Heat-ball kernels (Gamma = heat kernel, a = Gamma^{-1/d} = smallest radius whose ball contains the point):
  Watson   K_rho  = rho^{-d} |xi|^2 / (4 delta^2) 1{a < rho}
  source   Psi_rho = Gamma - rho^{-d} (1 + d log(rho / a)) 1{a < rho}      (S_rho[h] = int int h Psi_rho)
Averaging omega(x0) = M_rho[omega] + S_rho[h] over rho in (0, r) with a weight w(rho) gives kernels Kbar, Psibar that
vanish on the boundary of E_r, so the identity can be differentiated in x0 under the integral without a surface term
(closed forms in kernels(); w(rho) ~ rho^{d+1} keeps the variance of the gradient estimator finite).
For f_rot = 0 (gradient drift, e.g. OU):
  omega(x0)      = int int omega(y, tau) [Kbar - V(y) Psibar] dy dtau
  grad omega(x0) = int int omega(y, tau) [grad_xi Kbar - V(y) grad_xi Psibar] dy dtau
so grad u(x0) = int int u(y, tau) Kvec(x0, y, tau) dy dtau with
  Kvec = e^{(Phi(y) - Phi(x0))/2} [grad Kbar - V grad Psibar - grad Phi(x0)/2 (Kbar - V Psibar)].
Score matching:  int u |s_theta - s|^2 = E_{x~p_tau0} |s_theta|^2 - 2 int s_theta . grad u + const, and
  int s_theta . grad u dx0 = E_{(xi, delta) ~ q, y ~ p_{tau0 - delta}} [ s_theta(y + xi) . Kvec / q(xi, delta) ]
needs only samples of the forward process at earlier times and the values of s_theta (no transition kernel, no
derivative of the network).  q: a Beta-Gamma heat-ball measure (hbmc.heatball), scaled to radius r.
"""
import math
import torch
from .heatball import sample_family, Z_family

__all__ = ["kernels", "kernel_grads", "sample_q", "kvec", "hbsm_cross", "hbsm_coloc"]


def _log_gamma(xi, delta, d):
    return -0.5 * d * torch.log(4 * math.pi * delta) - (xi * xi).sum(1) / (4 * delta)


def kernels(xi, delta, r, d, k=None):
    """Kbar, Psibar at (xi, delta) for radius r (tensors (n,)); zero outside E_r.
    Radius average with weight w(rho) = (k+1) rho^k / r^{k+1} on (0, r) (default k = d + 1).  Every single-radius
    identity is exact, so any weight is; k > d - 1 removes the a^{1-d} apex singularity of the uniform average, which
    otherwise makes the gradient estimator's variance infinite in low dimension.  With m = k + 1 - d > 0:
      Kbar   = |xi|^2/(4 delta^2) (k+1)/r^{k+1} (r^m - a^m)/m
      Psibar = (k+1)/r^{k+1} [ Gamma (r^{k+1} - a^{k+1})/(k+1) - (r^m - a^m)/m - d J1 ],
      J1 = r^m log(r/a)/m - (r^m - a^m)/m^2."""
    k = d + 1.0 if k is None else float(k)
    m = k + 1 - d
    lg = _log_gamma(xi, delta, d)
    lgr = -d * torch.log(torch.as_tensor(r, dtype=lg.dtype)).expand_as(lg)        # log Gamma on the boundary a = r
    inside = lg > lgr
    lg = torch.where(inside, lg, lgr)       # clamp before exp: no inf (and no 0 * inf in autograd) far outside the ball
    a = torch.exp(-lg / d)
    c = (k + 1) / r ** (k + 1)
    rm, am = r ** m, a ** m
    Kb = (xi * xi).sum(1) / (4 * delta ** 2) * c * (rm - am) / m
    J1 = rm * torch.log(r / a) / m - (rm - am) / m ** 2
    Pb = c * (torch.exp(lg) * (r ** (k + 1) - a ** (k + 1)) / (k + 1) - (rm - am) / m - d * J1)
    z = torch.zeros_like(Kb)
    return torch.where(inside, Kb, z), torch.where(inside, Pb, z)


def kernel_grads(xi, delta, r, d, k=None):
    """(Kbar, Psibar, grad_xi Kbar, grad_xi Psibar) by autograd in xi."""
    with torch.enable_grad():
        x = xi.detach().requires_grad_(True)
        Kb, Pb = kernels(x, delta, r, d, k)
        gK, = torch.autograd.grad(Kb.sum(), x, retain_graph=True)
        gP, = torch.autograd.grad(Pb.sum(), x)
    return Kb.detach(), Pb.detach(), gK, gP


def sample_q(n, d, r, abc=(1.0, 0.0, 2.0), dtype=torch.float64):
    """(xi, delta, q) with (z, sigma) ~ mu_{a,b,c} on the unit heat ball, xi = r z, delta = r^2 sigma;
    q(xi, delta) = |z|^{2a} psi0^b sigma^{-c} / Z_{abc} * r^{-d-2} (density in (xi, delta))."""
    a_, b_, c_ = abc
    z, sg, _ = sample_family(d, a_, b_, c_, n, dtype=dtype)
    r = torch.as_tensor(r, dtype=dtype).expand(n)
    R2 = -2.0 * d * sg * torch.log(4 * math.pi * sg)
    psi0 = (R2 - (z * z).sum(1)) / (4 * sg)
    dens = (z * z).sum(1) ** a_ * psi0.clamp_min(0) ** b_ * sg ** (-c_) / Z_family(d, a_, b_, c_)
    return r[:, None] * z, r ** 2 * sg, dens * r ** (-d - 2)


def kvec(xi, delta, r, y, tau, x0, G, d, k=None, s_y=None, with_scalar=False):
    """Kvec(x0, y, tau), shape (n, d); G: hbmc.walk.Gauge (f = -grad Phi + f_rot).
    Rotation part: the source -f_rot . grad omega = -omega (f_rot . grad eta) is kept in this form, with
    grad eta(y) = s(y) + grad Phi(y)/2 from the model's score s_y at y (stop-gradient), i.e. an effective potential
    V_eff = V + f_rot . grad eta.  (Integrating it by parts instead puts the Hessian of Psibar into the kernel, which is
    even in xi and makes the variance infinite near the apex.)  The fixed point is still the true score.
      omega(x0)      = int int omega(y) [Kbar - V_eff Psibar],   grad omega(x0) = int int omega(y) [grad Kbar - V_eff grad Psibar]
      Kvec = e^{(Phi(y) - Phi(x0))/2} [grad omega kernel - grad Phi(x0)/2 * omega kernel]."""
    yd, td = y.double(), tau.double()
    Kb, Pb, gK, gP = kernel_grads(xi, delta, r, d, k)
    V = G.V(yd, td).to(y.dtype)
    fr = G.frot(yd, td).to(y.dtype)
    if bool(fr.abs().max() > 0):
        if s_y is None:
            raise ValueError('a drift with a rotation part needs the score at y (s_y)')
        V = V + (fr * (s_y.detach() + 0.5 * G.gradPhi(yd, td).to(y.dtype))).sum(1)
    t0 = (tau + delta).double()
    gPhi0 = G.gradPhi(x0.double(), t0).to(y.dtype)
    ew = torch.exp(0.5 * (G.Phi(yd, td) - G.Phi(x0.double(), t0))).to(y.dtype)
    Kv = ew[:, None] * (gK - V[:, None] * gP - 0.5 * gPhi0 * (Kb - V * Pb)[:, None])
    if with_scalar:   # Ks: u(x0) = int int u(y) Ks dy dtau (the same identity without the gradient)
        return Kv, ew * (Kb - V * Pb)
    return Kv


def hbsm_cross(s_fn, xi, delta, q, r, y, tau, tau0, G, d, k=3.0):
    """Antithetic estimate of  int s_theta . grad u dx0  per sample (shape (n,)):
       1/2 [ s(y+xi) . Kvec(xi) + s(y-xi) . Kvec(-xi) ] / q.   Kbar is even and grad Kbar odd in xi, so the pairing
       cancels the leading (1/|xi|) singularity: per-sample variance a few times that of DSM (2D, OU) instead of ~10^3."""
    xp, xm = y + xi, y - xi
    t0v = torch.full_like(tau, tau0) if not torch.is_tensor(tau0) else tau0
    rot = bool(G.frot(y.double(), tau.double()).abs().max() > 0)
    s_y = s_fn(y, tau).detach() if rot else None          # rotation part: model score at y, stop-gradient
    Kp = kvec(xi, delta, r, y, tau, xp, G, d, k, s_y); Km = kvec(-xi, delta, r, y, tau, xm, G, d, k, s_y)
    return 0.5 * ((s_fn(xp, t0v) * Kp).sum(1) + (s_fn(xm, t0v) * Km).sum(1)) / q


def hbsm_coloc(s_fn, xi, delta, q, r, y, tau, tau0, G, d, k=3.0):
    """Co-located HBSM: both terms of  int u |s_theta|^2 - 2 int s_theta . grad u  at the same points x0 = y +- xi,
       the quadratic one through u(x0) = int int u(y) Ks:   per pair  1/2 sum_+- [Ks |s(x0)|^2 - 2 s(x0) . Kvec] / q
       = 1/2 sum_+- (Ks/q) |s - Kvec/Ks|^2 + const:  a weighted regression, bounded below when Ks >= 0 (Ks < 0 carries
       < 0.2% of the weight in the 2D tests; clipped at 0).  The split form (|s|^2 on x ~ p_tau, cross term on x0) is
       unbounded below in finite samples where x0 visits regions that x ~ p_tau does not (seen as divergence)."""
    xp, xm = y + xi, y - xi
    t0v = torch.full_like(tau, tau0) if not torch.is_tensor(tau0) else tau0
    rot = bool(G.frot(y.double(), tau.double()).abs().max() > 0)
    s_y = s_fn(y, tau).detach() if rot else None
    Kp, Sp = kvec(xi, delta, r, y, tau, xp, G, d, k, s_y, with_scalar=True)
    Km, Sm = kvec(-xi, delta, r, y, tau, xm, G, d, k, s_y, with_scalar=True)
    sp, sm = s_fn(xp, t0v), s_fn(xm, t0v)
    Sp, Sm = Sp.clamp_min(0), Sm.clamp_min(0)
    return 0.5 * ((Sp * (sp * sp).sum(1) - 2 * (sp * Kp).sum(1)) + (Sm * (sm * sm).sum(1) - 2 * (sm * Km).sum(1))) / q
