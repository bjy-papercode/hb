"""
hbmc.losses -- training objectives, all for the same scalar model v_theta (s_theta = grad v_theta).

  dsm     denoising score matching with the EXACT Gaussian transition kernel (linear drifts only)
  ssm     sliced score matching (variance-reduced), valid for any forward drift; needs samples only
  fp      pointwise Fokker-Planck residual of v_theta (FP-Diffusion-type regulariser)
  hb      heat-ball residual of v_theta (exact samplers, unbiased two-replica square)
  bnd     tau = 0 boundary for residual-only training: v_theta(., 0) = log p0 up to a constant

Weights: w(tau) = 1 - exp(-2 tau) for the score losses; residuals R (which grow like 1/tau as tau -> 0)
enter as (w(tau) R)^2, and the heat-ball residual is scaled to pointwise units (x 4 / (C1 r^2)), so that
lam_fp and lam_hb are comparable.
"""
import math
import torch
from .heatball import C1, admissible_r
from .residuals import hb_residual_v, pointwise_residual_v


def wtau(tau):
    return 1 - torch.exp(-2 * tau)


def dsm(model, gmm, drift, x0, tau):
    xt, tgt, _ = gmm.dsm_pair(x0.double(), tau.double(), drift)
    xt, tgt = xt.to(x0.dtype), tgt.to(x0.dtype)
    s = model.score(xt, tau, create_graph=True)
    return (wtau(tau) * ((s - tgt) ** 2).sum(1)).mean()


def ssm(model, x, tau):
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        v = model(xr, tau)
        s, = torch.autograd.grad(v.sum(), xr, create_graph=True)
        xi = torch.randn_like(xr)
        hv, = torch.autograd.grad((s * xi).sum(), xr, create_graph=True)
        obj = (hv * xi).sum(1) + 0.5 * (s * s).sum(1)
    return (wtau(tau) * obj).mean()


def fp(model, drift, x, tau, lap='exact'):
    R = pointwise_residual_v(model, drift, drift.div, x, tau, lap=lap, create_graph=True)
    return ((wtau(tau) * R) ** 2).mean()


LAST = None


def hb(model, drift, x0, tau0, r, N=16, Np=16, mode='ad', div_mode='analytic', ustat=True, clip=0.0, robust='none', kappa=100.0):
    d = x0.shape[1]
    rr = admissible_r(tau0, r)
    scale = 4.0 / (C1(d) * rr ** 2)
    kw = dict(N=N, Np=Np, mode=mode, div_mode=div_mode, create_graph=True)
    v0 = model(x0, tau0)
    R1 = scale * hb_residual_v(model, drift, drift.div, x0, tau0, rr, v0=v0, **kw)
    w2 = wtau(tau0) ** 2
    if ustat:
        R2 = scale * hb_residual_v(model, drift, drift.div, x0, tau0, rr, v0=v0, **kw)
        global LAST
        LAST = (R1.detach(), R2.detach(), w2.detach(), x0.detach(), tau0.detach())   # for diagnostics (HB_TRACE)
        c = w2 * R1 * R2
        if robust == 'norm':
            # scale-normalized residual: R / (1 + |s(x0)|^2) (normalizer detached).  The residual contains |grad v|^2,
            # so where the model develops a spuriously large score the raw loss grows like |s|^4 and its gradient
            # like |s|^3; normalizing keeps the growth quadratic.  Zero at the true solution as before.
            with torch.enable_grad():
                xr = x0.detach().requires_grad_(True)
                s0, = torch.autograd.grad(model(xr, tau0).sum(), xr)
            c = c / (1.0 + (s0.detach() ** 2).sum(1)) ** 2
        elif robust == 'huber':
            # adaptive pseudo-Huber weights: anchors whose term exceeds kappa x median |term| are down-weighted
            # (weight kappa*m/|c|, detached), so their contribution is bounded by about kappa x median
            K = kappa * c.detach().abs().median().clamp_min(1e-12)
            c = c / (1.0 + c.detach().abs() / K)
        if clip > 0:      # robust variant: per-anchor terms limited to clip x median |term| (biased, bounded)
            K = clip * c.detach().abs().median().clamp_min(1e-12)
            c = torch.clamp(c, -K, K)
        return c.mean()
    return (w2 * R1 ** 2).mean()


def bnd(model, x, logp0):
    t0 = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    diff = model(x, t0) - logp0
    return ((diff - diff.mean()) ** 2).mean()
