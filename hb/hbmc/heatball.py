"""
hbmc.heatball -- exact samplers and closed-form constants for the unit heat ball.

Unit heat ball (diffusion coefficient 1, i.e. d_tau u = Laplace u):
    E1 = {(z, sigma): 0 < sigma < 1/(4 pi), |z| <= R(sigma)},  R(sigma)^2 = -2 d sigma log(4 pi sigma)
    psi0(z, sigma) = (R(sigma)^2 - |z|^2) / (4 sigma)

Theorem (Beta-Gamma family).  For weights |z|^{2a} psi0^b sigma^{-c} the normalized measure
mu_{a,b,c} factorizes:   zhat ~ Unif(S^{d-1}),  t = |z|^2/R^2 ~ Beta(d/2+a, b+1),
                         Y = -log(4 pi sigma) ~ Gamma(d/2+a+b+1, rate d/2+a-c+1),
and Z_{a,b,c} = int_{E1} |z|^{2a} psi0^b sigma^{-c} has a closed form (Z_family).

Members used by the heat-ball identities:
    Watson / kappa  (a,b,c) = (1,0,2):  M_r[F] = E[F(x0 + r z, tau0 - r^2 sigma)]
    source  / psi   (a,b,c) = (0,1,0):  S_r[h] = (C1 r^2 / 4) E[h(x0 + r sqrt(U) z, tau0 - r^2 U sigma)]
    divergence IBP  (a,b,c) = (1/2,0,1): S_r[-div G] = -(d r Z/2) E[ 1/2 (G(x0+rUz)-G(x0-rUz)) . zhat ]
No truncation of sigma, no numerical inverse CDF, unit weights.
"""
import math
import torch

__all__ = ["R2", "C1", "Z_family", "sample_family", "sample_watson", "sample_source",
           "sample_ibp", "E1_volume", "sigma_star", "R_max", "admissible_r"]


def R2(sigma, d):
    return -2.0 * d * sigma * torch.log(4.0 * math.pi * sigma)


def C1(d):
    """C1 = int_{E1} |z|^2/sigma = (1/pi) (d/(d+2))^{d/2+2}."""
    return (d / (d + 2.0)) ** (d / 2.0 + 2.0) / math.pi


def Z_family(d, a, b, c):
    """Closed-form normalizing constant Z_{a,b,c} (float)."""
    m = d / 2.0 + a + b
    lam = d / 2.0 + a - c + 1.0
    if lam <= 0 or d / 2.0 + a <= 0 or b <= -1:
        raise ValueError("parameters outside the admissible range")
    lg = math.lgamma
    logZ = ((d / 2.0) * math.log(math.pi) - lg(d / 2.0)
            + lg(d / 2.0 + a) + lg(b + 1.0) - lg(d / 2.0 + a + b + 1.0)
            + m * math.log(2.0 * d) - b * math.log(4.0) - lam * math.log(4.0 * math.pi)
            + lg(m + 1.0) - (m + 1.0) * math.log(lam))
    return math.exp(logZ)


def E1_volume(d):
    return (d / 2.0) ** (d / 2.0) / (4.0 * math.pi * (d / 2.0 + 1.0) ** (d / 2.0 + 1.0))


def sigma_star():
    return 1.0 / (4.0 * math.pi * math.e)


def R_max(d):
    return math.sqrt(d / (2.0 * math.pi * math.e))


def sample_family(d, a, b, c, n, device=None, dtype=torch.float32, generator=None):
    """Exact i.i.d. samples of mu_{a,b,c}. Returns z (n,d), sigma (n,), zhat (n,d)."""
    out_device = torch.device(device) if device is not None else torch.device("cpu")
    # Apple MPS has no float64 and no Gamma/Beta samplers: draw on the CPU in float64, then move (the draw is cheap)
    device = out_device if out_device.type in ("cpu", "cuda") else torch.device("cpu")
    shape = torch.full((n,), d / 2.0 + a + b + 1.0, dtype=torch.float64, device=device)
    rate = d / 2.0 + a - c + 1.0
    Y = torch._standard_gamma(shape) / rate if generator is None else \
        torch.distributions.Gamma(shape, torch.full_like(shape, rate)).sample()
    sigma = torch.exp(-Y) / (4.0 * math.pi)
    beta = torch.distributions.Beta(torch.tensor(d / 2.0 + a, dtype=torch.float64, device=device),
                                    torch.tensor(b + 1.0, dtype=torch.float64, device=device))
    t = beta.sample((n,))
    g = torch.randn(n, d, dtype=torch.float64, device=device, generator=generator)
    zhat = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
    rad = torch.sqrt(t * (2.0 * d * sigma * Y))          # R^2 = 2 d sigma Y
    z = rad[:, None] * zhat
    return (z.to(dtype=dtype).to(out_device), sigma.to(dtype=dtype).to(out_device), zhat.to(dtype=dtype).to(out_device))


def sample_watson(d, n, **kw):   # kappa = |z|^2 / (4 sigma^2) on E1
    z, s, _ = sample_family(d, 1.0, 0.0, 2.0, n, **kw)
    return z, s


def sample_source(d, n, **kw):   # psi0-weighted measure on E1
    z, s, _ = sample_family(d, 0.0, 1.0, 0.0, n, **kw)
    return z, s


def sample_ibp(d, n, **kw):      # |z| / sigma-weighted measure on E1
    return sample_family(d, 0.5, 0.0, 1.0, n, **kw)


def admissible_r(tau0, r, frac=0.999):
    """Clip the radius so that the heat ball stays in {tau > 0}: r^2 <= 4 pi tau0."""
    rmax = torch.sqrt(4.0 * math.pi * tau0) * frac
    return torch.minimum(torch.as_tensor(r, dtype=tau0.dtype, device=tau0.device).expand_as(tau0), rmax)
