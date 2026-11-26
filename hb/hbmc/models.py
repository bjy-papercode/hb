"""
hbmc.models -- scalar log-density network v_theta(x, tau); score s_theta = grad_x v_theta.

One architecture for every method (DSM, SSM, FP residual, heat-ball residual) so that comparisons
differ only in the training objective.  The backbone mirrors networks.py (MLP + ResBlocks with
LayerNorm and FiLM time conditioning).  An optional analytic Gaussian term -|x|^2 / (2 c(tau)),
c(tau) = softplus(c0) + 2 tau, anchors the far field (as in networks_mnist_v4.LogPNetV2); it is
part of v_theta, so every identity is applied to the full v_theta.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        a = (t[:, None] * 20.0) * fr[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, h, td):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.f1, self.f2 = nn.Linear(h, h), nn.Linear(h, h)
        self.t = nn.Linear(td, 2 * h)

    def forward(self, x, te):
        h = F.silu(self.f1(self.n1(x)))
        h = self.f2(self.n2(h))
        s, b = self.t(te).chunk(2, -1)
        return x + h * (1 + s) + b


class VNet(nn.Module):
    def __init__(self, d=2, hidden=256, ted=128, blocks=4, gauss_ref=True, ref='heat'):
        """ref: Gaussian reference term of the far field (the MLP part is v = a.x + O(1) at infinity, since every
        block adds a LayerNorm-bounded increment):
             'heat'     -|x|^2 / (2 (c0 + 2 tau)), scalar c0 (original; variance growth of the heat equation)
             'ou_diag'  -sum_i x_i^2 / (2 c_i(tau)), c_i(tau) = e^{-2 tau} c0_i + 1 - e^{-2 tau} (OU propagation of a
                        diagonal Gaussian), c0_i learnable per coordinate
             'gauge'    -|x|^2 / 2, the stationary law of the isotropic OU process (v = net + phi~)
             'radial'   'heat' plus a learnable radial profile -d m(|x|^2/d, tau) (see below)"""
        super().__init__()
        self.ref = ref
        if ref == 'ou_diag':
            self.raw_c0d = nn.Parameter(torch.full((d,), math.log(math.expm1(1.0))))
        if ref == 'radial':
            # 'radial': the original reference plus a learnable radial profile  - d m(rho, tau),  rho = |x|^2 / d.
            # m is a small SiLU MLP without LayerNorm on (rho, log tau): unbounded in rho, so the radial tail of v is not
            # confined to a Gaussian envelope.  Zero-initialised: identical to 'heat' at initialisation.
            self.rad = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.rad[-1].weight); nn.init.zeros_(self.rad[-1].bias)
        self.te = nn.Sequential(SinEmb(ted), nn.Linear(ted, 2 * ted), nn.SiLU(), nn.Linear(2 * ted, ted))
        self.inp = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList([Block(hidden, ted) for _ in range(blocks)])
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.gauss_ref = gauss_ref
        self.raw_c0 = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def forward(self, x, tau):
        tau = tau.reshape(-1).expand(x.shape[0]).to(x.dtype)
        te = self.te(tau)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, te)
        v = self.head(h).squeeze(-1)
        if self.gauss_ref:
            if self.ref == 'ou_diag':
                e = torch.exp(-2.0 * tau)[:, None]
                c = e * F.softplus(self.raw_c0d)[None, :] + (1.0 - e)
                v = v - (x * x / (2 * c)).sum(1)
            elif self.ref == 'gauge':
                v = v - 0.5 * (x * x).sum(1)
            elif self.ref == 'radial':
                r2 = (x * x).sum(1); c = F.softplus(self.raw_c0) + 2.0 * tau
                m = self.rad(torch.stack([r2 / x.shape[1], torch.log(tau)], 1)).squeeze(-1)
                v = v - r2 / (2 * c) - x.shape[1] * m
            else:
                c = F.softplus(self.raw_c0) + 2.0 * tau
                v = v - (x * x).sum(1) / (2 * c)
        return v

    def score(self, x, tau, create_graph=False):
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            v = self(xr, tau)
            s, = torch.autograd.grad(v.sum(), xr, create_graph=create_graph)
        return s
