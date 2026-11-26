"""
hbmc.linear_gmm -- exact marginals of a Gaussian mixture under a linear forward process

    dx = B x dtau + sqrt(2) dW        (tau = cumulative-variance time, diffusion coefficient 1)

with B block-diagonal in 2x2 blocks (d even).  This covers isotropic and anisotropic
Ornstein-Uhlenbeck drifts and rotations (B = -A + omega J).  For data p0 = sum_k pi_k N(mu_k, S_k)
with block-diagonal S_k the marginal is again a Gaussian mixture:

    p_tau = sum_k pi_k N(Phi mu_k, Phi S_k Phi^T + Sigma(tau)),
    Phi = exp(B tau),   Sigma(tau) = int_0^tau Phi(s) 2I Phi(s)^T ds    (Van Loan).

Everything needed for exact references is provided: log density, score, sampling at any tau,
and denoising-score-matching targets of the Gaussian transition kernel.
"""
import math
import torch

__all__ = ["LinearDrift", "BlockGMM"]


def _blocks(x):                     # (..., d) -> (..., d/2, 2)
    return x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)


def _unblocks(x):
    return x.reshape(*x.shape[:-2], x.shape[-2] * 2)


class LinearDrift:
    """f(x) = B x with B = blockdiag(B_1..B_{d/2}), each B_j a 2x2 matrix."""

    def __init__(self, B_blocks, dtype=torch.float64):
        self.B = torch.as_tensor(B_blocks, dtype=dtype)          # (nb,2,2)
        self.nb = self.B.shape[0]
        self.d = 2 * self.nb
        self.dtype = dtype

    # -- constructors --
    @classmethod
    def ou(cls, d, a=1.0, omega=0.0, a2=None):
        """B_j = [[-a1, -omega], [omega, -a2]]; a may be a scalar or a length-d/2 list for anisotropy."""
        nb = d // 2
        a1 = torch.as_tensor(a, dtype=torch.float64).expand(nb).clone()
        a2_ = a1.clone() if a2 is None else torch.as_tensor(a2, dtype=torch.float64).expand(nb).clone()
        B = torch.zeros(nb, 2, 2, dtype=torch.float64)
        B[:, 0, 0] = -a1; B[:, 1, 1] = -a2_; B[:, 0, 1] = -omega; B[:, 1, 0] = omega
        return cls(B)

    # -- drift --
    def __call__(self, x, tau=None):
        dt = self.dtype if x.device.type != 'mps' else torch.float32      # MPS has no float64
        xb = _blocks(x.to(dt))
        return _unblocks(torch.einsum('bij,nbj->nbi', self.B.to(dt).to(x.device), xb)).to(x.dtype)

    def div(self, x, tau=None):
        return torch.full(x.shape[:1], float(self.B.diagonal(dim1=1, dim2=2).sum()), dtype=x.dtype, device=x.device)

    # -- propagators --
    def phi_sigma(self, tau):
        """tau (n,) -> Phi (n,nb,2,2), Sigma (n,nb,2,2) by Van Loan's block exponential
        (closed form when B is diagonal)."""
        tau = tau.to(self.dtype)
        n = tau.shape[0]
        B = self.B.to(tau.device)
        if bool((B[:, 0, 1] == 0).all() and (B[:, 1, 0] == 0).all()):
            a = -B.diagonal(dim1=1, dim2=2)                                   # (nb,2)
            e = torch.exp(-a[None] * tau[:, None, None])                       # (n,nb,2)
            safe = torch.where(a.abs() < 1e-12, torch.ones_like(a), a)
            var = torch.where(a.abs() < 1e-12, 2.0 * tau[:, None, None].expand_as(e), (1 - e ** 2) / safe[None])
            return torch.diag_embed(e), torch.diag_embed(var)
        M = torch.zeros(self.nb, 4, 4, dtype=self.dtype, device=tau.device)
        M[:, :2, :2] = -B
        M[:, :2, 2:] = 2.0 * torch.eye(2, dtype=self.dtype, device=tau.device)
        M[:, 2:, 2:] = B.transpose(1, 2)
        F = torch.linalg.matrix_exp(tau[:, None, None, None] * M[None])     # (n,nb,4,4)
        F12 = F[..., :2, 2:]; F22 = F[..., 2:, 2:]
        Phi = F22.transpose(-1, -2)
        Sig = Phi @ F12
        Sig = 0.5 * (Sig + Sig.transpose(-1, -2))
        return Phi, Sig


class BlockGMM:
    """Gaussian mixture with block-diagonal (2x2) covariances, propagated exactly by a LinearDrift."""

    def __init__(self, means, covs, weights, dtype=torch.float64):
        self.mu = torch.as_tensor(means, dtype=dtype)               # (K,d)
        K, d = self.mu.shape
        covs = torch.as_tensor(covs, dtype=dtype)
        if covs.dim() == 1:                                         # isotropic std per component
            covs = (covs ** 2)[:, None, None, None] * torch.eye(2, dtype=dtype).expand(K, d // 2, 2, 2)
        elif covs.dim() == 3 and d == 2:                            # (K,2,2) full covariance in 2D
            covs = covs[:, None]
        self.S0 = covs                                              # (K,nb,2,2)
        self.w = torch.as_tensor(weights, dtype=dtype); self.w = self.w / self.w.sum()
        self.K, self.d, self.nb = K, d, d // 2
        self.dtype = dtype

    # ---------- sampling ----------
    def sample0(self, n, device=None):
        k = torch.multinomial(self.w, n, replacement=True)
        L = torch.linalg.cholesky(self.S0[k])                       # (n,nb,2,2)
        eps = torch.randn(n, self.nb, 2, 1, dtype=self.dtype)
        x = self.mu[k] + _unblocks((L @ eps).squeeze(-1))
        return x if device is None else x.to(device)

    def sample_tau(self, n, tau, drift):
        """Exact samples of p_tau; tau scalar or (n,)."""
        tau = torch.as_tensor(tau, dtype=self.dtype).expand(n).clone()
        x0 = self.sample0(n)
        Phi, Sig = drift.phi_sigma(tau)
        L = torch.linalg.cholesky(Sig + 1e-300 * torch.eye(2, dtype=self.dtype))
        eps = torch.randn(n, self.nb, 2, 1, dtype=self.dtype)
        return _unblocks((Phi @ _blocks(x0)[..., None]).squeeze(-1) + (L @ eps).squeeze(-1)), x0

    # ---------- marginal density and score ----------
    def _comp_params(self, tau, drift):
        Phi, Sig = drift.phi_sigma(tau)                             # (n,nb,2,2)
        m = torch.einsum('nbij,kbj->nkbi', Phi, _blocks(self.mu))   # (n,K,nb,2)
        C = Phi[:, None] @ self.S0[None] @ Phi[:, None].transpose(-1, -2) + Sig[:, None]
        return m, C

    def logp_score(self, x, tau, drift, chunk=4096, need_score=True):
        x = x.to(self.dtype); tau = torch.as_tensor(tau, dtype=self.dtype).expand(x.shape[0])
        outs_l, outs_s = [], []
        for i in range(0, x.shape[0], chunk):
            xb = _blocks(x[i:i + chunk]); tb = tau[i:i + chunk]
            m, C = self._comp_params(tb, drift)
            diff = xb[:, None] - m                                  # (n,K,nb,2)
            a_, b_, c_ = C[..., 0, 0], C[..., 0, 1], C[..., 1, 1]      # closed-form 2x2 inverse
            det = a_ * c_ - b_ * b_
            Cd = torch.stack([(c_ * diff[..., 0] - b_ * diff[..., 1]) / det,
                              (a_ * diff[..., 1] - b_ * diff[..., 0]) / det], -1)
            quad = (diff * Cd).sum((-1, -2))
            logdet = (torch.log(det) + 2 * math.log(2 * math.pi)).sum(-1)
            L = torch.log(self.w)[None] - 0.5 * (quad + logdet)     # (n,K)
            lse = torch.logsumexp(L, 1)
            outs_l.append(lse)
            if need_score:
                r = torch.softmax(L, 1)
                outs_s.append(_unblocks(-(r[..., None, None] * Cd).sum(1)))
        lp = torch.cat(outs_l)
        return (lp, torch.cat(outs_s)) if need_score else lp

    def responsibilities0(self, x):
        """Component posteriors under p0 (for mode coverage metrics)."""
        x = x.to(self.dtype)
        diff = _blocks(x)[:, None] - _blocks(self.mu)[None]
        Cinv = torch.linalg.inv(self.S0)[None]
        quad = (diff * (Cinv @ diff[..., None]).squeeze(-1)).sum((-1, -2))
        logdet = torch.logdet(2 * math.pi * self.S0).sum(-1)[None]
        return torch.softmax(torch.log(self.w)[None] - 0.5 * (quad + logdet), 1)

    # ---------- denoising score matching under the exact Gaussian kernel ----------
    def dsm_pair(self, x0, tau, drift):
        """x_tau | x0 ~ N(Phi x0, Sigma).  Returns x_tau, target score -Sigma^{-1}(x_tau - Phi x0), weight tr(Sigma)/d."""
        x0 = x0.to(self.dtype); tau = tau.to(self.dtype)
        Phi, Sig = drift.phi_sigma(tau)
        L = torch.linalg.cholesky(Sig)
        eps = torch.randn(x0.shape[0], self.nb, 2, 1, dtype=self.dtype)
        noise = (L @ eps).squeeze(-1)
        xt = (Phi @ _blocks(x0)[..., None]).squeeze(-1) + noise
        target = -torch.linalg.solve(Sig, noise[..., None]).squeeze(-1)
        lam = Sig.diagonal(dim1=-2, dim2=-1).sum((-1, -2)) / self.d
        return _unblocks(xt), _unblocks(target), lam
