"""
hbmc.localgauss -- small-time local-Gaussian score of p_tau for a nonlinear drift and a known Gaussian-mixture data law.

For small tau the transition kernel of  dX = f(X) dtau + sqrt(2) dW  is close to a Gaussian whose mean is the flow
Phi_tau(x0) and whose covariance C solves the linearized (Lyapunov) equation along the flow,
    m' = f(m),  M' = Df(m) M,  C' = Df(m) C + C Df(m)^T + 2 I,   m(0) = x0, M(0) = I, C(0) = 0.
With p_0 = sum_k pi_k N(mu_k, S_k):
  'quad'   p_tau(y) ~ sum_k pi_k sum_j w_j N(y; m(x_kj), C(x_kj))     (Gauss-Hermite nodes x_kj of component k)
  'linear' p_tau(y) ~ sum_k pi_k N(y; m(mu_k), M S_k M^T + C(mu_k))   (flow linearized at the component mean; closed form,
           the analogue of the exact component posteriors of the OU case)
The score is the gradient of the log of this mixture (closed form).  Both are exact as tau -> 0; 'quad' keeps the
nonlinearity of the flow across a component and is limited only by the Gaussian kernel, 'linear' also by the curvature
of the flow over the component width.  Used as the small-tau part of a hybrid generator (network for tau > tau_c).
Works in 2-D (one block); hbsm_blocks64 applies it blockwise to the product law of the 64-D problem.
"""
import math
import numpy as np
import torch

__all__ = ["LocalGauss2D", "small_tau_reference"]


class LocalGauss2D:
    def __init__(self, G, f, jac, kind='quad', n_gh=12, n_rk=40):
        """G: hbmc.linear_gmm.BlockGMM with one 2-D block; f(x) (n,2); jac(x) (n,2,2)."""
        self.mu = G.mu.double()                      # (K, 2)
        self.S = G.S0[:, 0].double()                 # (K, 2, 2)
        self.w = G.w.double()
        self.f, self.jac, self.kind, self.n_rk = f, jac, kind, n_rk
        K = len(self.w)
        if kind == 'quad':
            t, wt = np.polynomial.hermite_e.hermegauss(n_gh)                  # probabilists' Hermite: N(0,1)
            t = torch.as_tensor(t); wt = torch.as_tensor(wt / wt.sum())
            Z = torch.stack(torch.meshgrid(t, t, indexing='ij'), -1).reshape(-1, 2)
            W2 = (wt[:, None] * wt[None, :]).reshape(-1)
            L = torch.linalg.cholesky(self.S)                                   # (K,2,2)
            self.nodes = (self.mu[:, None, :] + torch.einsum('kij,nj->kni', L, Z)).reshape(-1, 2)
            self.logw = (torch.log(self.w)[:, None] + torch.log(W2)[None, :]).reshape(-1)
        else:
            self.nodes = self.mu.clone(); self.logw = torch.log(self.w)
        self._cache = {}

    def _moments(self, tau):
        """flow mean, tangent M and covariance C of the nodes at time tau (RK4 on the joint ODE)."""
        key = round(float(tau), 12)
        if key in self._cache:
            return self._cache[key]
        n = len(self.nodes); m = self.nodes.clone(); M = torch.eye(2, dtype=m.dtype).expand(n, 2, 2).clone()
        C = torch.zeros(n, 2, 2, dtype=m.dtype); I2 = torch.eye(2, dtype=m.dtype)
        h = float(tau) / self.n_rk

        def rhs(m, M, C):
            J = self.jac(m)
            return self.f(m), J @ M, J @ C + C @ J.transpose(1, 2) + 2 * I2

        for _ in range(self.n_rk):
            k1 = rhs(m, M, C); k2 = rhs(*(a + 0.5 * h * b for a, b in zip((m, M, C), k1)))
            k3 = rhs(*(a + 0.5 * h * b for a, b in zip((m, M, C), k2))); k4 = rhs(*(a + h * b for a, b in zip((m, M, C), k3)))
            m, M, C = (a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip((m, M, C), k1, k2, k3, k4))
        if self.kind == 'linear':
            C = M @ self.S @ M.transpose(1, 2) + C
        C = 0.5 * (C + C.transpose(1, 2))
        self._cache[key] = (m, C)
        return m, C

    def score_logp(self, y, tau):
        """score (n,2) and log-density (n,) of the local-Gaussian mixture at time tau (scalar)."""
        m, C = self._moments(tau)
        Ci = torch.linalg.inv(C); ld = torch.logdet(C)
        dy = y.double()[:, None, :] - m[None]                                    # (n, J, 2)
        Cd = torch.einsum('jab,njb->nja', Ci, dy)
        logc = self.logw[None] - 0.5 * (dy * Cd).sum(-1) - 0.5 * ld[None] - math.log(2 * math.pi)
        lp = torch.logsumexp(logc, 1); r = torch.softmax(logc, 1)
        return -(r[..., None] * Cd).sum(1), lp


def small_tau_reference(dataset, drift, taus=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05), grid_N=600, grid_L=6.0, dt=1e-4,
                        cache_dir='cache'):
    """2-D finite-volume reference (hbmc.fp_grid) with a fine time step and snapshots at small tau."""
    import os, hashlib
    from .fp_grid import FPGrid, GridReference
    from . import datasets2d, drifts
    G = datasets2d.get(dataset)
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f'small{dataset}{type(drift).__name__}{getattr(drift, "omega", "")}{grid_N}{grid_L}{dt}{list(taus)}'.encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f'smallref_{dataset}_{key}.npz')
    grid = FPGrid(2, grid_L, grid_N, lambda x, t: drift(torch.as_tensor(x), t).numpy())
    if os.path.exists(path):
        z = np.load(path); sols = {float(t): z[f'u{i}'] for i, t in enumerate(z['taus'])}
    else:
        pts = grid.points()
        u0 = torch.exp(G.logp_score(torch.as_tensor(pts), torch.zeros(len(pts), dtype=torch.float64), drifts.make('ou'),
                                    need_score=False)).numpy()
        sols = grid.solve(u0, list(taus), dt=dt)
        np.savez(path, taus=np.array(sorted(sols)), **{f'u{i}': sols[t] for i, t in enumerate(sorted(sols))})
    return GridReference(grid, sols), G
