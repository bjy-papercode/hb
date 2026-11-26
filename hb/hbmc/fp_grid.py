"""
hbmc.fp_grid -- reference solutions of the Fokker-Planck equation for nonlinear / learned drifts.

    d_tau u = Laplace u - div(f u)       on [-L, L]^d (d = 1, 2), zero-flux boundary.

Finite volumes with Scharfetter-Gummel (exponentially fitted) fluxes, which stay accurate and
positivity-friendly for large drift; Rannacher start-up (backward Euler) followed by
Crank-Nicolson in time.  Drifts may depend on tau (learned drifts): the matrix is reassembled
when needed.  Outputs: density snapshots, log-density and score interpolators, exact-in-law
sampling from the grid density.  Validated against exact Gaussian-mixture solutions in
experiments/check_fp_grid.py.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator


def _bern(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - 0.5 * x[small]
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


class FPGrid:
    def __init__(self, d, L, N, drift, time_dependent=False):
        """drift(x, tau) -> (n, d) numpy array; x is (n, d)."""
        assert d in (1, 2)
        self.d, self.L, self.N = d, L, N
        self.h = 2 * L / N
        self.c = -L + self.h * (np.arange(N) + 0.5)            # cell centres
        self.drift = drift
        self.time_dependent = time_dependent
        self._A = None

    # ---------- assembly ----------
    def _assemble(self, tau):
        N, h, d = self.N, self.h, self.d
        faces = -self.L + h * np.arange(1, N)                    # interior faces
        rows, cols, vals = [], [], []
        if d == 1:
            P = h * self.drift(faces[:, None], tau)[:, 0]
            bm, bp = _bern(-P), _bern(P)
            i = np.arange(N - 1)
            # flux J_{i+1/2} = (bm u_i - bp u_{i+1})/h ; du_i/dt -= J/h, du_{i+1}/dt += J/h
            for (r_, c_, v_) in [(i, i, -bm), (i, i + 1, bp), (i + 1, i, bm), (i + 1, i + 1, -bp)]:
                rows.append(r_); cols.append(c_); vals.append(v_ / h ** 2)
            n = N
        else:
            idx = np.arange(N * N).reshape(N, N)                 # idx[i, j], i -> x, j -> y
            fx, fy = np.meshgrid(faces, self.c, indexing='ij')   # x-faces: (N-1, N)
            P = h * self.drift(np.stack([fx.ravel(), fy.ravel()], 1), tau)[:, 0].reshape(N - 1, N)
            bm, bp = _bern(-P), _bern(P)
            a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
            for (r_, c_, v_) in [(a, a, -bm.ravel()), (a, b, bp.ravel()), (b, a, bm.ravel()), (b, b, -bp.ravel())]:
                rows.append(r_); cols.append(c_); vals.append(v_ / h ** 2)
            gx, gy = np.meshgrid(self.c, faces, indexing='ij')   # y-faces: (N, N-1)
            P = h * self.drift(np.stack([gx.ravel(), gy.ravel()], 1), tau)[:, 1].reshape(N, N - 1)
            bm, bp = _bern(-P), _bern(P)
            a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
            for (r_, c_, v_) in [(a, a, -bm.ravel()), (a, b, bp.ravel()), (b, a, bm.ravel()), (b, b, -bp.ravel())]:
                rows.append(r_); cols.append(c_); vals.append(v_ / h ** 2)
            n = N * N
        return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    def points(self):
        if self.d == 1:
            return self.c[:, None]
        X, Y = np.meshgrid(self.c, self.c, indexing='ij')
        return np.stack([X.ravel(), Y.ravel()], 1)

    # ---------- time stepping ----------
    def solve(self, u0, taus, dt=2e-3, n_be=4):
        """u0: cell values (N^d,) at tau=0.  taus: save times, snapped to multiples of dt.
        Returns dict tau -> u (keys are the snapped times).  One LU factorization per scheme
        for time-independent drifts."""
        u = np.asarray(u0, float).copy(); u /= u.sum() * self.h ** self.d
        n = u.size; I = sp.identity(n, format='csc')
        steps = sorted(set(max(1, int(round(t / dt))) for t in taus if t > 0))
        out = {0.0: u.copy()} if any(t <= 0 for t in taus) else {}
        cache = {}
        for k in range(1, steps[-1] + 1):
            be = k <= n_be
            if self.time_dependent or be not in cache:
                A = self._assemble((k - (0 if be else 0.5)) * dt)
                cache[be] = (spla.splu(I - (1.0 if be else 0.5) * dt * A), None if be else I + 0.5 * dt * A)
            lu, rhs = cache[be]
            u = np.maximum(lu.solve(u if be else rhs @ u), 0.0)
            if self.time_dependent:
                cache.clear()
            if k in steps:
                out[round(k * dt, 10)] = u.copy()
        return out

    # ---------- post-processing ----------
    def log_score_interp(self, u, floor=1e-300):
        lu = np.log(np.maximum(u, floor))
        if self.d == 1:
            g = np.gradient(lu, self.h)
            return (RegularGridInterpolator((self.c,), lu, bounds_error=False, fill_value=None),
                    RegularGridInterpolator((self.c,), g, bounds_error=False, fill_value=None), None)
        LU = lu.reshape(self.N, self.N)
        gx, gy = np.gradient(LU, self.h)
        mk = lambda A: RegularGridInterpolator((self.c, self.c), A, bounds_error=False, fill_value=None)
        return mk(LU), mk(gx), mk(gy)

    def sample(self, u, n, rng):
        p = np.maximum(u, 0) / np.maximum(u, 0).sum()
        k = rng.choice(p.size, size=n, p=p)
        if self.d == 1:
            return (self.c[k] + (rng.random(n) - 0.5) * self.h)[:, None]
        i, j = np.unravel_index(k, (self.N, self.N))
        return np.stack([self.c[i], self.c[j]], 1) + (rng.random((n, 2)) - 0.5) * self.h


class GridReference:
    """Wraps an FPGrid solution: score(x, tau), logp(x, tau), linear interpolation in log(tau)
    between saved snapshots (use dense snapshots when the reference drives a sampler)."""

    def __init__(self, grid, sols):
        self.grid, self.sols = grid, sols
        self.taus = np.array(sorted(t for t in sols if t > 0))
        self.interp = {t: grid.log_score_interp(sols[t]) for t in self.taus}

    def _bracket(self, tau):
        if tau <= self.taus[0]:
            return self.taus[0], self.taus[0], 0.0
        if tau >= self.taus[-1]:
            return self.taus[-1], self.taus[-1], 0.0
        j = np.searchsorted(self.taus, tau)
        ta, tb = self.taus[j - 1], self.taus[j]
        return ta, tb, (np.log(tau) - np.log(ta)) / (np.log(tb) - np.log(ta))

    def _eval(self, t, x, which):
        lu, gx, gy = self.interp[t]
        if which == 'logp':
            return lu(x)
        return gx(x)[:, None] if self.grid.d == 1 else np.stack([gx(x), gy(x)], 1)

    def score(self, x, tau):
        ta, tb, w = self._bracket(tau)
        a = self._eval(ta, x, 'score')
        return a if w == 0 else (1 - w) * a + w * self._eval(tb, x, 'score')

    def logp(self, x, tau):
        ta, tb, w = self._bracket(tau)
        a = self._eval(ta, x, 'logp')
        return a if w == 0 else (1 - w) * a + w * self._eval(tb, x, 'logp')

    def _nearest(self, tau):
        return self.taus[np.argmin(np.abs(self.taus - tau))]

    def sample(self, n, tau, rng):
        return self.grid.sample(self.sols[self._nearest(tau)], n, rng)
