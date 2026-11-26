"""
hbmc.problem -- one forward process + data distribution, shared by every method.

Provides training samples (x, tau), reference scores / log-densities at evaluation times, samples
of p_{tau_max} to start generation, and data samples for sample-quality metrics.
  linear drift    -> everything exact (hbmc.linear_gmm)
  nonlinear drift -> training samples from Euler-Maruyama paths (dt = 1e-3); references and
                     p_{tau_max} from the Fokker-Planck grid solver (hbmc.fp_grid, cached to disk)
"""
import os, math, hashlib
import numpy as np
import torch
from . import datasets2d, drifts
from .fp_grid import FPGrid, GridReference
from .sampling import forward_bank


class Problem:
    def __init__(self, dataset, drift_name, tau_min=1e-3, tau_max=1.0, eval_taus=(0.05, 0.1, 0.3, 0.6, 1.0),
                 grid_N=400, grid_L=6.0, cache_dir='cache', n_bank=20000, seed=0, drift_obj=None, dense_ref=False):
        self.name = f'{dataset}/{drift_name}'
        self.G = datasets2d.get(dataset)
        self.drift = drift_obj if drift_obj is not None else drifts.make(drift_name, d=2)
        self.linear = getattr(self.drift, 'linear', False)
        self.tau_min, self.tau_max = tau_min, tau_max
        self.eval_taus = [t for t in eval_taus if t <= tau_max + 1e-12]
        if tau_max not in self.eval_taus:
            self.eval_taus.append(tau_max)
        self.logp0 = lambda x: self.G.logp_score(x.double(), torch.zeros(len(x), dtype=torch.float64),
                                                 drifts.make('ou'), need_score=False)
        if not self.linear:
            # the sample bank is drawn with its own fixed seed; the global RNG state (which carries the
            # run's --seed: network initialization, batches) is restored afterwards.  Before this fix the
            # global seed was reset to `seed`, so all runs with a nonlinear drift were identical across seeds.
            _state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            self.bank, self.bank_t = forward_bank(self.drift, self.G.sample0(n_bank), tau_max)
            torch.random.set_rng_state(_state)
            os.makedirs(cache_dir, exist_ok=True)
            self.grid_dt = 1e-3
            snap = sorted(set(self.eval_taus) | (set(np.geomspace(tau_min, tau_max, 60).tolist()) if dense_ref else set()))
            key = hashlib.md5(f'{dataset}{drift_name}{tau_max}{grid_N}{grid_L}{snap}'.encode()).hexdigest()[:10]
            path = os.path.join(cache_dir, f'grid_{dataset}_{drift_name}_{key}.npz')
            grid = FPGrid(2, grid_L, grid_N, lambda x, t: self.drift(torch.as_tensor(x), t).numpy(),
                          time_dependent=isinstance(self.drift, drifts.LearnedDrift))
            if os.path.exists(path):
                z = np.load(path); sols = {float(t): z[f'u{i}'] for i, t in enumerate(z['taus'])}
            else:
                pts = grid.points()
                u0 = torch.exp(self.logp0(torch.as_tensor(pts))).numpy()
                sols = grid.solve(u0, snap, dt=self.grid_dt)
                np.savez(path, taus=np.array(sorted(sols)), **{f'u{i}': sols[t] for i, t in enumerate(sorted(sols))})
            self.ref = GridReference(grid, sols)

    # ---------- training samples ----------
    def train_batch(self, n, dtype=torch.float32, tau_lo=None):
        lo = max(self.tau_min, tau_lo or 0.0)
        lt = torch.empty(n, dtype=torch.float64).uniform_(math.log(lo), math.log(self.tau_max))
        tau = torch.exp(lt)
        if self.linear:
            x0 = self.G.sample0(n)
            Phi, Sig = self.drift.phi_sigma(tau)
            L = torch.linalg.cholesky(Sig)
            xb = x0.reshape(n, -1, 2)
            x = (Phi @ xb[..., None]).squeeze(-1) + (L @ torch.randn(n, xb.shape[1], 2, 1, dtype=torch.float64)).squeeze(-1)
            return x.reshape(n, -1).to(dtype), tau.to(dtype), x0.to(dtype)
        k = torch.searchsorted(self.bank_t.double(), tau).clamp(max=len(self.bank_t) - 1)
        j = torch.randint(0, self.bank.shape[1], (n,))
        return self.bank[k, j].to(dtype), self.bank_t[k].to(dtype), None

    def data_samples(self, n):
        return self.G.sample0(n)

    # ---------- references ----------
    def ref_samples(self, n, tau, seed=0):
        if self.linear:
            return self.G.sample_tau(n, tau, self.drift)[0]
        return torch.as_tensor(self.ref.sample(n, tau, np.random.default_rng(seed)))

    def ref_score_logp(self, x, tau):
        if self.linear:
            lp, s = self.G.logp_score(x.double(), torch.full((len(x),), tau, dtype=torch.float64), self.drift)
            return s, lp
        xn = x.double().numpy()
        return torch.as_tensor(self.ref.score(xn, tau)), torch.as_tensor(self.ref.logp(xn, tau))

    def ref_score_fn(self, x, tau_vec):
        """Reference score at arbitrary times (linear: exact; nonlinear: nearest saved grid time)."""
        if self.linear:
            return self.G.logp_score(x.double(), tau_vec.double(), self.drift)[1].to(x.dtype)
        t = float(tau_vec[0])
        return torch.as_tensor(self.ref.score(x.double().numpy(), t)).to(x.dtype)
