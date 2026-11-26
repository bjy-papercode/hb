"""Entropy-tilted two-dimensional law q = p w / Z (Section 'An entropy-tilted training law'), as a dataset object.

Used by the joint experiment (experiments/joint_eval.py): the 64-dimensional tilted law is the product over the blocks of
the two-dimensional tilted law, so the exact finite-volume reference of hbmc.problem.Problem applies to it unchanged.
    w(y) = 1 + alpha * l(log theta - log p(y)),  l(z) = 0 (z <= 0), z^2/(2 delta) (0 < z < delta), z - delta/2 (z >= delta),
    log theta = qtheta-quantile of log p under p  (Proposition 'Properties of the tilted law').
Sampling: sampling-importance-resampling from p with m = sir * n candidates (bias O(1/m)); the density log q is exact up to
the Monte Carlo normalization Z (n_cal samples), which the finite-volume solver renormalizes anyway.
"""
import math
import torch

DT = torch.float64


def ell(z, delta):
    return torch.where(z <= 0, torch.zeros_like(z), torch.where(z < delta, z * z / (2 * delta), z - delta / 2))


def dell(z, delta):
    return torch.where(z <= 0, torch.zeros_like(z), torch.where(z < delta, z / delta, torch.ones_like(z)))


class TiltedGMM2D:
    def __init__(self, G, alpha=0.5, qtheta=0.2, delta=0.1, n_cal=400000, sir=20, seed=4321):
        self.G, self.alpha, self.delta, self.sir = G, float(alpha), float(delta), int(sir)
        st = torch.random.get_rng_state(); torch.manual_seed(seed)
        x = G.sample0(n_cal)
        lp = self.logp_base(x)
        self.log_theta = float(torch.quantile(lp, qtheta))
        self.Z = float(self.w(x).mean())
        torch.random.set_rng_state(st)
        self.d = G.d; self.nb = G.nb

    # ---- base law and weight
    def logp_base(self, x):
        from hbmc import drifts
        return self.G.logp_score(x.double(), torch.zeros(len(x), dtype=DT), drifts.make('ou'), need_score=False)

    def score_base(self, x):
        from hbmc import drifts
        return self.G.logp_score(x.double(), torch.zeros(len(x), dtype=DT), drifts.make('ou'))[1]

    def w(self, x):
        return 1 + self.alpha * ell(self.log_theta - self.logp_base(x), self.delta)

    def grad_logw(self, x):
        """grad log w = -alpha l'(log theta - log p) grad log p / w"""
        lp, s = self.logp_score_base(x)
        z = self.log_theta - lp
        ww = 1 + self.alpha * ell(z, self.delta)
        return (-self.alpha * dell(z, self.delta) / ww)[:, None] * s

    def logp_score_base(self, x):
        from hbmc import drifts
        return self.G.logp_score(x.double(), torch.zeros(len(x), dtype=DT), drifts.make('ou'))

    # ---- dataset interface used by hbmc.problem.Problem
    def sample0(self, n, device=None):
        c = self.G.sample0(self.sir * n)
        k = torch.multinomial(self.w(c), n, replacement=True)
        x = c[k]
        return x if device is None else x.to(device)

    def logp_score(self, x, tau, drift, chunk=4096, need_score=True):
        tau = torch.as_tensor(tau, dtype=DT)
        assert float(tau.abs().max()) == 0.0, 'TiltedGMM2D: only the initial law is available in closed form'
        lq = self.logp_base(x) + torch.log(self.w(x)) - math.log(self.Z)
        if not need_score:
            return lq
        return lq, self.score_base(x) + self.grad_logw(x)

    def responsibilities0(self, x):
        return self.G.responsibilities0(x)
