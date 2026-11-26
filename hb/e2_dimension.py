"""E2: dimension dependence.
(a) Bounding-cylinder importance sampling, constant integrand: EXACT chi^2(kappa||q)+1 in closed form
    (|S|/4) E_kappa[|z|^2/sigma^2] with E_kappa[|z|^2/sigma^2] = 8 pi d (d+2)/(d+4) E[Y e^Y],
    E[Y e^Y] = a lam^a (lam-1)^{-a-1}, a = d/2+2, lam = d/2  (finite iff d > 2), against the lower bound
    |S|/|E1| of the paper, and a Monte Carlo check.
(b) Leading-order variance of the Watson term: Var_kappa[F(x0 + r z)] / (r^2 |grad F|^2 C1/2) -> 1 as r -> 0.
(c) Heat ball vs Gaussian semigroup at matched mean time lag, and vs bounding-cylinder IS, d up to 1024,
    several r, mean +- sd over repetitions.
Output: results_e2.json"""
import sys, os, json, math
import numpy as np, torch
from scipy.special import gammaln
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import sample_watson, C1, E1_volume, R_max
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); out = {'a': [], 'b': [], 'c': []}


def logS(d):
    return (d / 2) * math.log(math.pi) - gammaln(d / 2 + 1) + d * math.log(R_max(d)) - math.log(4 * math.pi)


for d in [3, 4, 8, 16, 64, 256, 1024]:
    a, lam = d / 2 + 2, d / 2
    EYeY = math.exp(math.log(a) + a * math.log(lam) - (a + 1) * math.log(lam - 1))
    exact = math.exp(logS(d)) / 4 * 8 * math.pi * d * (d + 2) / (d + 4) * EYeY
    bound = math.exp((d / 2 + 1) * math.log(d / 2 + 1) - d / 2 - gammaln(d / 2 + 1))
    z, s = sample_watson(d, int(min(400000, 4e7 / d)), dtype=torch.float64)
    mc = math.exp(logS(d)) / 4 * float(((z ** 2).sum(1) / s ** 2).mean())
    row = dict(d=d, chi2p1_exact=exact, chi2p1_mc=mc, lower_bound=bound, asymptotic=math.e / (2 * math.sqrt(math.pi)) * math.sqrt(d))
    out['a'].append(row); print('(a)', row, flush=True)
out['a_note'] = 'For d <= 2, E[Y e^Y] diverges: bounding-cylinder importance sampling has infinite variance.'

for d in [2, 16, 256]:
    g = torch.randn(d, dtype=torch.float64); g /= g.norm(); x0 = torch.randn(d, dtype=torch.float64) * 0.3
    F = lambda X: torch.sin(X @ g) + 0.5 * (X @ g) ** 2           # |grad F(x0)| known
    gradF = (torch.cos(x0 @ g) + (x0 @ g)) * g
    for r in [0.4, 0.1, 0.025]:
        z, s = sample_watson(d, int(min(400000, 4e7 / d)), dtype=torch.float64)
        v = F(x0 + r * z)
        ratio = float(v.var()) / (r * r * float(gradF @ gradF) * C1(d) / 2 / d * d)  # E[(g.z)^2] = |g|^2 E|z|^2/d
        out['b'].append(dict(d=d, r=r, var_ratio=ratio)); print('(b)', out['b'][-1], flush=True)

for d in [2, 16, 64, 256, 1024]:
    mu = torch.randn(d, dtype=torch.float64); mu = 1.5 * mu / mu.norm()
    def u(X, T):  # zero-drift heat flow of N(mu, 0.5 I), normalised so u(x0,t0) ~ 1
        v = 0.25 + 2 * T
        return torch.exp(-((X - mu) ** 2).sum(1) / (2 * v) - d / 2 * torch.log(v) + d / 2 * math.log(0.25 + 2 * 0.5))
    x0 = mu + 0.3 * torch.randn(d, dtype=torch.float64) / math.sqrt(d) * 3; t0 = 0.5
    u0 = float(u(x0[None], torch.tensor([t0], dtype=torch.float64)))
    for r in [0.25, 1.0]:
        N = 4096; reps = 30; hb, sg = [], []
        lag = r * r * C1(d) / 4
        for _ in range(reps):
            z, s = sample_watson(d, N, dtype=torch.float64)
            hb.append(float(u(x0 + r * z, t0 - r * r * s).mean()) / u0 - 1)
            G = torch.randn(N, d, dtype=torch.float64)
            sg.append(float(u(x0 + math.sqrt(2 * lag) * G, torch.full((N,), t0 - lag, dtype=torch.float64)).mean()) / u0 - 1)
        row = dict(d=d, r=r, N=N, heatball_rel_rmse=float(np.sqrt(np.mean(np.square(hb)))),
                   semigroup_rel_rmse=float(np.sqrt(np.mean(np.square(sg)))),
                   heatball_bias=float(np.mean(hb)), semigroup_bias=float(np.mean(sg)))
        if d > 2:
            a, lam = d / 2 + 2, d / 2
            EYeY = math.exp(math.log(a) + a * math.log(lam) - (a + 1) * math.log(lam - 1))
            chi = math.exp(logS(d)) / 4 * 8 * math.pi * d * (d + 2) / (d + 4) * EYeY
            row['cylinder_IS_rmse_lower_bound_constant_part'] = float(math.sqrt((chi - 1) / N))
        out['c'].append(row); print('(c)', row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_e2.json'), 'w'), indent=1)
