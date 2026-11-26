"""
hbmc.sampling -- forward simulation and reverse-time generation (tau coordinates).

forward:  dx = f dtau + sqrt(2) dW
reverse:  x_{tau - h} = x - (f - 2 s) h + sqrt(2 h) eps           (Euler-Maruyama)
Steps are geometric in tau (dense near tau_end).  check_sampler.py runs this with the EXACT score
and verifies that the generated law matches the data (the pipeline check behind Referee 2, item 1).
"""
import math
import numpy as np
import torch


def tau_grid(tau_max, tau_end, n):
    return torch.tensor(np.geomspace(tau_max, tau_end, n + 1), dtype=torch.float64)


@torch.no_grad()
def reverse_sde(score_fn, drift, x_init, tau_max, tau_end=1e-3, n_steps=1000, last_noise=False):
    x = x_init.clone()
    T = tau_grid(tau_max, tau_end, n_steps).to(dtype=x.dtype).to(x.device)
    for i in range(n_steps):
        t, t_next = T[i], T[i + 1]; h = t - t_next
        tv = torch.full((x.shape[0],), float(t), dtype=x.dtype, device=x.device)
        s = score_fn(x, tv)
        x = x - (drift(x, tv) - 2 * s) * h
        if i < n_steps - 1 or last_noise:
            x = x + torch.sqrt(2 * h) * torch.randn_like(x)
    return x


@torch.no_grad()
def forward_bank(drift, x0, tau_max, dt=1e-3, save_every=10):
    """Euler-Maruyama paths from data; returns X (n_slices, n, d), taus (n_slices,)."""
    x = x0.clone(); n_steps = int(round(tau_max / dt))
    X, T = [], []
    for k in range(1, n_steps + 1):
        tv = torch.full((x.shape[0],), (k - 1) * dt, dtype=x.dtype)
        x = x + drift(x, tv) * dt + math.sqrt(2 * dt) * torch.randn_like(x)
        if k % save_every == 0:
            X.append(x.clone()); T.append(k * dt)
    return torch.stack(X), torch.tensor(T, dtype=x.dtype)
