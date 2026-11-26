"""
hbmc.metrics -- sample-quality and score-accuracy metrics (referee requests R2.1-R2.3, R2 minor 2).

  mmd2_unbiased     unbiased MMD^2, sum of Gaussian kernels at {0.5,1,2} x median distance of the reference
  sliced_w2         sliced 2-Wasserstein distance (random projections)
  mode_metrics      for Gaussian-mixture data with <= 20 components: modes recovered (>= 20% of the
                    expected count), total-variation error of the mode weights, mean ratio of generated to
                    true per-mode spread (sqrt of covariance trace)
  low_density_mass  fraction of samples whose true log p0 lies below the q-quantile of log p0 over data
                    samples (q = 5%: data have 0.05 by construction); > q means mass leaks into the tails
                    (over-dispersion), < q means over-concentration
  score_errors      relative score MSE under p_tau, overall and on the lowest-10%-density subset
"""
import math
import numpy as np
import torch


def _pdist2(a, b):
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T


def mmd2_unbiased(x, y, n_max=3000):
    x = torch.as_tensor(x, dtype=torch.float64)[:n_max]; y = torch.as_tensor(y, dtype=torch.float64)[:n_max]
    dyy = _pdist2(y, y).clamp_min(0)
    med = torch.median(dyy[~torch.eye(len(y), dtype=bool)]).sqrt().item()
    dxx = _pdist2(x, x).clamp_min(0); dxy = _pdist2(x, y).clamp_min(0)
    n, m = len(x), len(y); tot = 0.0
    for s in [0.5 * med, med, 2 * med]:
        g = 1.0 / (2 * s * s)
        kxx = torch.exp(-g * dxx); kyy = torch.exp(-g * dyy); kxy = torch.exp(-g * dxy)
        tot += ((kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                - 2 * kxy.mean()).item()
    return tot / 3


def sliced_w2(x, y, n_proj=500, seed=0):
    x = torch.as_tensor(x, dtype=torch.float64); y = torch.as_tensor(y, dtype=torch.float64)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(x.shape[1], n_proj, generator=g, dtype=torch.float64); P /= P.norm(dim=0, keepdim=True)
    px, _ = torch.sort(x @ P, 0); py, _ = torch.sort(y @ P, 0)
    return float(((px - py) ** 2).mean().sqrt())


def mode_metrics(x, gmm, min_frac=0.2):
    if gmm.K > 20:
        return {}
    x = torch.as_tensor(x, dtype=torch.float64)
    R = gmm.responsibilities0(x); k = R.argmax(1)
    counts = torch.bincount(k, minlength=gmm.K).double(); frac = counts / len(x)
    recovered = int((frac >= min_frac * gmm.w).sum())
    spreads = []
    for j in range(gmm.K):
        xs = x[k == j]
        if len(xs) > 5:
            gen = torch.cov(xs.T).trace().sqrt(); tru = gmm.S0[j].diagonal(dim1=-2, dim2=-1).sum().sqrt()
            spreads.append(float(gen / tru))
    return dict(modes_recovered=recovered, modes_total=gmm.K, mode_weight_tv=float(0.5 * (frac - gmm.w).abs().sum()),
                spread_ratio_mean=float(np.mean(spreads)) if spreads else float('nan'))


def low_density_mass(x, data, logp0_fn, q=0.05):
    thr = torch.quantile(logp0_fn(torch.as_tensor(data, dtype=torch.float64)), q)
    lx = logp0_fn(torch.as_tensor(x, dtype=torch.float64))
    return dict(low_density_threshold_q=q, low_density_mass=float((lx < thr).double().mean()))


def sample_metrics(x, data, gmm=None, logp0_fn=None):
    out = dict(mmd2=mmd2_unbiased(x, data), sliced_w2=sliced_w2(x, data))
    if gmm is not None:
        out.update(mode_metrics(x, gmm))
    if logp0_fn is not None:
        out.update(low_density_mass(x, data, logp0_fn))
    return out


def score_errors(s_model, s_ref, logp_ref, low_q=0.10):
    s_model = torch.as_tensor(s_model, dtype=torch.float64); s_ref = torch.as_tensor(s_ref, dtype=torch.float64)
    e = ((s_model - s_ref) ** 2).sum(1); nrm = (s_ref ** 2).sum(1)
    lp = torch.as_tensor(logp_ref, dtype=torch.float64)
    low = lp <= torch.quantile(lp, low_q)
    return dict(rel_mse=float(e.mean() / nrm.mean()), rel_mse_low=float(e[low].mean() / nrm[low].mean()),
                mse=float(e.mean()), mse_low=float(e[low].mean()))


def logp_errors(v_model, logp_ref, low_q=0.10):
    """Log-density errors of a scalar model v ~ log p at one time.  A score-matching model determines v only up
    to a time-dependent constant c(tau); `offset` = mean(v - log p) estimates it, `rmse_c` is the error after
    removing it (shape of log p), and the spread of `offset` over tau measures whether the model's log-density is
    consistent across times (it is for the true log-density, and for any model satisfying the Fokker-Planck
    equation, which conserves mass)."""
    v = torch.as_tensor(v_model, dtype=torch.float64).reshape(-1); lp = torch.as_tensor(logp_ref, dtype=torch.float64).reshape(-1)
    dlt = v - lp; off = dlt.mean(); low = lp <= torch.quantile(lp, low_q)
    return dict(offset=float(off), rmse_c=float(((dlt - off) ** 2).mean().sqrt()),
                rmse_c_low=float(((dlt[low] - off) ** 2).mean().sqrt()), sd_logp=float(lp.std()))


def logp_consistency(rows):
    """rows: {tau: logp_errors(...)}.  Returns the spread of the offsets over tau and the RMSE of v - log p after
    removing ONE constant for all times (what matters for likelihoods compared across times)."""
    offs = np.array([r['offset'] for r in rows.values()]); rc = np.array([r['rmse_c'] for r in rows.values()])
    return dict(offset_spread=float(offs.std()), rmse_one_const=float(np.sqrt(np.mean((offs - offs.mean()) ** 2 + rc ** 2))))
