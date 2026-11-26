"""W2: heat-ball walk for a NONLINEAR drift (double well + rotation, hbmc.drifts.DWRot, omega = 1.5), initial law
asymmetric_gmm, against a 400^2 finite-volume Fokker--Planck reference; and the walk with EMPIRICAL data at the
terminal (k-nearest-neighbour kernel sum, cost per path independent of the data size n) against the global
alternative: forward Euler--Maruyama simulation of the n data points to tau0 plus a Gaussian KDE.
Metrics at anchors (typical and low-density): relative error of p, error of log p, relative score error.
Output: results_w2.json"""
import sys, os, json, math, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.walk import gauge_dwrot, walk_score, terminal_gmm_short, terminal_knn_short
from hbmc.fp_grid import FPGrid, GridReference
from hbmc import datasets2d, drifts
from hbmc.linear_gmm import LinearDrift
torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
DT = torch.float64
G0 = datasets2d.get('asymmetric_gmm'); Dw = drifts.make('dw_rot'); gg = gauge_dwrot(1.5)
x = torch.randn(7, 2, dtype=DT); assert float((gg.drift(x, None) - Dw(x)).abs().max()) < 1e-12
taus = [0.5] if QUICK else [0.2, 0.5]
P = 20000 if QUICK else 50000
ns = [1000, 10000] if QUICK else [1000, 10000, 100000]
# ---- reference ----
t = time.time()
grid = FPGrid(2, 6.0, 200 if QUICK else 400, lambda x, tt: Dw(torch.as_tensor(x)).numpy())
lp0 = lambda X: G0.logp_score(torch.as_tensor(X), torch.zeros(len(X), dtype=DT), LinearDrift.ou(2, a=1.0), need_score=False).numpy()
sols = grid.solve(np.exp(lp0(grid.points())), taus, dt=1e-3)
REF = GridReference(grid, sols); t_ref = time.time() - t
rng = np.random.default_rng(0)
out = dict(config=dict(drift='dw_rot omega=1.5', data='asymmetric_gmm', P=P, c=0.5, eps='0.01 (mixture), 1e-3 (knn, k=1024; n>1e4: fresh subset of 1e4 per batch)', terminal='short-time Gaussian', schemes='gauge (main), direct (mixture terminal only)',
                       grid=grid.N, ref_seconds=t_ref), rows=[], kde=[])
data_all = G0.sample0(max(ns)).numpy()
for t0 in taus:
    tk = min(sols, key=lambda s: abs(s - t0))
    cand = grid.sample(sols[tk], 3000, rng); lpc = REF.logp(cand, t0); o = np.argsort(lpc)
    anchors = [('typical', cand[i]) for i in range(2 if QUICK else 4)] + [('low', cand[o[j]]) for j in ([150] if QUICK else [60, 200])]
    for k, (lab, a) in enumerate(anchors):
        x0 = torch.as_tensor(a, dtype=DT)
        lr = float(REF.logp(a[None], t0)[0]); sr = REF.score(a[None], t0)[0]
        for sch, tname, term in [('gauge', 'mixture', terminal_gmm_short(G0, gg.drift)),
                                 ('direct', 'mixture', terminal_gmm_short(G0, gg.drift))] + \
                                [('gauge', f'knn_n{n}', terminal_knn_short(data_all[:n], gg.drift, k=1024,
                                                                           m_sub=10000 if n > 10000 else None, seed=k)) for n in ns]:
            tt = time.time()
            try:
                r = walk_score(x0, t0, gg, term, P=P, batches=10, seed=k, scheme=sch, c=0.5,
                               eps=1e-3 if tname.startswith('knn') else 0.01)
                ok = math.isfinite(r['u']) and np.isfinite(r['score'].numpy()).all()
                msg = '' if ok else 'non-finite estimate'
            except FloatingPointError as e:
                ok, msg = False, str(e)
            if not ok:
                row = dict(tau0=t0, anchor=lab, scheme=sch, terminal=tname, diverged=True, message=msg, sec=time.time() - tt)
                out['rows'].append(row); print(f'tau0={t0} {lab} {sch} {tname}: DIVERGED ({msg})', flush=True)
                continue
            sc = r['score'].numpy()
            row = dict(tau0=t0, anchor=lab, scheme=sch, terminal=tname, diverged=False, logp_ref=lr, u_rel_err=r['u'] / math.exp(lr) - 1,
                       u_rel_se=r['u_se'] / max(r['u'], 1e-300), logp_err=math.log(max(r['u'], 1e-300)) - lr,
                       score_ref=sr.tolist(), score=sc.tolist(), score_rel_err=float(np.linalg.norm(sc - sr) / np.linalg.norm(sr)),
                       steps=r['stats']['mean_steps'], sec=time.time() - tt,
                       knn_truncation_frac=float(np.mean(term.trunc)) if getattr(term, 'trunc', None) else 0.0)
            out['rows'].append(row)
            print(f"tau0={t0} {lab} {sch} {tname}: u relerr {row['u_rel_err']:+.3f} (se {row['u_rel_se']:.3f}) logp err {row['logp_err']:+.3f} "
                  f"score relerr {row['score_rel_err']:.3f}  {row['sec']:.0f}s", flush=True)
            json.dump(out, open(os.path.join(HERE, 'results_w2.json'), 'w'), indent=1)
    # ---- baseline: forward simulation of n data points + KDE ----
    A = torch.as_tensor(np.stack([a for _, a in anchors]), dtype=DT)
    for n in ns:
        tt = time.time(); X = torch.as_tensor(data_all[:n], dtype=DT); dt = 1e-3
        for i in range(int(round(t0 / dt))):
            X = X + Dw(X) * dt + math.sqrt(2 * dt) * torch.randn_like(X)
        t_sim = time.time() - tt; tt = time.time()
        h = float(X.std(0).mean()) * n ** (-1 / 6)                  # Scott's rule, d = 2
        diff = A[:, None, :] - X[None]; K = torch.exp(-(diff ** 2).sum(-1) / (2 * h * h))
        dens = K.mean(1) / (2 * math.pi * h * h); sc = -(K[..., None] * diff).sum(1) / (h * h) / K.sum(1, keepdim=True)
        t_eval = time.time() - tt
        for j, (lab, a) in enumerate(anchors):
            lr = float(REF.logp(a[None], t0)[0]); sr = REF.score(a[None], t0)[0]
            out['kde'].append(dict(tau0=t0, anchor=lab, n=n, bandwidth=h, u_rel_err=float(dens[j]) / math.exp(lr) - 1,
                                   logp_err=math.log(max(float(dens[j]), 1e-300)) - lr,
                                   score_rel_err=float(np.linalg.norm(sc[j].numpy() - sr) / np.linalg.norm(sr)),
                                   sim_seconds=t_sim, eval_seconds_per_anchor=t_eval / len(anchors)))
        print(f'KDE n={n}: sim {t_sim:.1f}s', flush=True)
json.dump(out, open(os.path.join(HERE, 'results_w2.json'), 'w'), indent=1)
