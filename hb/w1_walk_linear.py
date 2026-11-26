"""W1: heat-ball walk for the density under a LINEAR drift with rotation (exact reference).

Drift f = -diag(0.5, 2) x + 2 J x (anisotropic OU + rotation), data asymmetric_gmm, tau0 = 0.5; u, grad u and the
score are in closed form.  Terminal: exact u at tau <= 1e-3 (isolates the walk error).
 (a) frozen moving heat-ball walk (positive weights), 'direct' scheme, radius factor c in {1, 0.5, 0.25}:
     bias should decrease like c^2, error ~ Monte Carlo standard error;
 (b) 'gauge' scheme (curl-free part as potential V) at c = 0.5;
 (c) the exact signed walk (unbiased branching estimator of hbmc.walk.walk_exact): variance blow-up.
Output: results_w1.json"""
import sys, os, json, math, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.walk import gauge_linear, walk_score, walk_exact
from hbmc.linear_gmm import LinearDrift
from hbmc import datasets2d
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
DT = torch.float64
G0 = datasets2d.get('asymmetric_gmm')
a1, a2, om, t0 = 0.5, 2.0, 2.0, 0.5
D = LinearDrift.ou(2, a=a1, a2=a2, omega=om); gg = gauge_linear(a1, a2, om)
term = lambda X, T: torch.exp(G0.logp_score(X, T, D, need_score=False))
torch.manual_seed(3)
xs, _ = G0.sample_tau(2000, t0, D); lp = G0.logp_score(xs, torch.full((2000,), t0, dtype=DT), D, need_score=False)
order = lp.argsort()
anchors = [('typical', xs[i]) for i in range(8 if not QUICK else 2)] + \
          [('low', xs[order[j]]) for j in ([20, 60, 120, 180] if not QUICK else [60])]
P = 20000 if QUICK else 200000
out = dict(config=dict(drift='f=-diag(0.5,2)x+2Jx', data='asymmetric_gmm', tau0=t0, P=P, terminal='exact u at tau<=1e-3'),
           rows=[], exact_walk=[])
for c, sch, na in [(1.0, 'direct', None), (0.5, 'direct', None), (0.25, 'direct', 4), (0.5, 'gauge', None)]:
    for k, (lab, x0) in enumerate(anchors if na is None else anchors[:2] + anchors[-2:]):
        lpx, sx = G0.logp_score(x0[None], torch.tensor([t0], dtype=DT), D)
        t = time.time()
        r = walk_score(x0, t0, gg, term, P=P, batches=10, seed=k, scheme=sch, c=c, eps=1e-3)
        ue = float(lpx.exp()); se = sx[0]
        row = dict(c=c, scheme=sch, anchor=lab, u_exact=ue, u=r['u'], u_rel_err=r['u'] / ue - 1, u_z=(r['u'] - ue) / r['u_se'],
                   score_exact=se.tolist(), score=r['score'].tolist(), score_rel_err=float((r['score'] - se).norm() / se.norm()),
                   score_z=((r['score'] - se) / r['score_se']).tolist(), steps=r['stats']['mean_steps'], sec=time.time() - t)
        out['rows'].append(row)
        print(f"c={c} {sch} {lab}: u relerr {row['u_rel_err']:+.3f} (z {row['u_z']:+.1f})  score relerr {row['score_rel_err']:.3f}  "
              f"steps {row['steps']:.0f}  {row['sec']:.0f}s", flush=True)
        json.dump(out, open(os.path.join(HERE, 'results_w1.json'), 'w'), indent=1)
for lab, x0 in anchors[:2]:
    ue = float(G0.logp_score(x0[None], torch.tensor([t0], dtype=DT), D, need_score=False).exp())
    for c in [1.0, 0.6]:
        est, st = walk_exact(x0, t0, gg.direct(), term, P=20000, c=c, eps=1e-3, pV=0.0, pG=0.2, seed=0, stats=True)
        row = dict(c=c, anchor=lab, u_exact=ue, u=float(est.mean()), rel_se=float(est.std() / math.sqrt(len(est)) / abs(est.mean() + 1e-300)),
                   weight_sd=st['w_sd'], frac_negative=st['frac_negative'], steps=st['mean_steps'])
        out['exact_walk'].append(row); print('exact signed walk', row, flush=True)
json.dump(out, open(os.path.join(HERE, 'results_w1.json'), 'w'), indent=1)
