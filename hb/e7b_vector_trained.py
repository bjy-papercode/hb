"""E7 (completion): the vector residual for networks trained by denoising score matching.

Same problem and anchors as e7_diagnostic.py (d = 2, anisotropic OU with rotation, exact score), same DSM training
budgets.  For each budget: relative score MSE, the scalar heat-ball residual (two-sample mean square, as in E7) and
the VECTOR residual s(x0) - M_r[s] - S_r[grad h_v] of s_theta = grad v_theta, in the VJP form and in the form
integrated by parts (values of s only), two-sample mean square |R|^2 in pointwise units, N = N' = 2048.
The vector residual is invariant under v -> v + c(tau); the question is whether it tracks the score error.
Output: results_e7b.json"""
import sys, os, json, math
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.heatball import C1, admissible_r
from hbmc.linear_gmm import LinearDrift, BlockGMM
from hbmc.residuals import hb_residual_v, hb_residual_s
from hbmc.models import VNet
from hbmc import losses
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get('QUICK', '0')))
torch.manual_seed(0); dt = torch.float64
R_HB = 0.3
N = 256 if QUICK else 2048
d = 2
g = torch.Generator().manual_seed(7)
mu = torch.randn(3, d, generator=g, dtype=dt); mu = 2 * mu / mu.norm(dim=1, keepdim=True)
G = BlockGMM(mu.numpy(), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.3, 0.2]))
D = LinearDrift.ou(d, a=0.5, a2=1.5, omega=1.0)
Bm = torch.block_diag(*D.B)
sstar = lambda x, t: G.logp_score(x, t, D)[1]
vstar = lambda x, t: G.logp_score(x, t, D, need_score=False)
B = 64 if QUICK else 256
tau = torch.exp(torch.empty(B, dtype=dt).uniform_(math.log(R_HB ** 2 / (4 * math.pi) * 1.05), 0.0))
x = torch.stack([G.sample_tau(1, float(t), D)[0][0] for t in tau])
lp = vstar(x, tau); low = lp <= torch.quantile(lp, 0.10)
s_ref = sstar(x, tau); nrm = (s_ref ** 2).sum(1)
sc = lambda tt: (4 / (C1(d) * admissible_r(tt, R_HB) ** 2))


def chunked(f, k=16):
    return torch.cat([f(x[i:i + k], tau[i:i + k]) for i in range(0, B, k)])


out = dict(config=dict(r=R_HB, N=N, anchors=B), rows=[])
OUTP = os.path.join(HERE, 'results_e7b.json')
if os.path.exists(OUTP) and not QUICK:                      # resume: keep finished budgets
    out = json.load(open(OUTP))
done_b = {r['dsm_steps'] for r in out['rows']}
# sanity: the exact score has zero vector residual
for form in ([] if 'exact_ibp' in out else ['vjp', 'ibp']):
    R2 = [chunked(lambda xx, tt: hb_residual_s(lambda X, T: sstar(X, T), lambda X, T: D(X), lambda X, T: D.div(X), xx, tt, R_HB,
                                               N=N, Np=N, form=form, create_graph=False, gradf_T_s=lambda X, T, s: s @ Bm,
                                               grad_div_f=lambda X, T: torch.zeros_like(X)) * sc(tt)[:, None]) for _ in range(2)]
    out[f'exact_{form}'] = float((R2[0] * R2[1]).sum(1).mean())
    print('exact score, vector residual', form, out[f'exact_{form}'], flush=True)
torch.manual_seed(1)
net = VNet(2, 128, 64, 3); opt = torch.optim.Adam(net.parameters(), 1e-3)
done = 0
for budget in ([50, 400] if QUICK else [50, 200, 800, 3200, 8000]):
    for _ in range(budget - done):
        x0 = G.sample0(512).float(); tt = torch.exp(torch.empty(512).uniform_(math.log(1e-3), 0.0))
        L = losses.dsm(net, G, D, x0, tt); opt.zero_grad(); L.backward(); opt.step()
    done = budget
    if budget in done_b:
        print('budget', budget, 'done before'); continue
    vn = lambda X, T: net(X.float(), T.float()).double()

    def sn(X, T):
        with torch.enable_grad():
            Xr = X if X.requires_grad else X.detach().requires_grad_(True)
            s, = torch.autograd.grad(net(Xr.float(), T.float()).sum(), Xr, create_graph=X.requires_grad)
        return s.double()
    err = ((sn(x, tau) - s_ref) ** 2).sum(1)
    row = dict(dsm_steps=budget, score_rel_mse=float(err.mean() / nrm.mean()), score_rel_mse_low=float(err[low].mean() / nrm[low].mean()))
    Rs = [chunked(lambda xx, tt: hb_residual_v(vn, D, D.div, xx, tt, R_HB, N=N, Np=N, create_graph=False).detach() * sc(tt)) for _ in range(2)]
    row['scalar_res2'] = float((Rs[0] * Rs[1]).mean())
    for form in ['vjp', 'ibp']:
        R2 = [chunked(lambda xx, tt: hb_residual_s(sn, lambda X, T: D(X), lambda X, T: D.div(X), xx, tt, R_HB, N=N, Np=N, form=form,
                                                   create_graph=False, gradf_T_s=lambda X, T, s: s @ Bm,
                                                   grad_div_f=lambda X, T: torch.zeros_like(X)).detach() * sc(tt)[:, None]) for _ in range(2)]
        prod = (R2[0] * R2[1]).sum(1)
        row[f'vector_res2_{form}'] = float(prod.mean()); row[f'vector_res2_{form}_low'] = float(prod[low].mean())
    out['rows'].append(row); out['rows'].sort(key=lambda r: r['dsm_steps']); print(row, flush=True)
    json.dump(out, open(os.path.join(HERE, 'results_e7b.json'), 'w'), indent=1)
sm = np.array([r['score_rel_mse'] for r in out['rows']])
for k in ['scalar_res2', 'vector_res2_vjp', 'vector_res2_ibp']:
    v = np.array([r[k] for r in out['rows']])
    out[f'spearman_{k}'] = float(np.corrcoef(np.argsort(np.argsort(sm)), np.argsort(np.argsort(v)))[0, 1])
json.dump(out, open(os.path.join(HERE, 'results_e7b.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if k.startswith('spearman') or k.startswith('exact')})
