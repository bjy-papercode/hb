"""
hbmc.hbframe -- heat-ball kernels in a moving frame (moving-frame heat-ball score matching).

Setting (tau = cumulative variance):  d_tau u = Lap u - div(f u).
The gauge form omega = e^{Phi/2} u carries a factor e^{(Phi(y) - Phi(x0))/2} whose exponent contains the first-order
term grad Phi(x0) . (y - x0); its spread grows like |grad Phi| |y - x0| ~ sqrt(d) r and forces tiny heat balls in high
dimension.  A moving frame removes it.

Frame attached to the apex (x0, tau0), for tau = tau0 - delta < tau0 (frozen at the apex, closed form except m):
    m(tau):  centre path, m(tau0) = x0: backward flow of the linearised drift f(x0) + Df(x0)(m - x0) (default,
             closed form by an augmented matrix exponential) or the exact characteristic m' = f(m) (RK4)
    a(tau) = exp(-c0 delta),     c0 = tr Df(x0) / d                     (conformal part)
    R(tau) = exp(-delta Omega0), Omega0 = (Df(x0) - Df(x0)^T) / 2      (rotation part)
    s:  ds/dtau = a^{-2},  Delta s(delta) = s(tau0) - s(tau) = (e^{2 c0 delta} - 1) / (2 c0)
With y = m + a R z and v(z, s) = a^d u(y, tau):
    d_s v = Lap_z v - div_z(h v),   h(z) = a R^T [ f(y) - m' - (c0 I + Omega0)(y - m) ]
exactly for this frame, with m' the centre path's own derivative (any differentiable centre path works; for linear
drifts both choices are the characteristic, h vanishes identically for isotropic / rotating OU and the frame is the
caloric coordinate system).
Watson's identity with a source, averaged over radii with weight rho^k (hbmc.hbsm.kernels), at the apex (z, s) = (0, s0):
    u(x0, tau0) = int int u(y, tau) Kf(x0; y, tau) dy dtau,
    Kf = a^{-2} [ Kbar(z, Delta s) + h(z) . grad Psibar(z, Delta s) ]        (v dz ds = u a^{-2} dy dtau; IBP of the source)
and, Kf being a differentiable function of x0 (the radius-averaged kernels vanish on the heat-ball boundary),
    grad u(x0, tau0) = int int u(y, tau) grad_{x0} Kf(x0; y, tau) dy dtau        (autograd through the frame).
Radius: r = frac sqrt(4 pi Delta s(tau0)) (the ball stays in tau >= 0).
"""
import math
import torch
from .hbsm import kernels

__all__ = ["Op", "Frame", "frame_kernel", "LinearJac", "BlocksDWRotJac", "BlocksShearWellJac", "sample_depth", "sample_apex", "frame_terms",
           "hbsm_frame_coloc", "radius"]


class Op:
    """n linear operators A_i = Q blockdiag(B_i1, ..., B_iK) Q^T  (B: (n, K, b, b); Q: (d, d) orthogonal or None = I).
    A dense Jacobian is the case K = 1, b = d.  Exponentials, products and the affine flow are done blockwise; this is
    only linear algebra for a drift whose Jacobian has this structure (the heat ball itself stays d-dimensional)."""

    def __init__(self, B, Q=None):
        self.B, self.Q = B, Q
        self.n, self.K, self.b = B.shape[0], B.shape[1], B.shape[2]

    @staticmethod
    def dense(A):
        return Op(A[:, None])

    def _to(self, v):
        return (v if self.Q is None else v @ self.Q.to(v.dtype)).reshape(v.shape[0], self.K, self.b)

    def _from(self, w):
        v = w.reshape(w.shape[0], self.K * self.b)
        return v if self.Q is None else v @ self.Q.to(v.dtype).T

    def mv(self, v):
        return self._from(torch.einsum('nkij,nkj->nki', self.B, self._to(v)))

    def T(self):
        return Op(self.B.transpose(-1, -2), self.Q)

    def trace(self):
        return self.B.diagonal(dim1=-2, dim2=-1).sum((-1, -2))

    def skew(self):
        return Op(0.5 * (self.B - self.B.transpose(-1, -2)), self.Q)

    def scale(self, s):
        return Op(self.B * s[:, None, None, None], self.Q)

    def expm(self, t):
        """expm(t A), t (n,); closed form for 2x2 blocks."""
        tB = t[:, None, None, None] * self.B
        return Op(_expm2(tB) if self.b == 2 else torch.linalg.matrix_exp(tB), self.Q)

    def __matmul__(self, o):
        return Op(self.B @ o.B, self.Q)

    def to_dense(self):
        D = torch.block_diag(*[torch.zeros(self.b, self.b)] * self.K).to(self.B.dtype).expand(self.n, -1, -1).clone()
        for k in range(self.K):
            D[:, k * self.b:(k + 1) * self.b, k * self.b:(k + 1) * self.b] = self.B[:, k]
        if self.Q is not None:
            Q = self.Q.to(D.dtype); D = Q @ D @ Q.T
        return D

    def affine_back(self, x0, bvec, t):
        """m = flow of m' = A m + bvec backward over time t from m = x0 (blockwise augmented exponential)."""
        n = x0.shape[0]
        Aa = torch.zeros(n, self.K, self.b + 1, self.b + 1, dtype=x0.dtype)
        Aa[..., :self.b, :self.b] = self.B; Aa[..., :self.b, self.b] = self._to(bvec)
        E = torch.linalg.matrix_exp(-t[:, None, None, None] * Aa)
        return self._from(torch.einsum('nkij,nkj->nki', E[..., :self.b, :self.b], self._to(x0)) + E[..., :self.b, self.b])


def _expm2(A):
    """exp of 2x2 matrices (..., 2, 2):  A = s I + N, N^2 = Delta I  =>  e^A = e^s [C(Delta) I + S(Delta) N],
    C = cosh sqrt(Delta), S = sinh sqrt(Delta) / sqrt(Delta) (cos / sin for Delta < 0)."""
    a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    s = 0.5 * (a + d); p = 0.5 * (a - d)
    D = p * p + b * c
    q = torch.sqrt(D.abs().clamp_min(1e-30))
    small = D.abs() < 1e-10
    C = torch.where(small, 1 + 0.5 * D, torch.where(D > 0, torch.cosh(q), torch.cos(q)))
    S = torch.where(small, 1 + D / 6, torch.where(D > 0, torch.sinh(q), torch.sin(q)) / q)
    e = torch.exp(s)
    return torch.stack([torch.stack([e * (C + S * p), e * S * b], -1), torch.stack([e * S * c, e * (C - S * p)], -1)], -2)


class LinearJac:
    """Jacobian of hbmc.linear_gmm.LinearDrift (blockdiag of 2x2 blocks)."""

    def __init__(self, drift):
        self.B = drift.B

    def __call__(self, x):
        return Op(self.B.to(x.dtype)[None].expand(x.shape[0], -1, -1, -1))


class BlocksDWRotJac:
    """Jacobian of K independent dw_rot blocks in y = Q^T x coordinates:  Df = Q blockdiag(J_k(y_k)) Q^T,
    J(y) = -[(|y|^2/4 - 1) I + y y^T / 2] + omega [J / (1+|y|^2/4) - (J y) y^T / (2 (1+|y|^2/4)^2)]."""

    def __init__(self, Q, omega=1.5):
        self.Q = Q; self.omega = omega

    def __call__(self, x):
        n, d = x.shape; Q = self.Q.to(x.dtype)
        y = (x @ Q).view(n, d // 2, 2)
        r2 = (y * y).sum(-1, keepdim=True)[..., None]                       # (n, K, 1, 1)
        I2 = torch.eye(2, dtype=x.dtype)
        Jm = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=x.dtype)
        yy = y[..., :, None] * y[..., None, :]
        Jy = torch.einsum('ij,nkj->nki', Jm, y)
        Jb = -((r2 / 4 - 1) * I2 + yy / 2) + self.omega * (Jm / (1 + r2 / 4) - Jy[..., :, None] * y[..., None, :] / (2 * (1 + r2 / 4) ** 2))
        return Op(Jb, Q)


class BlocksShearWellJac:
    """Jacobian of K blocks of hbmc.drifts.ShearWell in y = Q^T x coordinates: Df = Q blockdiag((I + omega J) H_k) Q^T,
    H = -I + a [I / sqrt(1+|y|^2) - y y^T / (1+|y|^2)^{3/2}] (Hessian of phi)."""

    def __init__(self, Q, omega=8.0, a=2.5):
        self.Q = Q; self.omega = float(omega); self.a = float(a)

    def __call__(self, x):
        n, d = x.shape; Q = self.Q.to(x.dtype)
        y = (x @ Q).view(n, d // 2, 2)
        r2 = (y * y).sum(-1, keepdim=True)[..., None]
        I2 = torch.eye(2, dtype=x.dtype)
        H = -I2 + self.a * (I2 / torch.sqrt(1 + r2) - y[..., :, None] * y[..., None, :] / (1 + r2) ** 1.5)
        Jm = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=x.dtype)
        return Op((I2 + self.omega * Jm) @ H, Q)


def _as_op(J):
    return J if isinstance(J, Op) else Op.dense(J)


class Frame:
    """Frame quantities at per-sample depths delta for apices x0 (n, d), tau0 (n,); differentiable in x0.
    frozen=True: the parameters (Df, f, c0, Omega0 at x0) are computed without gradient."""

    def __init__(self, x0, tau0, delta, f, jac, M=8, centre='affine', frozen=False):
        n, d = x0.shape
        xp = x0.detach() if frozen else x0          # point at which the frame parameters (Df, f, c0, Omega) are taken
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            J0 = _as_op(jac(xp))
            self.c0 = J0.trace() / d
            self.Om = J0.skew()
            self.a = torch.exp(-self.c0 * delta)
            self.R = self.Om.expm(-delta)
            c = self.c0
            small = c.abs() < 1e-8
            cs = torch.where(small, torch.ones_like(c), c)
            self.ds = torch.where(small, delta, torch.expm1(2 * cs * delta) / (2 * cs))
            self.ds_total = torch.where(small, tau0, torch.expm1(2 * cs * tau0) / (2 * cs))
            if centre == 'affine':
                # centre path = backward flow of the drift linearised at the apex, f(x0) + J0 (m - x0): defined for every
                # depth (the exact backward characteristic of a superlinear drift blows up in finite time); the frame
                # identity holds for any differentiable centre path when h uses the path's own derivative m'.
                b = f(xp) - J0.mv(xp)
                if frozen:   # m is affine in x0 with frozen coefficients: m = E x0 + e
                    e = J0.affine_back(torch.zeros_like(xp), b, delta)
                    Ex = J0.expm(-delta)
        if centre == 'affine':
            if frozen:
                self.m = Ex.mv(x0) + e
            else:
                self.m = J0.affine_back(x0, b, delta)
            self.fm = J0.mv(self.m) + b                                      # m' = f(x0) + J0 (m - x0)
        else:                                                                # exact characteristic m' = f(m) (RK4)
            m = x0; h = (delta / M)[:, None]
            for _ in range(M):
                k1 = f(m); k2 = f(m - 0.5 * h * k1); k3 = f(m - 0.5 * h * k2); k4 = f(m - h * k3)
                m = m - h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.m = m
            self.fm = f(m)
        self.f = f; self.d = d

    def coords(self, y):
        """z = R^T (y - m) / a and the residual drift h(z) (n, d)."""
        dy = y - self.m
        RT = self.R.T()
        z = RT.mv(dy) / self.a[:, None]
        lin = self.c0[:, None] * dy + self.Om.mv(dy)
        h = self.a[:, None] * RT.mv(self.f(y) - self.fm - lin)
        return z, h


def frame_kernel(x0, tau0, y, delta, f, jac, frac, k=None, M=8, grad=True, frame_out=None, centre='affine', frozen=True,
                 c_ref=None):
    """Kf(x0; y, tau0 - delta) and (if grad) grad_{x0} Kf, for per-sample apices x0, points y (n, d).
    frozen=True: the frame parameters P = (Df(x0), f(x0), c0, Omega0, r) are held fixed in the x0-derivative and set to
    their values at x0 afterwards.  The identity u(x0) = int int u Kf(x0; P) holds for every fixed P (any constant
    rates c, Omega and any affine centre path define an exact frame), so grad u(x0) = int int u d_{x0} Kf(x0; P)|_{P=P(x0)}
    is exact too; differentiating P(x0) as well only adds terms with zero mean that the antithetic pairing does not
    cancel (they are even in z), i.e. pure variance."""
    d = x0.shape[1]
    k = d + 1.0 if k is None else k
    with torch.enable_grad():
        xr = x0.detach().requires_grad_(grad)
        F = Frame(xr, tau0, delta, f, jac, M, centre, frozen=frozen and centre == 'affine')
        r = frac * torch.sqrt(4 * math.pi * F.ds_total) * 0.999
        if c_ref is not None:
            # radius as a function of tau0 only (the same for both members of an antithetic pair), capped by the
            # largest ball that stays in tau >= 0 at this apex
            r = radius(tau0, F.c0, frac, c_ref)
        if frozen:
            r = r.detach()
        z, h = F.coords(y)
        if not grad:
            z = z.detach().requires_grad_(True); h = h.detach()
        # kernels(xi, ...) is written for xi = apex - point = -z; both kernels are even and grad Psibar is odd,
        # so int Psibar(-z) (-div(h v)) dz = int v h . (grad Psibar)(z) dz
        Kb, Pb = kernels(z, F.ds, r, d, k)
        gP, = torch.autograd.grad(Pb.sum(), z, create_graph=grad)
        Kf = (Kb + (h * gP).sum(1)) / F.a ** 2
        if frame_out is not None:
            frame_out.update(frame=F, z=z.detach(), h=h.detach(), r=r.detach())
        if not grad:
            return Kf.detach(), None
        gK, = torch.autograd.grad(Kf.sum(), xr)
    return Kf.detach(), gK


# ----------------------------------------------------------------------------------------------------------------------
# Proposal for training.  The apex must be generated from a sample y ~ p_{tau0 - delta} of the forward process:
#   (component, delta)                     chosen before y (y is drawn at tau0 - delta)
#   y ~ p_{tau0 - delta}
#   xhat:  m(xhat) = y for the affine centre path (Newton from the forward characteristic)
#   frame P frozen at xhat:  c, Omega from Df(xhat),  a = e^{-c delta},  R = e^{-delta Omega}
#   x0 = xhat -/+ M z,   M = expm(delta Df(xhat)) a R,   det M = 1   (linearised inverse of y = m(x0) + a R z)
# Three components (weights w):
#   W  Watson:   Delta s from the Watson marginal at the reference radius, z ~ |z|^2-radial on the heat-ball section
#                (the shape of Kbar: bounded weights for the Watson term)
#   G  source:   Delta s uniform on (0, r^2/(4 pi)), z ~ N(0, 2 Delta s I) = Gamma itself (the source kernel Psibar
#                contains Gamma, which exceeds r^{-d} by up to (r/a)^d deep inside the ball: exponentially in d; a
#                Watson-shaped proposal then gives exploding weights for the source term h . grad Psibar)
#   U  coverage: delta uniform on (0, tau0), z as in W (mixed with a small Gaussian part)
# The density of (delta, z) is the full mixture, so the estimator
#   int s(x0) . grad u(x0) dx0 = E[ s(x0) . grad_{x0} Kf(x0; y, tau0 - delta) / q(delta, z) ]
# is unbiased for any drift; the frame and the mixture only decide the variance.  For linear drifts M z is exactly
# the frame coordinate of y.  Radius: a function of tau0 only (reference rate c_ref), capped at the largest ball that
# stays in tau >= 0 at the apex, so that the two members of an antithetic pair share it (otherwise the odd part of
# grad Kbar no longer cancels in the pair).
# ----------------------------------------------------------------------------------------------------------------------
def _ds_of(delta, c):
    cs = torch.where(c.abs() < 1e-8, torch.ones_like(c), c)
    return torch.where(c.abs() < 1e-8, delta, torch.expm1(2 * cs * delta) / (2 * cs))


def _delta_of(ds, c):
    cs = torch.where(c.abs() < 1e-8, torch.ones_like(c), c)
    return torch.where(c.abs() < 1e-8, ds, torch.log1p(2 * cs * ds) / (2 * cs))


MIX = (0.8, 0.15, 0.05)        # weights of the components W, G, U


def sample_depth(n, tau0, frac, d, c_ref, w=MIX, dtype=torch.float64):
    """delta (n,), component (n,) in {0: W, 1: G, 2: U}, depth densities (n, 3) of the three components at delta."""
    c = torch.full_like(tau0, float(c_ref))
    r2 = frac ** 2 * 4 * math.pi * _ds_of(tau0, c)
    gam = torch.distributions.Gamma(torch.tensor(d / 2 + 2.0, dtype=dtype), torch.tensor(d / 2.0, dtype=dtype))
    # stratified: fixed counts n w_c per batch (multiple importance sampling with the balance heuristic); the batch mean
    # is unbiased and carries no variance from the random choice of component, which in high dimension (mean ~ d,
    # carried by the Watson component) would dominate
    cnt = [int(round(n * wc)) for wc in w]; cnt[0] = n - sum(cnt[1:])
    comp = torch.cat([torch.full((c_,), i, dtype=torch.long) for i, c_ in enumerate(cnt)])[torch.randperm(n)]
    dsw = r2 * torch.exp(-gam.sample((n,))) / (4 * math.pi)
    dsg = r2 / (4 * math.pi) * torch.rand(n, dtype=dtype)
    dl = torch.where(comp == 0, _delta_of(dsw, c), torch.where(comp == 1, _delta_of(dsg, c), tau0 * torch.rand(n, dtype=dtype)))
    dl = dl.clamp(1e-10 * tau0, tau0 * (1 - 1e-10))
    ds = _ds_of(dl, c); sg = 4 * math.pi * ds / r2
    inside = sg < 1
    jac_ = torch.exp(2 * c * dl)                                             # d Delta s / d delta
    pW = torch.where(inside, torch.exp(gam.log_prob(-torch.log(sg.clamp(1e-300, 1 - 1e-15)))) / ds * jac_, torch.zeros_like(ds))
    pG = torch.where(inside, 4 * math.pi / r2 * jac_, torch.zeros_like(ds))
    pU = 1.0 / tau0
    return dl, comp, torch.stack([pW, pG, pU.expand_as(pW)], 1)


def _log_sphere(d):
    return math.log(2.0) + 0.5 * d * math.log(math.pi) - math.lgamma(d / 2)


def _affine_centre(x0, delta, f, jac):
    """m(tau0 - delta) of the affine centre path from x0, and J0 = Df(x0) (Op)."""
    J0 = _as_op(jac(x0))
    return J0.affine_back(x0, f(x0) - J0.mv(x0), delta), J0


def radius(tau0, c_apex, frac, c_ref):
    """r (n,): frac sqrt(4 pi Delta s(tau0; c_ref)), capped at sqrt(4 pi Delta s(tau0; c_apex))."""
    return torch.minimum(frac * torch.sqrt(4 * math.pi * _ds_of(tau0, torch.full_like(c_apex, float(c_ref)))),
                         torch.sqrt(4 * math.pi * _ds_of(tau0, c_apex))) * 0.999


def sample_apex(y, tau0, delta, comp, pdep, f, jac, frac, c_ref, w=MIX, eps_g=0.05, M=4, newton=2):
    """x0 = xhat - M z and the antithetic xhat + M z; q(delta, z) (n,) of the full mixture."""
    n, d = y.shape; dt = y.dtype
    with torch.no_grad():
        x = y; h = (delta / M)[:, None]
        for _ in range(M):                                                   # forward characteristic (initial guess)
            k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        for _ in range(newton):                                              # m_aff(xhat) = y
            m, J = _affine_centre(x, delta, f, jac)
            x = x + J.expm(delta).mv(y - m)
        xhat = x
        J = _as_op(jac(xhat)); c = J.trace() / d
        Mz = (J.expm(delta) @ J.skew().expm(-delta)).scale(torch.exp(-c * delta))
        ds = _ds_of(delta, c); r = radius(tau0, c, frac, c_ref)
        L = torch.log(r * r / (4 * math.pi * ds))
        has = L > 0
        Rmax = torch.where(has, torch.sqrt((2 * d * ds * L).clamp_min(1e-300)), torch.ones_like(L))   # section radius
        u = torch.randn(n, d, dtype=dt); u = u / u.norm(dim=1, keepdim=True)
        zb = (Rmax * torch.rand(n, dtype=dt) ** (1.0 / (d + 2)))[:, None] * u
        zg = torch.sqrt(2 * ds)[:, None] * torch.randn(n, d, dtype=dt)
        gauss = (comp == 1) | ((torch.rand(n, dtype=dt) < eps_g) & (comp != 1)) | ~has
        z = torch.where(gauss[:, None], zg, zb)
        z2 = (z * z).sum(1)
        qg = torch.exp(-z2 / (4 * ds) - 0.5 * d * torch.log(4 * math.pi * ds))
        lq_b = math.log(d + 2) + torch.log(z2.clamp_min(1e-300)) - _log_sphere(d) - (d + 2) * torch.log(Rmax)
        qb = torch.where(has & (z2 <= Rmax * Rmax), torch.exp(lq_b), torch.zeros_like(lq_b))
        qmix = torch.where(has, (1 - eps_g) * qb + eps_g * qg, qg)          # z | delta for components W and U
        q = w[0] * pdep[:, 0] * qmix + w[1] * pdep[:, 1] * qg + w[2] * pdep[:, 2] * qmix
        mz = Mz.mv(z)
    return xhat - mz, xhat + mz, q


def frame_terms(y, tau0, delta, comp, pdep, f, jac, frac, c_ref, k=None, M=8, w=MIX, eps_g=0.05):
    """Antithetic apices x0+-, weights W = 1 / q(delta, z), and W Kf, W grad Kf at both (detached).
    Returns [(x0, W Kf, W grad Kf)] for the two members of the pair."""
    xp, xm, q = sample_apex(y, tau0, delta, comp, pdep, f, jac, frac, c_ref, w, eps_g, M)
    W = 1.0 / q
    out = []
    for x0 in (xp, xm):
        Kf, gK = frame_kernel(x0, tau0, y, delta, f, jac, frac, k=k, M=M, c_ref=c_ref)
        out.append((x0, W * Kf, W[:, None] * gK))
    return out


def hbsm_frame_coloc(s_fn, terms, tau0, clip=False, x_p=None, mix=0.0):
    """Co-located moving-frame HBSM per sample: 1/2 sum_+- [ (W Kf) |s(x0)|^2 - 2 s(x0) . W grad Kf ].
    clip=True replaces W Kf by its positive part (bounded below, but biased: the source term h . grad Psibar makes
    3-5% of the weight negative in d = 64, and the minimiser is then shrunk by 3-5% at large tau0).
    mix > 0: the quadratic term is mix * |s(x_p)|^2 with x_p ~ p_tau0 plus (1 - mix) times the co-located one, unbiased
    for every mix in [0, 1]; the exact samples keep the loss anchored where the co-located weights are negative."""
    L = 0.0
    for x0, WK, WG in terms:
        s = s_fn(x0, tau0)
        w = WK.clamp_min(0) if clip else WK
        L = L + 0.5 * ((1 - mix) * w * (s * s).sum(1) - 2 * (s * WG).sum(1))
    if mix > 0:
        sp = s_fn(x_p, tau0)
        L = L + mix * (sp * sp).sum(1)
    return L
