"""
hbmc.drifts -- forward drifts used in the experiments (tau coordinates, diffusion coefficient 1).

Every drift has  f(x, tau) -> (n, d)  and  div(x, tau) -> (n,).
  linear:     ou, ou_aniso, ou_rot, ou_aniso_rot   (exact references via hbmc.linear_gmm)
  nonlinear:  dw_rot  (2D)   f = -(|x|^2/4 - 1) x + omega J x / (1 + |x|^2/4)
                              a ring-shaped potential well plus a *differential* (non-Killing) rotation;
                              stationary law  p_inf ~ exp(-(|x|^2/4 - 1)^2)  (non-Gaussian);
                              div f = 2 - |x|^2 in closed form.
              dw1d   (1D)    f = 2 (x - x^3)   (the double-well diffusion dX = (X - X^3) dt + dW in tau = t/2)
  learned:    frozen PhiNet/PsiNet from the original train_forward.py checkpoint (networks.py)
"""
import math
import torch
from .linear_gmm import LinearDrift


class DWRot:
    """Nonlinear 2D drift with a differential rotation."""
    linear = False

    def __init__(self, omega=1.5):
        self.omega = omega; self.d = 2

    def __call__(self, x, tau=None):
        r2 = (x * x).sum(1, keepdim=True)
        Jx = torch.stack([-x[:, 1], x[:, 0]], 1)
        return -(r2 / 4 - 1) * x + self.omega * Jx / (1 + r2 / 4)

    def div(self, x, tau=None):
        return 2.0 - (x * x).sum(1)


class ShearWell:
    """2D well phi(y) = -|y|^2/2 + a sqrt(1+|y|^2) (minimum on the circle |y| = sqrt(a^2-1)) with the shear
    omega J grad phi: f = (I + omega J) grad phi.  The shear is divergence-free and tangent to the level sets of phi, so
    e^{phi} is stationary for every omega (T5 forward drift; omega varies the rotation at a fixed gradient part)."""
    linear = False

    def __init__(self, omega=8.0, a=2.5):
        self.omega = float(omega); self.a = float(a); self.d = 2

    def grad_phi(self, x):
        r2 = (x * x).sum(1, keepdim=True)
        return (-1 + self.a / torch.sqrt(1 + r2)) * x

    def phi(self, x):
        r2 = (x * x).sum(1)
        return -0.5 * r2 + self.a * torch.sqrt(1 + r2)

    def __call__(self, x, tau=None):
        g = self.grad_phi(x)
        return g + self.omega * torch.stack([-g[:, 1], g[:, 0]], 1)

    def div(self, x, tau=None):
        r2 = (x * x).sum(1)
        return -2.0 + self.a * (2 + r2) / (1 + r2) ** 1.5


class DW1D:
    linear = False

    def __init__(self):
        self.d = 1

    def __call__(self, x, tau=None):
        return 2.0 * (x - x ** 3)

    def div(self, x, tau=None):
        return (2.0 - 6.0 * x ** 2)[:, 0]


class LearnedDrift:
    """Frozen drift f = get_f1 + get_f2 from the original 2D networks (PhiNet, PsiNet).
    Divergence by central differences of the (frozen) drift; it does not depend on the score model."""
    linear = False

    def __init__(self, ckpt_path, networks_module, device='cpu', eps=1e-3):
        ck = torch.load(ckpt_path, map_location=device)
        cfg = ck.get('config', {})
        self.phi = networks_module.PhiNet(cfg.get('hidden', 256), cfg.get('ted_dim', 128), cfg.get('num_blocks', 4)).to(device)
        self.psi = networks_module.PsiNet(cfg.get('hidden', 256), cfg.get('ted_dim', 128), cfg.get('num_blocks', 4)).to(device)
        self.phi.load_state_dict(ck['phi_net']); self.psi.load_state_dict(ck['psi_net'])
        for p in list(self.phi.parameters()) + list(self.psi.parameters()):
            p.requires_grad_(False)
        self.d = 2; self.eps = eps

    def __call__(self, x, tau):
        dt = next(self.phi.parameters()).dtype
        tau = torch.as_tensor(tau, dtype=dt, device=x.device).reshape(-1).expand(x.shape[0])
        f = self.phi.get_f1(x.to(dt), tau) + self.psi.get_f2(x.to(dt), tau)
        return f.to(x.dtype)

    def div(self, x, tau):
        e = self.eps; out = 0.0
        for i in range(x.shape[1]):
            ei = torch.zeros_like(x); ei[:, i] = e
            out = out + (self(x + ei, tau)[:, i] - self(x - ei, tau)[:, i]) / (2 * e)
        return out


def make(name, d=2, **kw):
    if name == 'ou':
        D = LinearDrift.ou(d, a=1.0)
    elif name == 'ou_aniso':
        D = LinearDrift.ou(d, a=0.5, a2=2.0)
    elif name == 'ou_rot':
        D = LinearDrift.ou(d, a=1.0, omega=kw.get('omega', 2.0))
    elif name == 'ou_aniso_rot':
        D = LinearDrift.ou(d, a=0.5, a2=2.0, omega=kw.get('omega', 2.0))
    elif name == 'dw_rot':
        return DWRot(kw.get('omega', 1.5))
    elif name == 'dw1d':
        return DW1D()
    else:
        raise ValueError(name)
    D.linear = True
    return D
