"""
hbmc.datasets2d -- the 2D benchmark distributions as *exact* Gaussian mixtures.

Every dataset here is sampled from the mixture whose density is returned, so the reference
density and score are exact at tau = 0 and, under any linear drift, at every tau
(hbmc.linear_gmm).  This replaces the following in the original datasets.py:
  * two_spiral / three_spiral / two_moon / pinwheel: log_p0 was a KDE built on a *fresh noisy*
    sample at every call (random, and with doubled variance).  Here the mixture centres are the
    noiseless curve points and the component std equals the sampling noise, which is exactly the
    sampling distribution.
  * ring / double_ring: log_p0 omitted the 1/(2 pi r) Jacobian (score off by -x/|x|^2).  Here the
    ring is a mixture of 360 isotropic Gaussians on the circle, sampled from that mixture.
  * pinwheel: blades are exact anisotropic Gaussian components.
Parameters (radii, stds, weights) follow the original datasets.py.
"""
import math
import numpy as np
import torch
from .linear_gmm import BlockGMM

_REG = {}


def _iso(name, means, stds, wts):
    _REG[name] = lambda: BlockGMM(np.asarray(means, float), np.asarray(stds, float), np.asarray(wts, float))


_iso("symmetric_gmm", [[2.5, 0], [0, 2.5], [-2.5, 0], [0, -2.5]], [0.45] * 4, [0.25] * 4)
_iso("asymmetric_gmm", [[2.0, 0.5], [-1.5, 2.5], [-2.0, -1.0], [1.0, -2.8]], [0.20, 0.60, 0.30, 0.15],
     [0.50, 0.20, 0.20, 0.10])
_iso("sparse_dense_gmm", [[0, 0], [3.0, 1.5], [-2.5, -2.0]], [1.80, 0.12, 0.12], [0.40, 0.30, 0.30])
_iso("rare_mode", [[0, 0], [4.5, 3.5], [-4.0, 3.0]], [0.50, 0.20, 0.20], [0.97, 0.015, 0.015])


def _diag_chain():
    pos = np.array([0.0, 0.7, 1.6, 2.7, 4.0, 5.5, 7.2]); a = math.radians(40)
    m = np.stack([pos * math.cos(a) - 2.5, pos * math.sin(a) - 2.0], 1)
    return BlockGMM(m, np.full(7, 0.25), np.full(7, 1 / 7))


def _grid():
    xs = np.linspace(-3, 3, 4); xx, yy = np.meshgrid(xs, xs)
    return BlockGMM(np.stack([xx.ravel(), yy.ravel()], 1), np.full(16, 0.30), np.full(16, 1 / 16))


def _concentric():
    m, w = [], []
    for R, K, wt in [(1.0, 6, 0.2), (2.0, 8, 0.4), (3.0, 10, 0.4)]:
        for i in range(K):
            t = 2 * math.pi * i / K; m.append([R * math.cos(t), R * math.sin(t)]); w.append(wt / K)
    return BlockGMM(np.array(m), np.full(len(m), 0.20), np.array(w))


def _elongated():
    comps = [((2.0, 1.0), 30, 1.6, 0.20, 0.40), ((-2.5, 0.5), 120, 1.4, 0.18, 0.35),
             ((0.0, -2.5), 80, 1.5, 0.19, 0.25)]
    m, C, w = [], [], []
    for mu, ang, sl, ss, wt in comps:
        a = math.radians(ang); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        m.append(mu); C.append(R @ np.diag([sl ** 2, ss ** 2]) @ R.T); w.append(wt)
    return BlockGMM(np.array(m), np.array(C), np.array(w))


def _offset_gauss():
    """one anisotropic Gaussian off the centre (item 8: unimodal blocks, so that a generic network can learn the 64-D
    product; the shear well still turns it into a ring with sheared spirals)"""
    a = math.radians(30); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return BlockGMM(np.array([[2.0, 0.5]]), np.array([R @ np.diag([0.8 ** 2, 0.25 ** 2]) @ R.T]), np.array([1.0]))


def _pinwheel():
    m, C = [], []
    for k in range(5):
        a = 2 * math.pi * k / 5; er = np.array([math.cos(a), math.sin(a)]); et = np.array([-math.sin(a), math.cos(a)])
        m.append(2.0 * er); C.append(0.3 ** 2 * np.outer(er, er) + 0.15 ** 2 * np.outer(et, et))
    return BlockGMM(np.array(m), np.array(C), np.full(5, 0.2))


def _rings(spec, n_per=360):
    m, s, w = [], [], []
    for R, noise, wt in spec:
        th = 2 * math.pi * np.arange(n_per) / n_per
        m.append(np.stack([R * np.cos(th), R * np.sin(th)], 1)); s += [noise] * n_per; w += [wt / n_per] * n_per
    return BlockGMM(np.concatenate(m), np.array(s), np.array(w))


def _spiral(arms, n_per=500, noise=0.12, turns=1.5):
    m = []
    for k in range(arms):
        t = np.linspace(0.3, turns * 2 * math.pi, n_per); r = t / (2 * math.pi); off = 2 * math.pi * k / arms
        m.append(np.stack([r * np.cos(t + off), r * np.sin(t + off)], 1))
    m = np.concatenate(m)
    return BlockGMM(m, np.full(len(m), noise), np.full(len(m), 1.0 / len(m)))


def _moons(n_per=500, noise=0.12, R=1.5):
    t = np.linspace(0, math.pi, n_per)
    a = np.stack([R * np.cos(t) + 0.5, R * np.sin(t) - 0.3], 1)
    b = np.stack([R * np.cos(t) - 0.5, -(R * np.sin(t) + 0.3)], 1)
    m = np.concatenate([a, b])
    return BlockGMM(m, np.full(len(m), noise), np.full(len(m), 1.0 / len(m)))


_REG.update({"offset_gauss": _offset_gauss, "diagonal_chain": _diag_chain, "grid_gmm": _grid, "concentric_gmm": _concentric,
             "elongated_gmm": _elongated, "pinwheel": _pinwheel,
             "ring": lambda: _rings([(2.5, 0.12, 1.0)]),
             "double_ring": lambda: _rings([(1.5, 0.10, 0.4), (3.0, 0.12, 0.6)]),
             "two_spiral": lambda: _spiral(2), "three_spiral": lambda: _spiral(3),
             "two_moon": _moons})

MAIN5 = ["asymmetric_gmm", "sparse_dense_gmm", "elongated_gmm", "two_spiral", "ring"]
SUPP3 = ["double_ring", "pinwheel", "two_moon"]


def _rare_mode_tilt():
    """entropy-tilted rare_mode (joint experiment): q = p w / Z with the T3 weight (alpha 0.5, 20% quantile, delta 0.1)"""
    from hbmc.tilt2d import TiltedGMM2D
    return TiltedGMM2D(_REG["rare_mode"](), alpha=0.5, qtheta=0.2, delta=0.1)


_REG["rare_mode_tilt"] = _rare_mode_tilt


def get(name):
    if name not in _REG:
        raise ValueError(f"unknown dataset {name}; available {sorted(_REG)}")
    return _REG[name]()


def names():
    return sorted(_REG)
