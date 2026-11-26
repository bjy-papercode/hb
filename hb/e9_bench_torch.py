"""E9 (GPU TODO): cost of the training-time residuals in PyTorch on CPU or GPU, for an MLP and a
convolutional (U-Net-like) scalar log-density network, d in {16, 64, 256, 1024} (conv: d = 16x16 = 256, 32x32 = 1024).

Per training step (forward + backward through the parameters) for B anchors:
  hb_ad      heat-ball residual, grad v by autograd (double backprop)       N = Np = 16
  hb_fd      heat-ball residual, values of v only (finite differences)      N = Np = 16
  fp_exact   pointwise FP residual, exact Laplacian (d Hessian-vector products; skipped for d > 64)
  fp_hutch   pointwise FP residual, Hutchinson (1 probe)
  ssm        sliced score matching (reference cost of a score-matching step)
Reported: ms per step and peak memory (CUDA: max_memory_allocated; CPU: max RSS delta).
Usage:  python e9_bench_torch.py [--device cuda] [--B 64] [--reps 10]
Output: results_e9_<device>.json"""
import sys, os, json, math, time, argparse, resource
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hbmc.models import VNet, SinEmb
from hbmc.linear_gmm import LinearDrift
from hbmc import losses
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--B', type=int, default=64)
ap.add_argument('--reps', type=int, default=10)
ap.add_argument('--dims', default='16,64,256,1024')
ap.add_argument('--archs', default='mlp,conv')
ap.add_argument('--out', default=None, help='output json (default experiments/results_e9_<device>.json)')
a = ap.parse_args()
dev = torch.device(a.device)


class ConvV(nn.Module):
    """Small U-Net-style scalar network on a side x side image, FiLM time conditioning, Gaussian far field."""
    def __init__(self, side, ch=32):
        super().__init__(); self.side = side
        self.te = nn.Sequential(SinEmb(64), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 2 * ch))
        self.c1 = nn.Conv2d(1, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1); self.u = nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1)
        self.c4 = nn.Conv2d(2 * ch, 1, 3, padding=1)
        self.raw_c0 = nn.Parameter(torch.tensor(0.5413))

    def forward(self, x, tau):
        tau = tau.reshape(-1).expand(x.shape[0]).to(x.dtype)
        X = x.view(-1, 1, self.side, self.side)
        s, b = self.te(tau).chunk(2, -1)
        h1 = F.silu(self.c1(X)) * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h2 = F.silu(self.c3(F.silu(self.c2(h1))))
        h = self.c4(torch.cat([F.silu(self.u(h2)), h1], 1))
        c = F.softplus(self.raw_c0) + 2 * tau
        return (h.flatten(1) * x).sum(1) / math.sqrt(x.shape[1]) - (x * x).sum(1) / (2 * c)


def peak_reset():
    if dev.type == 'mps':
        torch.mps.synchronize(); torch.mps.empty_cache(); return torch.mps.driver_allocated_memory()
    if dev.type == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def peak_read(r0):
    if dev.type == 'mps':                          # no peak counter on MPS: driver memory after the run
        torch.mps.synchronize(); return (torch.mps.driver_allocated_memory() - r0) / 2 ** 20
    if dev.type == 'cuda':
        torch.cuda.synchronize(); return torch.cuda.max_memory_allocated() / 2 ** 20
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r0) / 1024


out = dict(device=str(dev), torch=torch.__version__, B=a.B, rows=[])
if dev.type == 'cuda':
    out['gpu'] = torch.cuda.get_device_name(0)
elif dev.type == 'mps':
    import platform; out['gpu'] = 'Apple MPS (' + platform.machine() + ')'
for arch in a.archs.split(','):
    for d in [int(v) for v in a.dims.split(',')]:
        if arch == 'conv' and int(math.isqrt(d)) ** 2 != d:
            continue
        torch.manual_seed(0)
        net = (VNet(d, 256, 128, 4) if arch == 'mlp' else ConvV(math.isqrt(d))).to(dev)
        D = LinearDrift.ou(d, a=1.0, omega=0.5); D.B = D.B.float().to(dev); D.dtype = torch.float32
        x = torch.randn(a.B, d, device=dev); tau = torch.rand(a.B, device=dev) * 0.9 + 0.1
        jobs = {'ssm': lambda: losses.ssm(net, x, tau),
                'hb_ad': lambda: losses.hb(net, D, x, tau, 0.3, N=16, Np=16, mode='ad'),
                'hb_fd': lambda: losses.hb(net, D, x, tau, 0.3, N=16, Np=16, mode='fd'),
                'fp_hutch': lambda: losses.fp(net, D, x, tau, lap='hutch')}
        if d <= 64:
            jobs['fp_exact'] = lambda: losses.fp(net, D, x, tau, lap='exact')
        for name, job in jobs.items():
            try:
                job().backward()                     # warm-up
                r0 = peak_reset(); t = time.perf_counter()
                for _ in range(a.reps):
                    net.zero_grad(set_to_none=True); job().backward()
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                elif dev.type == 'mps':
                    torch.mps.synchronize()
                ms = 1e3 * (time.perf_counter() - t) / a.reps
                row = dict(arch=arch, d=d, method=name, ms_per_step=ms, peak_mem_MB=peak_read(r0))
            except RuntimeError as e:                # e.g. CUDA OOM
                row = dict(arch=arch, d=d, method=name, error=str(e)[:200])
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
            out['rows'].append(row); print(row, flush=True)
        json.dump(out, open(a.out or os.path.join(HERE, f'results_e9_{dev.type}.json'), 'w'), indent=1)
