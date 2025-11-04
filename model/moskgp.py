
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Base kernels (float64, batched [B,N,d] inputs)
# ---------------------------
class _Positive(nn.Module):
    def __init__(self, init: float, eps: float = 1e-6):
        super().__init__()
        self.raw = nn.Parameter(torch.as_tensor(init, dtype=torch.float64))
        self.eps = eps
    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw) + self.eps

class RBFKernel(nn.Module):
    def __init__(self, lengthscale: float = 0.5, variance: float = 1.0):
        super().__init__()
        self._ls = _Positive(lengthscale)
        self._var = _Positive(variance)
    @property
    def lengthscale(self): return self._ls()
    @property
    def variance(self): return self._var()
    def _pairwise_sqdist(self, x1, x2):
        if x1.dim() == 2: x1 = x1.unsqueeze(0)
        if x2.dim() == 2: x2 = x2.unsqueeze(0)
        x1n = (x1 * x1).sum(-1, keepdim=True)
        x2n = (x2 * x2).sum(-1, keepdim=True).transpose(-1,-2)
        return x1n + x2n - 2.0 * x1 @ x2.transpose(-1, -2)
    def forward(self, x1, x2):
        dist_sq = self._pairwise_sqdist(x1, x2)
        ls2 = self.lengthscale ** 2
        var = self.variance
        return var * torch.exp(-0.5 * dist_sq / ls2)
    def diag(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        var = self.variance
        return var * torch.ones((x.shape[0], x.shape[1]), dtype=torch.float64, device=x.device)

class RationalQuadraticKernel(nn.Module):
    def __init__(self, lengthscale: float = 0.5, variance: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self._ls = _Positive(lengthscale)
        self._var = _Positive(variance)
        self._alpha = _Positive(alpha)
    @property
    def lengthscale(self): return self._ls()
    @property
    def variance(self): return self._var()
    @property
    def alpha(self): return self._alpha()
    def _pairwise_sqdist(self, x1, x2):
        if x1.dim() == 2: x1 = x1.unsqueeze(0)
        if x2.dim() == 2: x2 = x2.unsqueeze(0)
        x1n = (x1 * x1).sum(-1, keepdim=True)
        x2n = (x2 * x2).sum(-1, keepdim=True).transpose(-1,-2)
        return x1n + x2n - 2.0 * x1 @ x2.transpose(-1, -2)
    def forward(self, x1, x2):
        dist_sq = self._pairwise_sqdist(x1, x2)
        ls2 = self.lengthscale ** 2
        var = self.variance
        a = self.alpha
        return var * (1.0 + 0.5 * dist_sq / (a * ls2)) ** (-a)
    def diag(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        var = self.variance
        return var * torch.ones((x.shape[0], x.shape[1]), dtype=torch.float64, device=x.device)

class PeriodicKernel(nn.Module):
    def __init__(self, lengthscale=0.5, variance=1.0, period=0.25):
        super().__init__()
        self._ls = _Positive(lengthscale)
        self._var = _Positive(variance)
        self._p = _Positive(period)
    @property
    def lengthscale(self): return self._ls()
    @property
    def variance(self): return self._var()
    @property
    def period(self): return self._p()
    def forward(self, x1, x2):
        if x1.dim() == 2: x1 = x1.unsqueeze(0)
        if x2.dim() == 2: x2 = x2.unsqueeze(0)
        diff = x1 - x2.transpose(-1, -2)
        s = torch.sin(math.pi * diff / self.period)
        ls2 = self.lengthscale ** 2
        var = self.variance
        return var * torch.exp(-2.0 * (s * s) / ls2)
    def diag(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        var = self.variance
        return var * torch.ones((x.shape[0], x.shape[1]), dtype=torch.float64, device=x.device)

# ---------------------------
# Kernel dictionary (additive)
# ---------------------------
class KernelDictionary(nn.Module):
    def __init__(self, kernels: Dict[str, nn.Module], temperature: float = 1.0, share_weights: bool = True):
        super().__init__()
        assert len(kernels) >= 1, "Provide at least one base kernel"
        self.kernels = nn.ModuleDict(kernels)
        self.names = list(kernels.keys())
        self.share_weights = share_weights
        self.temperature = temperature
        self.raw_weights = nn.Parameter(torch.zeros(len(kernels), dtype=torch.float64))
    def mixture_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights / self.temperature, dim=0)
    def forward(self, x1, x2):
        w = self.mixture_weights()
        out = None
        for i, name in enumerate(self.names):
            k = self.kernels[name](x1, x2)
            out = k * w[i] if out is None else out + w[i] * k
        return out
    def diag(self, x):
        w = self.mixture_weights()
        out = None
        for i, name in enumerate(self.names):
            d = self.kernels[name].diag(x)
            out = d * w[i] if out is None else out + w[i] * d
        return out
    def weight_dict(self):
        w = self.mixture_weights().detach().cpu().numpy().tolist()
        return {n: float(w[i]) for i, n in enumerate(self.names)}

# ---------------------------
# Sparse GP (single-output)
# ---------------------------
class SparseGPLayer(nn.Module):
    def __init__(self, input_dim: int, num_inducing_points: int, kernel: nn.Module, sigma_y: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self._raw_sigma_y = nn.Parameter(torch.tensor(float(sigma_y), dtype=torch.float64))
        self.kernel = kernel
        self.S_m_raw = nn.Parameter(torch.logit(torch.linspace(0.05, 0.95, num_inducing_points, dtype=torch.float64)))
        self.m = nn.Parameter(torch.zeros((num_inducing_points, 1), dtype=torch.float64))
        self.register_buffer("S", torch.eye(num_inducing_points, dtype=torch.float64))
        self.register_buffer("Prec", torch.eye(num_inducing_points, dtype=torch.float64))
        with torch.no_grad():
            self._init_from_prior()
    @property
    def sigma_y(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._raw_sigma_y) + 1e-6
    def inducing_points(self) -> torch.Tensor:
        s = torch.sigmoid(self.S_m_raw).unsqueeze(-1)
        return s.unsqueeze(0)
    @torch.no_grad()
    def _init_from_prior(self):
        Z = self.inducing_points()
        Kuu = self.kernel(Z, Z).squeeze(0)
        L = torch.linalg.cholesky(Kuu + 1e-6 * torch.eye(self.num_inducing_points, dtype=torch.float64, device=Kuu.device))
        self.S.copy_(Kuu)
        self.Prec.copy_(torch.cholesky_inverse(L))
    def _chol_jitter(self, K: torch.Tensor, scale: float = 1e-6) -> torch.Tensor:
        jitter = scale * torch.mean(torch.diagonal(K)).clamp_min(1e-6)
        return K + jitter * torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    def _Ht_and_H(self, Kuu: torch.Tensor, Kuf: torch.Tensor):
        Kuu_jit = self._chol_jitter(Kuu)
        Luu = torch.linalg.cholesky(Kuu_jit)
        Linv_Kuf = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
        Ht = torch.linalg.solve_triangular(Luu.transpose(-1, -2), Linv_Kuf, upper=True)
        H = Ht.transpose(-1, -2)
        return Ht, H, Luu
    def forward(self, x: torch.Tensor):
        Z = self.inducing_points()
        Kuu = self.kernel(Z, Z).squeeze(0)
        Kuf = self.kernel(Z, x)
        Kff = self.kernel(x, x)
        Ht, H, _ = self._Ht_and_H(Kuu, Kuf)
        mean = torch.einsum("bnm,md->bnd", H, self.m)
        cov  = Kff - torch.matmul(H, Kuf) + torch.matmul(torch.matmul(H, self.S), Ht)
        return mean, cov, Kuu, Kuf, Kff, Z
    def filtered_elbo_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 2: y = y.unsqueeze(-1)
        B, N, _ = x.shape
        Z = self.inducing_points()
        Kuu = self.kernel(Z, Z).squeeze(0)
        Kuf = self.kernel(Z, x)
        Kff = self.kernel(x, x)
        Ht, H, Luu = self._Ht_and_H(Kuu, Kuf)
        m_pred = torch.einsum("bnm,md->bnd", H, self.m)
        HS = torch.einsum("bnm,mp->bnp", H, self.S)
        S_pred = torch.einsum("bnm,bpm->bnp", HS, H)
        S_pred = S_pred + (self.sigma_y ** 2) * torch.eye(N, dtype=x.dtype, device=x.device).unsqueeze(0)
        R = y - m_pred
        Ls = torch.linalg.cholesky(self._chol_jitter(S_pred))
        alpha = torch.linalg.solve_triangular(Ls, R, upper=False)
        logdetS = 2.0 * torch.sum(torch.log(torch.diagonal(Ls, dim1=1, dim2=2)), dim=1)
        quad = torch.sum(alpha**2, dim=(1,2))
        loglik = -0.5 * (N * 1 * math.log(2.0 * math.pi) + logdetS + quad)
        A = torch.linalg.solve_triangular(Luu, Kuf, upper=False) / self.sigma_y
        trAAT = torch.einsum("bmn,bmn->b", A, A)
        trK = torch.sum(self.kernel.diag(x), dim=1)
        trace_term = -0.5 * (trK - trAAT) / (self.sigma_y ** 2)
        return (loglik + trace_term).mean()
    @torch.no_grad()
    def update_online(self, x: torch.Tensor, y: torch.Tensor):
        if y.dim() == 2: y = y.unsqueeze(-1)
        Z = self.inducing_points()
        Kuu = self.kernel(Z, Z).squeeze(0)
        Kuf = self.kernel(Z, x)
        Ht, _, _ = self._Ht_and_H(Kuu, Kuf)
        sig2_inv = 1.0 / (self.sigma_y ** 2)
        HtH = torch.einsum("bmn,bpn->bmp", Ht, Ht).mean(dim=0)
        Prec_next = self.Prec + sig2_inv * HtH
        L = torch.linalg.cholesky(self._chol_jitter(Prec_next))
        S_next = torch.cholesky_inverse(L)
        b_term = sig2_inv * torch.mean(torch.bmm(Ht, y), dim=0) + self.Prec @ self.m
        m_next = torch.linalg.solve(Prec_next, b_term)
        self.m.data.copy_(m_next)
        self.S.data.copy_(S_next)
        self.Prec.data.copy_(Prec_next)
    def compute_vfe_loss(self, x: torch.Tensor, y: torch.Tensor):
        felbo = self.filtered_elbo_term(x, y)
        zero = torch.tensor(0.0, dtype=torch.float64, device=x.device)
        return felbo, zero, zero, zero

class MultiOutputSparseGPLayer(nn.Module):
    def __init__(self, input_dim: int, num_inducing_points: int, num_outputs: int, kernel: nn.Module, sigma_y: float = 0.1, share_kernel: bool = True):
        super().__init__()
        self.num_outputs = num_outputs
        if share_kernel:
            self.gps = nn.ModuleList([
                SparseGPLayer(input_dim, num_inducing_points, kernel, sigma_y=sigma_y)
                for _ in range(num_outputs)
            ])
        else:
            import copy
            self.gps = nn.ModuleList([
                SparseGPLayer(input_dim, num_inducing_points, copy.deepcopy(kernel), sigma_y=sigma_y)
                for _ in range(num_outputs)
            ])
    def forward(self, x: torch.Tensor):
        means, covs, Zs = [], [], []
        for gp in self.gps:
            m, c, Kuu, Kuf, Kff, Z = gp.forward(x)
            means.append(m)
            covs.append(c)
            Zs.append(Z)
        return means, covs, Zs
    def update_online(self, x_list: List[torch.Tensor], y_list: List[torch.Tensor]):
        for gp, x, y in zip(self.gps, x_list, y_list):
            gp.update_online(x, y)
    def compute_loss(self, x_list: List[torch.Tensor], y_list: List[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for gp, x, y in zip(self.gps, x_list, y_list):
            felbo, *_ = gp.compute_vfe_loss(x, y)
            total = total + felbo
        return total / self.num_outputs
