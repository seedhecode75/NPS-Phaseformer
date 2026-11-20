"""Small, independently ablatable changes to the released Phaseformer."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model_with_eca import inv_mag


def phase_only(x):
    # FFT in float32 under autocast, float64 retained for numerical tests.
    with torch.autocast(device_type=x.device.type, enabled=False):
        z = x if x.dtype == torch.float64 else x.float()
        return inv_mag(z).to(x.dtype)


class UnitPhase(nn.Module):
    """Algebra-only speed control: F/abs(F), preserving upstream zero phasors."""
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x if x.dtype == torch.float64 else x.float()
            f = torch.fft.fft2(z)
            a = f.abs()
            unit = f / torch.where(a > 0, a, torch.ones_like(a))
            unit = torch.where(a > 0, unit, torch.ones_like(unit))
            return torch.fft.ifft2(unit).real.to(x.dtype)


class ReliabilityPhase(nn.Module):
    """R(A) exp(i phi) = F*A/(A^2+tau^2), without an angle singularity.

    Per-example, per-channel scale; one shared positive threshold per module.
    full: handoff RMS; ac: exclude DC from RMS (still gate all coefficients).
    This is a shrinkage heuristic, NOT a calibrated posterior probability.
    """
    def __init__(self, tau=0.1, learnable=True, scale='full', eps=1e-8):
        super().__init__()
        if tau <= 0 or scale not in ('full', 'ac'):
            raise ValueError('tau > 0; scale must be full or ac')
        t = torch.tensor(math.log(math.expm1(tau)))
        if learnable:
            self.raw_tau = nn.Parameter(t)
        else:
            self.register_buffer('raw_tau', t)
        self.scale, self.eps = scale, eps

    def spectrum(self, x):
        z = torch.fft.fft2(x)
        a = z.abs()
        power = a.square()
        if self.scale == 'ac':
            n = x.shape[-2] * x.shape[-1]
            mean_power = (power.sum((-2, -1), keepdim=True) - power[..., :1, :1]) / max(n - 1, 1)
        else:
            mean_power = power.mean((-2, -1), keepdim=True)
        tau2 = F.softplus(self.raw_tau).square() * mean_power.clamp_min(self.eps**2)
        denom = power + tau2 + self.eps**2
        return z * (a / denom)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x if x.dtype == torch.float64 else x.float()
            return torch.fft.ifft2(self.spectrum(z)).real.to(x.dtype)


def phase_descriptors(p, eps=1e-8):
    """Nonlinear-before-pooling descriptors, [B,2,C].

    Squared energy alone cannot encode phase (Parseval); L1 statistics can.
    Circular box averaging preserves cyclic shift equivariance.
    """
    z = p - p.mean((-2, -1), keepdim=True)
    rms = (z.square().mean((-2, -1), keepdim=True) + eps**2).sqrt()
    local = F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode='circular'), 3, stride=1)
    return torch.stack(((z.abs().mean((-2, -1), keepdim=True) / rms).squeeze(-1).squeeze(-1),
                        (local.abs().mean((-2, -1), keepdim=True) / rms).squeeze(-1).squeeze(-1)), dim=1)


class PhaseStatisticSkip(nn.Module):
    """Retain legacy logit, add a zero-initialized phase-statistic correction."""
    def __init__(self, original, extractor=None, descriptor='phase', statistics='both'):
        super().__init__()
        self.avg_pool, self.conv, self.sigmoid = original.avg_pool, original.conv, original.sigmoid
        self.extractor = extractor
        self.descriptor, self.statistics = descriptor, statistics
        k = original.conv.kernel_size[0]
        self.correction = nn.Conv1d(2 if statistics == 'both' else 1, 1, k, padding=k // 2, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, x):
        p = phase_only(x) if self.extractor is None else self.extractor(x)
        legacy = self.conv(self.avg_pool(p).flatten(2).transpose(1, 2))
        # 'spatial' is the matched-parameter non-Fourier control.
        d = phase_descriptors(x.float() if self.descriptor == 'spatial' else p.float())
        if self.statistics == 'l1':
            d = d[:, :1]
        logit = legacy + self.correction(d.to(self.correction.weight.dtype))
        return x * torch.sigmoid(logit.transpose(1, 2).unsqueeze(-1)).to(x.dtype)


class ReliabilitySkip(nn.Module):
    """Handoff skip-only RA control: deliberately retains DC-only pooling."""
    def __init__(self, original, extractor):
        super().__init__()
        self.avg_pool, self.conv, self.sigmoid = original.avg_pool, original.conv, original.sigmoid
        self.extractor = extractor

    def forward(self, x):
        p = self.extractor(x)
        y = self.conv(self.avg_pool(p).flatten(2).transpose(1, 2))
        return x * self.sigmoid(y.transpose(1, 2).unsqueeze(-1))


class ReliableMDTA(nn.Module):
    def __init__(self, original, **kwargs):
        super().__init__()
        self.num_heads, self.temperature = original.num_heads, original.temperature
        self.qkv, self.qkv_conv, self.project_out = original.qkv, original.qkv_conv, original.project_out
        self.q_phase, self.k_phase = ReliabilityPhase(**kwargs), ReliabilityPhase(**kwargs)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = self.q_phase(q).reshape(b, self.num_heads, -1, h*w)
        k = self.k_phase(k).reshape(b, self.num_heads, -1, h*w)
        v = v.reshape(b, self.num_heads, -1, h*w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax((q @ k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        return self.project_out((attn @ v).reshape(b, c, h, w))
