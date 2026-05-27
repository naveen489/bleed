"""
Expanded loss library for BleedDemucs drum separation.

Available losses:
    si_sdr   - Scale-Invariant SDR (waveform domain, scale-invariant)
    l1       - Mean Absolute Error (simple, time-domain baseline)
    mse      - Mean Squared Error (penalises large errors more)
    mrstft   - Multi-Resolution STFT (spectral domain, good for transients)
    hybrid   - MRSTFT + SI-SDR combined (recommended for drums)

Usage:
    from python.training.losses import get_loss
    criterion = get_loss('hybrid', stems=['kick','snare','toms','overheads'])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _to_mono(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) → (B, T) by averaging channels."""
    if x.dim() == 3:
        return x.mean(dim=1)
    return x


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample SI-SDR in dB. estimate/target: (B, C, T)."""
    est = _to_mono(estimate)
    tgt = _to_mono(target)
    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)
    dot = (est * tgt).sum(dim=-1, keepdim=True)
    tgt_energy = (tgt ** 2).sum(dim=-1, keepdim=True) + eps
    proj = (dot / tgt_energy) * tgt
    noise = est - proj
    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


# ---------------------------------------------------------------------------
# SI-SDR Loss
# ---------------------------------------------------------------------------

class SISNRLoss(nn.Module):
    """Negative mean SI-SDR across all stems. Good overall metric."""

    def __init__(self, stems: list[str], **kwargs):
        super().__init__()
        self.stems = stems

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        device = next(iter(predictions.values())).device
        total = torch.zeros(1, device=device)
        for s in self.stems:
            total = total + si_sdr(predictions[s], targets[s]).mean()
        return -total / len(self.stems)


# ---------------------------------------------------------------------------
# L1 / MSE Baselines
# ---------------------------------------------------------------------------

class L1Loss(nn.Module):
    """Simple waveform L1. Fast but doesn't directly optimize perceptual quality."""

    def __init__(self, stems: list[str], **kwargs):
        super().__init__()
        self.stems = stems

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        total = 0.0
        for s in self.stems:
            total = total + F.l1_loss(predictions[s], targets[s])
        return total / len(self.stems)


class MSELoss(nn.Module):
    """Waveform MSE. Over-penalises phase mismatches, often produces 'muddy' separation."""

    def __init__(self, stems: list[str], **kwargs):
        super().__init__()
        self.stems = stems

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        total = 0.0
        for s in self.stems:
            total = total + F.mse_loss(predictions[s], targets[s])
        return total / len(self.stems)


# ---------------------------------------------------------------------------
# Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------

# (fft_size, hop_size, win_size)
# Small FFT → resolves fast transients (kick/snare attack within 1ms)
# Large FFT → resolves tonal content and low-frequency pitch
_RESOLUTIONS = [
    (256,   64,   256),   # ~1.5ms resolution  – transient detail
    (1024,  256,  1024),  # ~6ms  resolution  – mid-range content
    (2048,  512,  2048),  # ~12ms resolution  – low-freq + room
]


def _stft_mag(signal: torch.Tensor, fft: int, hop: int, win: int,
              window: torch.Tensor) -> torch.Tensor:
    """(B, T) → (B, F, time_frames) magnitude."""
    return torch.stft(
        signal, n_fft=fft, hop_length=hop, win_length=win,
        window=window, return_complex=True,
    ).abs()


class MRSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss.

    For each stem, computes spectral convergence + log-magnitude L1 at three
    resolutions. This strongly preserves drum transients (which waveform losses miss)
    while also capturing tonal content at larger windows.
    """

    def __init__(self, stems: list[str], sample_rate: int = 44100, **kwargs):
        super().__init__()
        self.stems = stems
        # Register Hann windows as non-trainable buffers
        for i, (_, _, ws) in enumerate(_RESOLUTIONS):
            self.register_buffer(f"window_{i}", torch.hann_window(ws))

    def _windows(self):
        return [getattr(self, f"window_{i}") for i in range(len(_RESOLUTIONS))]

    def _stem_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_m = _to_mono(pred)
        tgt_m = _to_mono(target)
        loss = torch.zeros(1, device=pred.device)

        for (fft, hop, win), window in zip(_RESOLUTIONS, self._windows()):
            pm = _stft_mag(pred_m, fft, hop, win, window)
            tm = _stft_mag(tgt_m,  fft, hop, win, window)

            # Guard: skip spectral convergence if target is near-silent
            # (||T||_F < threshold), which would blow up the ratio.
            # This happens with clean stems that have no hits in the chunk.
            norm_tgt = torch.norm(tm, p="fro")
            if norm_tgt > 1e-4:
                sc = torch.norm(tm - pm, p="fro") / (norm_tgt + 1e-8)
            else:
                sc = torch.zeros(1, device=pred.device)

            # Log-magnitude L1 (compresses dynamic range, treats all freqs equally)
            log_l1 = F.l1_loss(torch.log(pm + 1e-7), torch.log(tm + 1e-7))

            loss = loss + sc + log_l1

        return loss / len(_RESOLUTIONS)

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        total = torch.zeros(1, device=next(iter(predictions.values())).device)
        for s in self.stems:
            total = total + self._stem_loss(predictions[s], targets[s])
        return total / len(self.stems)


# ---------------------------------------------------------------------------
# Hybrid Loss (recommended for drums)
# ---------------------------------------------------------------------------

class HybridLoss(nn.Module):
    """
    MRSTFT + SI-SDR combined.

    Why this works well for drums:
    - MRSTFT: preserves transient timing (attack of kick/snare) and spectral shape
    - SI-SDR: ensures overall waveform alignment and scale invariance

    Default weighting: equal split. Increase w_mrstft if transients are weak.
    Increase w_sisdr if global level/bleed is the dominant problem.
    """

    def __init__(self, stems: list[str], sample_rate: int = 44100,
                 w_mrstft: float = 0.5, w_sisdr: float = 0.5, **kwargs):
        super().__init__()
        self.mrstft = MRSTFTLoss(stems, sample_rate)
        self.sisdr  = SISNRLoss(stems)
        self.w_mrstft = w_mrstft
        self.w_sisdr  = w_sisdr

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        return (self.w_mrstft * self.mrstft(predictions, targets) +
                self.w_sisdr  * self.sisdr(predictions, targets))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "si_sdr":  SISNRLoss,
    "l1":      L1Loss,
    "mse":     MSELoss,
    "mrstft":  MRSTFTLoss,
    "hybrid":  HybridLoss,
}


def get_loss(name: str, stems: list[str], **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](stems, **kwargs)
