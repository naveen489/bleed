"""
Loss functions for drum stem separation.

We use Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as the primary loss.
SI-SDR is the standard metric in source separation and is invariant to the
absolute scale of the output — critical for drums where transient peaks vary wildly.

Reference: Le Roux et al., "SDR – Half-baked or Well Done?" (ICASSP 2019)
"""

import torch
import torch.nn as nn


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant SDR (SI-SDR) between estimate and target.

    Args:
        estimate: (B, C, T) predicted stem waveform.
        target:   (B, C, T) ground-truth stem waveform.
        eps:      Numerical stability floor.

    Returns:
        si_sdr_val: (B,) per-sample SI-SDR in dB.
    """
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection of estimate onto target
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    projection = (dot / target_energy) * target

    noise = estimate - projection

    ratio = (projection ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


class SISNRLoss(nn.Module):
    """
    Negative mean SI-SDR over all stems (minimise to maximise separation quality).

    Args:
        stems: List of stem names, must match the keys in model output dict.
    """

    def __init__(self, stems: list[str]):
        super().__init__()
        self.stems = stems

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            predictions: dict[stem → (B, C, T)] model outputs.
            targets:     dict[stem → (B, C, T)] ground-truth stems.

        Returns:
            loss: scalar tensor (negative mean SI-SDR across stems and batch).
        """
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        for stem in self.stems:
            sdr = si_sdr(predictions[stem], targets[stem])   # (B,)
            total = total + sdr.mean()

        return -total / len(self.stems)
