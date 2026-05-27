"""
Drum-tuned Demucs (time-domain, U-Net style) with 4-stem output.

Based on the original Demucs architecture by Alexandre Defossez et al.
https://arxiv.org/abs/1911.13254

Key adaptations for Bleed:
- Output heads reduced to 4 drum stems: kick, snare, toms, overheads
- Mono input (single mic) → stereo output per stem
- Encoder/decoder depth and channel width tuned for drum transient preservation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit activation."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return x * torch.sigmoid(gate)


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU encoder block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 8, stride: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """ConvTranspose → BatchNorm → GLU decoder block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 8, stride: int = 4):
        super().__init__()
        # GLU doubles channels then halves
        self.conv_t = nn.ConvTranspose1d(in_ch, out_ch * 2, kernel_size, stride=stride)
        self.glu = GLU(dim=1)

    def forward(self, x):
        return self.glu(self.conv_t(x))


# ---------------------------------------------------------------------------
# Bidirectional LSTM bottleneck
# ---------------------------------------------------------------------------

class BiLSTMBottleneck(nn.Module):
    """Two-layer BiLSTM applied at the bottleneck."""
    def __init__(self, channels: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            channels, channels // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (B, C, T) → lstm expects (B, T, C)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# BleedDemucs — the main model
# ---------------------------------------------------------------------------

STEMS = ["kick", "snare", "toms", "overheads"]

class BleedDemucs(nn.Module):
    """
    Time-domain Demucs variant for drum stem separation.

    Architecture:
        Encoder:  6 convolutional blocks, doubling channels each layer.
        Bottleneck: 2-layer BiLSTM.
        Decoder:  6 mirrored transposed-conv blocks with skip connections.
        Output:   4 stems × 2 channels (stereo), produced by a final Conv1d.

    Args:
        in_channels:    Number of input channels. 1 = mono mic, 2 = stereo bus.
        out_channels:   Channels per stem (default 2 = stereo out).
        num_stems:      How many stems to separate (4: kick, snare, toms, OH).
        depth:          Number of encoder/decoder layers.
        channels:       Base number of channels in the first encoder layer.
        kernel_size:    Convolution kernel size for all blocks.
        stride:         Downsampling stride for all encoder blocks.
        lstm_layers:    Depth of the LSTM bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        num_stems: int = 4,
        depth: int = 6,
        channels: int = 64,
        kernel_size: int = 8,
        stride: int = 4,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.num_stems = num_stems
        self.out_channels = out_channels
        self.depth = depth

        # --- Encoder ---
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = channels * (2 ** i)
            self.encoder.append(ConvBlock(in_ch, out_ch, kernel_size, stride))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = BiLSTMBottleneck(in_ch, num_layers=lstm_layers)

        # --- Decoder ---
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = channels * (2 ** i)          # skip connection channels
            dec_in_ch = in_ch + skip_ch             # concat with skip
            if i == 0:
                dec_out_ch = num_stems * out_channels
            else:
                dec_out_ch = channels * (2 ** (i - 1)) if i > 0 else num_stems * out_channels
            self.decoder.append(
                ConvTransposeBlock(dec_in_ch, dec_out_ch, kernel_size, stride)
            )
            in_ch = dec_out_ch

        # Final projection to ensure exact channel count
        self.output_proj = nn.Conv1d(num_stems * out_channels, num_stems * out_channels, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, T) — the mixed drum recording.

        Returns:
            dict mapping stem name → (B, out_channels, T) tensor.
        """
        original_length = x.shape[-1]

        # Pad input to be divisible by stride^depth
        pad_to = self.stride_product()
        if original_length % pad_to != 0:
            pad_len = pad_to - (original_length % pad_to)
            x = F.pad(x, (0, pad_len))

        # --- Encode ---
        skips = []
        for enc_block in self.encoder:
            x = enc_block(x)
            skips.append(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decode with skip connections ---
        for i, dec_block in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            # Align temporal dimension (encoder output may differ by 1 due to striding)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.output_proj(x)

        # Trim back to original length
        x = x[..., :original_length]

        # Split into stems
        stem_tensors = x.chunk(self.num_stems, dim=1)
        return {name: stem for name, stem in zip(STEMS, stem_tensors)}

    def stride_product(self) -> int:
        """Total temporal downsampling factor across all encoder layers."""
        # Each ConvBlock uses stride=4 by default
        return 4 ** self.depth


# ---------------------------------------------------------------------------
# Helper: count parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n / 1e6:.2f}M"


if __name__ == "__main__":
    model = BleedDemucs(in_channels=1, depth=6, channels=64)
    print(f"BleedDemucs parameters: {count_parameters(model)}")

    dummy = torch.randn(2, 1, 44100 * 4)  # batch=2, mono, 4 sec @ 44.1 kHz
    out = model(dummy)
    for stem, t in out.items():
        print(f"  {stem}: {tuple(t.shape)}")
