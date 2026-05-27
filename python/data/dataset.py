import random
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class DrumStemDataset(Dataset):
    """
    Dataset that loads drum stems (kick, snare, toms, overheads) from take folders,
    applies augmentations, and mixes them on-the-fly.

    Uses soundfile for loading to avoid torchaudio backend version issues.
    """
    def __init__(self, root_dir, sample_rate=44100, chunk_length_sec=4.0,
                 stems=None, augment=True):
        if stems is None:
            stems = ['kick', 'snare', 'toms', 'overheads']
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_length_sec)
        self.stems = stems
        self.augment = augment

        # Find all valid take folders
        self.takes = []
        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir():
                if all((d / f"{s}.wav").exists() for s in self.stems):
                    self.takes.append(d)

        if not self.takes:
            print(f"Warning: No valid takes found in {self.root_dir} containing all stems: {self.stems}")

    def __len__(self):
        # Artificially inflate length; we pick random chunks each call
        return len(self.takes) * 10

    def _load_chunk(self, path: Path, start_frame: int) -> torch.Tensor:
        """
        Load a fixed-length chunk from a WAV file using soundfile.
        Returns a (2, chunk_size) float32 tensor (stereo).
        """
        info = sf.info(str(path))
        end_frame = min(start_frame + self.chunk_size, info.frames)

        data, sr = sf.read(
            str(path),
            start=start_frame,
            stop=end_frame,
            dtype="float32",
            always_2d=True,   # always (frames, channels)
        )
        # data shape: (frames, channels)
        waveform = torch.from_numpy(data.T)  # → (channels, frames)

        # Resample if needed (simple linear resample via torch)
        if sr != self.sample_rate:
            # Use interpolate as a quick resampler (quality is fine for training)
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                size=int(waveform.shape[-1] * self.sample_rate / sr),
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        # Pad if shorter than chunk_size
        if waveform.shape[-1] < self.chunk_size:
            waveform = F.pad(waveform, (0, self.chunk_size - waveform.shape[-1]))
        else:
            waveform = waveform[:, :self.chunk_size]

        # Ensure exactly 2 channels
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform  # (2, chunk_size)

    def __getitem__(self, idx):
        take_idx = idx % len(self.takes)
        take_dir = self.takes[take_idx]

        # Get total frames to pick a random start
        info = sf.info(str(take_dir / f"{self.stems[0]}.wav"))
        total_frames = info.frames

        start_frame = (
            random.randint(0, max(0, total_frames - self.chunk_size))
            if total_frames > self.chunk_size else 0
        )

        target_tensors = {}
        mixture = None

        for stem in self.stems:
            waveform = self._load_chunk(take_dir / f"{stem}.wav", start_frame)

            # Augmentation: random gain per stem
            if self.augment:
                gain = random.uniform(0.5, 1.5)
                waveform = waveform * gain

            target_tensors[stem] = waveform
            mixture = waveform.clone() if mixture is None else mixture + waveform

        # Normalise mixture to prevent clipping
        max_val = mixture.abs().max()
        if max_val > 1.0:
            scale = 1.0 / max_val
            mixture = mixture * scale
            target_tensors = {k: v * scale for k, v in target_tensors.items()}

        return mixture, target_tensors
