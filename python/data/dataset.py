import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path

class DrumStemDataset(Dataset):
    """
    Dataset that loads drum stems (kick, snare, toms, overheads) from take folders,
    applies augmentations, and mixes them on-the-fly.
    """
    def __init__(self, root_dir, sample_rate=44100, chunk_length_sec=4.0, 
                 stems=['kick', 'snare', 'toms', 'overheads'],
                 augment=True):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_length_sec)
        self.stems = stems
        self.augment = augment
        
        # Find all valid take folders
        self.takes = []
        for d in self.root_dir.iterdir():
            if d.is_dir():
                # Check if all stem files exist
                valid = True
                for stem in self.stems:
                    if not (d / f"{stem}.wav").exists():
                        valid = False
                        break
                if valid:
                    self.takes.append(d)
        
        if not self.takes:
            print(f"Warning: No valid takes found in {self.root_dir} containing all stems: {self.stems}")

    def __len__(self):
        # We can artificially increase length because we pick random chunks
        return len(self.takes) * 10 

    def __getitem__(self, idx):
        # Map idx to a take
        take_idx = idx % len(self.takes)
        take_dir = self.takes[take_idx]
        
        # We need to ensure we pick the *same* random chunk for all stems in this take.
        # First, find total frames (assuming all stems in a take are same length)
        info = torchaudio.info(take_dir / f"{self.stems[0]}.wav")
        total_frames = info.num_frames
        
        if total_frames > self.chunk_size:
            start_frame = random.randint(0, total_frames - self.chunk_size)
        else:
            start_frame = 0
            
        mixture = torch.zeros(1, self.chunk_size)
        target_tensors = {}
        first = True
        
        for stem in self.stems:
            file_path = take_dir / f"{stem}.wav"
            waveform, sr = torchaudio.load(file_path, frame_offset=start_frame, num_frames=self.chunk_size)
            
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            if waveform.shape[1] < self.chunk_size:
                pad_amount = self.chunk_size - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                
            # Force stereo (2 channels)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
                
            # Augmentation: Random Gain
            if self.augment:
                gain = random.uniform(0.5, 1.5) # Random gain
                waveform = waveform * gain
                
            target_tensors[stem] = waveform
            
            if first:
                mixture = waveform.clone()
                first = False
            else:
                mixture += waveform
                
        # Normalize mixture slightly to prevent clipping if we added gains
        max_val = torch.max(torch.abs(mixture))
        if max_val > 1.0:
            mixture = mixture / max_val
            for stem in self.stems:
                target_tensors[stem] = target_tensors[stem] / max_val
                
        return mixture, target_tensors
