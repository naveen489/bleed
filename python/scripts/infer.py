"""
Offline inference: load a trained BleedDemucs checkpoint, run it on a
mixed audio file (or a take folder), and write the separated stems to disk.

Usage — run on a prepared take folder:
    python python/scripts/infer.py \
        --checkpoint models/checkpoints/final.pt \
        --take_dir   data/raw/Synthetic_Take_01 \
        --output_dir output/Synthetic_Take_01

Usage — run on a raw WAV mix file:
    python python/scripts/infer.py \
        --checkpoint models/checkpoints/final.pt \
        --mix_file   my_drum_recording.wav \
        --output_dir output/my_drums
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.models.demucs import BleedDemucs, STEMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = 44100) -> tuple[torch.Tensor, int]:
    """Load a WAV as a (2, T) float32 tensor, resampled to target_sr."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)          # (channels, T)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    if sr != target_sr:
        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=int(waveform.shape[-1] * target_sr / sr),
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        sr = target_sr

    return waveform, sr


def save_wav(path: Path, waveform: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu().numpy().T     # (T, channels)
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_24")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BleedDemucs, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Infer model config from checkpoint state_dict heuristically
    sd = ckpt.get("model_state_dict", ckpt)

    # Try to reconstruct the model — read depth and channel count from stored keys
    # Fall back to defaults matching our overfit-test config (depth=3, ch=16)
    model = BleedDemucs(
        in_channels=1,
        out_channels=2,
        num_stems=4,
        depth=ckpt.get("depth", 3),
        channels=ckpt.get("channels", 16),
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, ckpt


def chunk_infer(
    model: BleedDemucs,
    waveform: torch.Tensor,         # (2, T)
    device: torch.device,
    chunk_sec: float = 4.0,
    sr: int = 44100,
    overlap: float = 0.25,
) -> dict[str, torch.Tensor]:
    """
    Overlap-add inference on a full-length waveform.

    Splits the stereo mix into overlapping chunks, runs the model on each
    (as a mono sum), and reconstructs each stem with an overlap-add Hann window.
    """
    chunk_len  = int(chunk_sec * sr)
    hop_len    = int(chunk_len * (1.0 - overlap))
    T          = waveform.shape[-1]

    # Output accumulators
    out_stems = {s: torch.zeros(2, T) for s in STEMS}
    weight    = torch.zeros(T)

    window = torch.hann_window(chunk_len)

    pos = 0
    while pos < T:
        end = min(pos + chunk_len, T)
        chunk = waveform[:, pos:end]                    # (2, chunk)

        # Pad to chunk_len if needed
        if chunk.shape[-1] < chunk_len:
            chunk = F.pad(chunk, (0, chunk_len - chunk.shape[-1]))

        # Convert to mono (model expects 1-channel input)
        mono = chunk.mean(dim=0, keepdim=True).unsqueeze(0).to(device)  # (1, 1, chunk_len)

        with torch.no_grad():
            preds = model(mono)                          # dict[str → (1, 2, chunk_len)]

        w = window[:end - pos]
        for s in STEMS:
            stem_chunk = preds[s][0, :, :end - pos].cpu()  # (2, actual_len)
            out_stems[s][:, pos:end] += stem_chunk * w
        weight[pos:end] += w

        pos += hop_len
        if pos >= T:
            break

    # Normalize by overlap weight
    weight = weight.clamp(min=1e-8)
    for s in STEMS:
        out_stems[s] /= weight.unsqueeze(0)

    return out_stems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return

    model, _ = load_model(ckpt_path, device)
    print(f"Model       : BleedDemucs loaded OK")

    # --- Build the mixture ---
    if args.take_dir:
        take_dir = Path(args.take_dir)
        print(f"Take folder : {take_dir}")
        mixture = None
        for stem in STEMS:
            wav_path = take_dir / f"{stem}.wav"
            if not wav_path.exists():
                print(f"  WARNING: missing {wav_path}, skipping stem")
                continue
            w, sr = load_wav(wav_path, args.sample_rate)
            mixture = w.clone() if mixture is None else mixture + w
        if mixture is None:
            print("ERROR: no stems found in take_dir")
            return
        # Normalise
        mx = mixture.abs().max()
        if mx > 1.0:
            mixture = mixture / mx
    elif args.mix_file:
        print(f"Mix file    : {args.mix_file}")
        mixture, sr = load_wav(Path(args.mix_file), args.sample_rate)
    else:
        print("ERROR: provide --take_dir or --mix_file")
        return

    sr = args.sample_rate
    print(f"Mix length  : {mixture.shape[-1] / sr:.2f}s  ({mixture.shape[-1]} samples @ {sr} Hz)")

    # --- Infer ---
    print("Running inference ...")
    stems = chunk_infer(model, mixture, device, chunk_sec=args.chunk_sec, sr=sr)

    # --- Save ---
    output_dir = Path(args.output_dir)
    print(f"\nWriting stems to {output_dir}/")

    # Also save the input mixture for easy A/B listening
    save_wav(output_dir / "_mixture.wav", mixture, sr)
    print(f"  _mixture.wav  (original mix)")

    for stem_name, stem_wav in stems.items():
        out_path = output_dir / f"{stem_name}.wav"
        save_wav(out_path, stem_wav, sr)
        peak = stem_wav.abs().max().item()
        print(f"  {stem_name}.wav  (peak={peak:.3f})")

    print(f"\nDone. Open the files in {output_dir}/ to listen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BleedDemucs offline inference")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--take_dir",     type=str, default=None,
                        help="Take folder containing kick/snare/toms/overheads.wav (stems summed as input)")
    parser.add_argument("--mix_file",     type=str, default=None,
                        help="Path to a raw drum bus WAV file")
    parser.add_argument("--output_dir",   type=str, default="output",
                        help="Directory to write separated stems")
    parser.add_argument("--sample_rate",  type=int, default=44100)
    parser.add_argument("--chunk_sec",    type=float, default=0.25,
                        help="Chunk size for inference (match training chunk_sec)")
    args = parser.parse_args()
    main(args)
