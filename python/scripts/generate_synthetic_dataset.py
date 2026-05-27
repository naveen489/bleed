"""
Generate a synthetic drum dataset for pipeline validation.

Creates drum-like audio stems (kick, snare, toms, overheads) using
synthesized waveforms so the full training pipeline can be tested
without needing real multitrack recordings.

This is intentionally not realistic — it's purely for validating
that the DataLoader → Model → Loss → Optimizer pipeline is wired
correctly (i.e., the "overfit-one" sanity check).

Usage:
    # Create 5 takes, 30 seconds each, at 44.1 kHz
    python python/scripts/generate_synthetic_dataset.py --output_dir data/raw --num_takes 5 --duration 30

    # Smaller/faster for a quick test
    python python/scripts/generate_synthetic_dataset.py --output_dir data/raw --num_takes 3 --duration 10
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


def apply_envelope(signal: np.ndarray, attack: int, decay: int) -> np.ndarray:
    """Apply a simple attack-decay envelope to a signal burst."""
    env = np.zeros(len(signal))
    atk = min(attack, len(signal))
    dec = min(decay, len(signal) - atk)
    if atk > 0:
        env[:atk] = np.linspace(0, 1, atk)
    if dec > 0:
        env[atk:atk + dec] = np.linspace(1, 0, dec)
    return signal * env


def synth_kick(sr: int, duration_samples: int, bpm: float = 100.0) -> np.ndarray:
    """Synthesise a kick drum pattern: pitched sine + noise burst, 4-on-the-floor."""
    out = np.zeros(duration_samples)
    beat_samples = int(sr * 60.0 / bpm)
    for beat in range(0, duration_samples, beat_samples):
        n = min(int(sr * 0.25), duration_samples - beat)
        if n <= 0:
            break
        t = np.linspace(0, 0.25, n)
        freq = np.linspace(160, 50, n)  # Pitch drop: 160Hz → 50Hz
        sine = np.sin(2 * np.pi * np.cumsum(freq) / sr)
        noise = np.random.randn(n) * 0.15
        burst = apply_envelope(sine + noise, int(sr * 0.002), int(sr * 0.1))
        out[beat:beat + n] += burst * 0.9
    return np.clip(out, -1.0, 1.0)


def synth_snare(sr: int, duration_samples: int, bpm: float = 100.0) -> np.ndarray:
    """Synthesise a snare drum pattern: noise + sine at beats 2 and 4."""
    out = np.zeros(duration_samples)
    beat_samples = int(sr * 60.0 / bpm)
    for beat_idx, beat in enumerate(range(0, duration_samples, beat_samples)):
        if beat_idx % 4 not in (1, 3):  # Only beats 2 and 4
            continue
        n = min(int(sr * 0.15), duration_samples - beat)
        if n <= 0:
            break
        noise = np.random.randn(n) * 0.8
        tone = np.sin(2 * np.pi * 200 * np.linspace(0, 0.15, n))
        burst = apply_envelope(noise + tone * 0.3, int(sr * 0.001), int(sr * 0.07))
        out[beat:beat + n] += burst * 0.8
    return np.clip(out, -1.0, 1.0)


def synth_toms(sr: int, duration_samples: int, bpm: float = 100.0) -> np.ndarray:
    """Synthesise sparse tom hits (every 2 bars on beat 3)."""
    out = np.zeros(duration_samples)
    beat_samples = int(sr * 60.0 / bpm)
    bar_samples = beat_samples * 4
    freqs = [120, 95, 75]  # Hi, mid, floor tom
    for bar_idx, bar in enumerate(range(0, duration_samples, bar_samples)):
        if bar_idx % 2 != 0:
            continue
        beat_3 = bar + 2 * beat_samples
        if beat_3 >= duration_samples:
            break
        freq = freqs[bar_idx % len(freqs)]
        n = min(int(sr * 0.3), duration_samples - beat_3)
        t = np.linspace(0, 0.3, n)
        sine = np.sin(2 * np.pi * freq * t)
        noise = np.random.randn(n) * 0.1
        burst = apply_envelope(sine + noise, int(sr * 0.003), int(sr * 0.15))
        out[beat_3:beat_3 + n] += burst * 0.7
    return np.clip(out, -1.0, 1.0)


def synth_overheads(sr: int, duration_samples: int, bpm: float = 100.0) -> np.ndarray:
    """Synthesise hi-hat / cymbal pattern: 8th note filtered noise."""
    out = np.zeros(duration_samples)
    eighth_samples = int(sr * 60.0 / bpm / 2)
    for hit in range(0, duration_samples, eighth_samples):
        n = min(int(sr * 0.08), duration_samples - hit)
        if n <= 0:
            break
        noise = np.random.randn(n)
        # High-pass by differencing (crude but effective for a test signal)
        if len(noise) > 1:
            noise = np.diff(noise, prepend=noise[0])
        vel = 0.4 if (hit // eighth_samples) % 2 == 0 else 0.25
        burst = apply_envelope(noise, int(sr * 0.001), int(sr * 0.04))
        out[hit:hit + n] += burst * vel
    return np.clip(out, -1.0, 1.0)


def write_stereo(path: Path, mono: np.ndarray, sr: int, spread: float = 0.05):
    """Write a mono signal as a stereo WAV with slight random panning variation."""
    left  = mono + np.random.randn(len(mono)) * spread * 0.05
    right = mono + np.random.randn(len(mono)) * spread * 0.05
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(str(path), stereo, sr, subtype="PCM_24")


def generate_take(take_dir: Path, sr: int, duration_sec: float, bpm: float):
    take_dir.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_sec)
    stems = {
        "kick":      synth_kick(sr, n, bpm),
        "snare":     synth_snare(sr, n, bpm),
        "toms":      synth_toms(sr, n, bpm),
        "overheads": synth_overheads(sr, n, bpm),
    }
    for name, signal in stems.items():
        out_path = take_dir / f"{name}.wav"
        write_stereo(out_path, signal, sr)
        print(f"  Wrote {out_path}")


def main(args):
    output_dir = Path(args.output_dir)
    print(f"Generating {args.num_takes} synthetic take(s) -> {output_dir}\n")
    for i in range(args.num_takes):
        bpm = np.random.uniform(85, 130)
        take_dir = output_dir / f"Synthetic_Take_{i+1:02d}"
        print(f"Take {i+1}/{args.num_takes}  (BPM={bpm:.1f})")
        generate_take(take_dir, args.sample_rate, args.duration, bpm)

    print(f"\nDone. {args.num_takes} takes written to {output_dir}")
    print("You can now run the training pipeline:")
    print(f"  python python/training/train.py --data_dir {output_dir} --overfit_one --epochs 100 --batch_size 2 --depth 4 --channels 32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic drum dataset for pipeline testing")
    parser.add_argument("--output_dir",   type=str,   default="data/raw",  help="Where to write the take folders")
    parser.add_argument("--num_takes",    type=int,   default=5,           help="Number of takes to generate")
    parser.add_argument("--duration",     type=float, default=30.0,        help="Duration of each take in seconds")
    parser.add_argument("--sample_rate",  type=int,   default=44100)
    args = parser.parse_args()
    main(args)
