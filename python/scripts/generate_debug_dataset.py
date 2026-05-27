"""
Controlled synthetic debugging dataset for BleedDemucs diagnostic verification.

Generates 10-20 tightly controlled drum examples with:
  - Realistic synthesis (kick sweep, snare body+buzz, toms, hi-hat+crash)
  - Physical bleed simulation (each mic bleeds into adjacent mics)
  - Simple EQ, soft-knee compression, and algorithmic reverb
  - Intentional variation in BPM, tuning, and velocity

The goal is a dataset TINY enough to overfit on quickly (< 5 min CPU),
but REALISTIC enough that failure to overfit reveals a real pipeline problem.

Usage:
    python python/scripts/generate_debug_dataset.py --output_dir data/debug --num_takes 15
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

SR = 44100


# ---------------------------------------------------------------------------
# Signal Processing Primitives
# ---------------------------------------------------------------------------

def db(x: float) -> float:
    return 10 ** (x / 20.0)


def normalize(signal: np.ndarray, peak_db: float = -3.0) -> np.ndarray:
    mx = np.max(np.abs(signal)) + 1e-9
    return signal / mx * db(peak_db)


def hpf(signal: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, signal)


def lpf(signal: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, signal)


def bpf(signal: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, signal)


def peak_eq(signal: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Simple peak EQ via biquad."""
    w0 = 2 * np.pi * freq / SR
    A = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha * A;   b1 = -2 * np.cos(w0);  b2 = 1 - alpha * A
    a0 = 1 + alpha / A;   a1 = -2 * np.cos(w0);  a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    from scipy.signal import lfilter
    return lfilter(b, a, signal)


def compress(signal: np.ndarray, threshold_db: float = -12, ratio: float = 3.0,
             makeup_db: float = 0.0) -> np.ndarray:
    """Static soft-knee compressor (no look-ahead, vectorized)."""
    thr = db(threshold_db)
    makeup = db(makeup_db)
    rms = np.abs(signal)
    # Soft knee: transition 6dB around threshold
    knee_width = 6.0
    knee_thr_low  = db(threshold_db - knee_width / 2)
    knee_thr_high = db(threshold_db + knee_width / 2)

    gain = np.ones_like(signal)
    above = rms > knee_thr_high
    knee  = (rms > knee_thr_low) & ~above

    # Linear region above threshold
    gain[above] = (thr + (rms[above] - thr) / ratio) / (rms[above] + 1e-9)
    # Soft knee region
    if np.any(knee):
        k = rms[knee]
        gain[knee] = (thr * 0.5 + (k - thr * 0.5) / (ratio * 0.5 + 0.5)) / (k + 1e-9)

    return np.sign(signal) * rms * gain * makeup


def reverb(signal: np.ndarray, room_size: float = 0.4, wet: float = 0.25,
           pre_delay_ms: float = 10.0) -> np.ndarray:
    """Simple algorithmic reverb using FFT convolution (fast even for long signals)."""
    from scipy.signal import fftconvolve
    pre = int(pre_delay_ms * SR / 1000)
    # Keep IR short (4096 samples ≈ 93ms) — FFT conv is O((N+M)logN, but M stays small
    ir_len = min(4096, int(room_size * SR))
    decay = np.exp(-np.arange(ir_len) / (ir_len * 0.4))
    rng = np.random.default_rng(42)
    ir = decay * rng.standard_normal(ir_len) * 0.3
    ir[0] = 1.0

    wet_sig = fftconvolve(signal, ir, mode="full")[: len(signal)]
    if pre > 0:
        wet_sig = np.roll(wet_sig, pre)
        wet_sig[:pre] = 0
    return signal * (1 - wet) + wet_sig * wet


def ad_envelope(n: int, attack_ms: float, decay_ms: float,
                sustain_level: float = 0.0) -> np.ndarray:
    """Attack-Decay-Sustain envelope."""
    atk = int(attack_ms * SR / 1000)
    dec = int(decay_ms * SR / 1000)
    env = np.zeros(n)
    atk = min(atk, n)
    env[:atk] = np.linspace(0.0, 1.0, atk)
    dec_end = min(atk + dec, n)
    if dec_end > atk:
        env[atk:dec_end] = np.linspace(1.0, sustain_level, dec_end - atk)
    env[dec_end:] = sustain_level
    return env


# ---------------------------------------------------------------------------
# Drum Synthesizers
# ---------------------------------------------------------------------------

def synth_kick(n_total: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Kick: exponential pitch sweep + noise click + compression."""
    beat = int(SR * 60 / bpm)
    out = np.zeros(n_total)

    for i, start in enumerate(range(0, n_total, beat)):
        vel = rng.uniform(0.75, 1.0)
        hit_len = min(int(SR * 0.30), n_total - start)
        if hit_len < 100:
            break

        t = np.arange(hit_len) / SR
        f0 = rng.uniform(130, 180)
        f1 = rng.uniform(40, 60)
        tau = rng.uniform(0.06, 0.12)
        freq = f0 * np.exp(-t / tau) + f1
        sine = np.sin(2 * np.pi * np.cumsum(freq) / SR)

        # Transient click
        click_len = min(int(SR * 0.008), hit_len)
        click = rng.standard_normal(hit_len) * np.exp(-t / 0.003)
        click[click_len:] = 0

        env = ad_envelope(hit_len, 0.5, 150)
        hit = (sine * 0.8 + click * 0.3) * env * vel
        out[start: start + hit_len] += hit

    out = hpf(out, 30)
    out = peak_eq(out, 65,  +4.0)   # low punch
    out = peak_eq(out, 100, -3.0)   # reduce boxiness
    out = peak_eq(out, 3000, +2.0)  # click presence
    out = compress(out, threshold_db=-10, ratio=4.0, makeup_db=3.0)
    return normalize(out, -3)


def synth_snare(n_total: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Snare: noise body + ring tone + backbeat pattern."""
    beat = int(SR * 60 / bpm)
    out = np.zeros(n_total)

    for i, start in enumerate(range(0, n_total, beat)):
        if i % 4 not in (1, 3):  # beats 2 and 4
            continue
        vel = rng.uniform(0.7, 1.0)
        hit_len = min(int(SR * 0.20), n_total - start)
        if hit_len < 100:
            break

        # Body: bandpass noise
        noise = rng.standard_normal(hit_len)
        noise = bpf(noise, 200, 8000)

        # Tone ring
        f_body = rng.uniform(170, 220)
        t = np.arange(hit_len) / SR
        ring = np.sin(2 * np.pi * f_body * t)

        env = ad_envelope(hit_len, 0.5, rng.uniform(60, 120))
        hit = (noise * 0.7 + ring * 0.3) * env * vel
        out[start: start + hit_len] += hit

    out = hpf(out, 100)
    out = peak_eq(out, 200, +2.0)   # body
    out = peak_eq(out, 3000, +4.0)  # crack
    out = peak_eq(out, 8000, +2.0)  # air
    out = compress(out, threshold_db=-12, ratio=3.0, makeup_db=2.0)
    return normalize(out, -4)


def synth_toms(n_total: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Toms: tuned hits on off-beats, 3 pitches."""
    beat = int(SR * 60 / bpm)
    bar  = beat * 4
    out  = np.zeros(n_total)
    freqs = [rng.uniform(140, 170), rng.uniform(95, 115), rng.uniform(65, 80)]  # hi/mid/floor

    for bar_start in range(0, n_total, bar * 2):  # every other bar
        # Tom fill on beat 3+4
        for j, hit_pos in enumerate([beat * 2, beat * 2 + beat // 2, beat * 3]):
            start = bar_start + hit_pos
            if start >= n_total:
                break
            hit_len = min(int(SR * 0.35), n_total - start)
            if hit_len < 100:
                break
            f0 = freqs[j % 3]
            t = np.arange(hit_len) / SR
            freq = f0 * np.exp(-t / rng.uniform(0.08, 0.15)) + f0 * 0.4
            sine = np.sin(2 * np.pi * np.cumsum(freq) / SR)
            noise = rng.standard_normal(hit_len) * 0.15
            env = ad_envelope(hit_len, 1, rng.uniform(120, 220))
            hit = (sine + noise) * env * rng.uniform(0.7, 1.0)
            out[start: start + hit_len] += hit

    out = hpf(out, 50)
    out = peak_eq(out, 80, +3.0)
    out = compress(out, threshold_db=-12, ratio=2.0, makeup_db=1.0)
    return normalize(out, -5)


def synth_overheads(n_total: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Overheads: hi-hat on 8ths + crash on bar 1."""
    beat = int(SR * 60 / bpm)
    eighth = beat // 2
    bar = beat * 4
    out = np.zeros(n_total)

    # Hi-hats on every 8th note
    for start in range(0, n_total, eighth):
        is_open = (start // eighth) % 4 == 2  # open hat on 3rd 8th
        vel = rng.uniform(0.4, 0.8)
        hit_len = min(int(SR * (0.15 if is_open else 0.04)), n_total - start)
        if hit_len < 10:
            break
        noise = rng.standard_normal(hit_len)
        noise = hpf(noise, 6000)
        env = ad_envelope(hit_len, 0.3, int(1000 * hit_len / SR * 0.6))
        out[start: start + hit_len] += noise * env * vel

    # Crash on bar 1 of every 4 bars
    for bar_start in range(0, n_total, bar * 4):
        hit_len = min(int(SR * 1.5), n_total - bar_start)
        if hit_len < 1000:
            break
        noise = rng.standard_normal(hit_len)
        noise = hpf(noise, 2000)
        noise = peak_eq(noise, 5000, +5.0)
        env = ad_envelope(hit_len, 5, int(1000 * 1.2))
        out[bar_start: bar_start + hit_len] += noise * env * rng.uniform(0.6, 1.0)

    out = hpf(out, 60)
    out = peak_eq(out, 10000, +3.0)
    out = reverb(out, room_size=rng.uniform(0.3, 0.5), wet=rng.uniform(0.3, 0.5))
    return normalize(out, -5)


# ---------------------------------------------------------------------------
# Bleed Simulation
# ---------------------------------------------------------------------------

# Bleed matrix: how much (in dB) each source leaks into each mic
# Rows = source, Cols = destination mic [kick, snare, toms, overheads]
# Physical reasoning:
#   kick → snare (shared air): -22dB | kick → OH: -18dB (boom in room)
#   snare → kick: -28dB | snare → OH: -17dB
#   toms → OH: -20dB | toms → snare: -28dB
#   OH → all close mics: -30dB (background hiss)
BLEED_DB = np.array([
    # kick  snare  toms   OH
    [   0,  -22,   -28,  -18],  # kick source
    [ -28,    0,   -30,  -17],  # snare source
    [ -30,  -28,     0,  -20],  # toms source
    [ -32,  -30,   -28,    0],  # overheads source
])


def add_bleed(stems: dict[str, np.ndarray], rng: np.random.Generator,
              variation_db: float = 3.0) -> dict[str, np.ndarray]:
    """Add physically-motivated bleed between mics."""
    names = ["kick", "snare", "toms", "overheads"]
    out = {n: stems[n].copy() for n in names}

    for src_i, src_name in enumerate(names):
        src = stems[src_name]
        for dst_i, dst_name in enumerate(names):
            if src_i == dst_i:
                continue
            bleed_db = BLEED_DB[src_i, dst_i] + rng.uniform(-variation_db, variation_db)
            if bleed_db < -50:
                continue
            # Tiny delay (physical distance simulation: 0.5–3 ms)
            delay_samples = int(rng.uniform(0.5, 3.0) * SR / 1000)
            bleed_signal = np.roll(src, delay_samples)
            bleed_signal[:delay_samples] = 0
            out[dst_name] += bleed_signal * db(bleed_db)

    return out


# ---------------------------------------------------------------------------
# Main Generation Loop
# ---------------------------------------------------------------------------

def generate_take(take_dir: Path, bpm: float, rng: np.random.Generator,
                  duration_sec: float = 10.0, add_bleed_flag: bool = True):
    take_dir.mkdir(parents=True, exist_ok=True)
    n = int(SR * duration_sec)

    clean = {
        "kick":      synth_kick(n, bpm, rng),
        "snare":     synth_snare(n, bpm, rng),
        "toms":      synth_toms(n, bpm, rng),
        "overheads": synth_overheads(n, bpm, rng),
    }

    if add_bleed_flag:
        mics = add_bleed(clean, rng)
    else:
        mics = clean

    for stem_name, signal in mics.items():
        # Write as 24-bit stereo WAV (slight L/R decorrelation for realism)
        jitter = rng.standard_normal(n) * 1e-4
        stereo = np.stack([signal + jitter, signal - jitter], axis=1).astype(np.float32)
        sf.write(str(take_dir / f"{stem_name}.wav"), stereo, SR, subtype="PCM_24")

    print(f"  [done] {take_dir.name}  (BPM={bpm:.1f})")


def main(args):
    out = Path(args.output_dir)
    print(f"Generating {args.num_takes} debug takes -> {out}")
    print(f"  Bleed: {'ON' if not args.no_bleed else 'OFF'}  "
          f"Duration: {args.duration}s  SR: {SR}Hz\n")

    rng = np.random.default_rng(args.seed)
    bpms = rng.uniform(85, 130, size=args.num_takes)

    for i in range(args.num_takes):
        take_dir = out / f"Debug_Take_{i+1:02d}"
        generate_take(take_dir, bpms[i], rng,
                      duration_sec=args.duration,
                      add_bleed_flag=not args.no_bleed)

    print(f"\nDone. {args.num_takes} takes written to {out}")
    print(f"Run overfit test:")
    print(f"  python python/training/train.py --data_dir {out} --overfit_one "
          f"--epochs 200 --batch_size 2 --depth 3 --channels 32 "
          f"--chunk_sec 0.5 --loss hybrid --export_epochs 5,10,20,50,100,200")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic drum dataset")
    parser.add_argument("--output_dir", type=str, default="data/debug")
    parser.add_argument("--num_takes",  type=int, default=15)
    parser.add_argument("--duration",   type=float, default=10.0,
                        help="Duration per take in seconds")
    parser.add_argument("--no_bleed",   action="store_true",
                        help="Disable bleed simulation (clean stems for upper-bound test)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    main(args)
