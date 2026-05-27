"""
Per-stem diagnostic evaluation for BleedDemucs.

Runs inference on a dataset and answers the key listening questions:

  - Is the target element dominant?       [SI-SDR vs ground truth]
  - Are transients preserved?             [peak/RMS ratio vs ground truth]
  - Is there cross-stem bleed?            [energy of leaked content]
  - Are there spectral artifacts?         [spectral flatness comparison]
  - Would this be usable in a mix?        [SI-SDR > 10dB threshold]

Usage:
    python python/scripts/evaluate.py \\
        --checkpoint models/checkpoints/final.pt \\
        --data_dir   data/debug \\
        --chunk_sec  0.5
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
from python.training.losses import si_sdr as compute_si_sdr


# ---------------------------------------------------------------------------
# Audio I/O (minimal, no torchaudio dependency)
# ---------------------------------------------------------------------------

def load_stem(path: Path, sr: int = 44100) -> torch.Tensor:
    """Load WAV → (2, T) float32 tensor."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]
    if file_sr != sr:
        wav = F.interpolate(wav.unsqueeze(0),
                            size=int(wav.shape[-1] * sr / file_sr),
                            mode="linear", align_corners=False).squeeze(0)
    return wav


def load_mixture(take_dir: Path, sr: int = 44100) -> tuple[torch.Tensor, dict]:
    """Load and sum all stems; return mixture + individual stems."""
    stems = {}
    mixture = None
    for s in STEMS:
        p = take_dir / f"{s}.wav"
        if not p.exists():
            continue
        wav = load_stem(p, sr)
        stems[s] = wav
        mixture = wav.clone() if mixture is None else mixture + wav

    if mixture is not None:
        mx = mixture.abs().max()
        if mx > 1.0:
            scale = 1 / mx
            mixture = mixture * scale
            stems = {k: v * scale for k, v in stems.items()}

    return mixture, stems


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def chunk_infer(model: BleedDemucs, mixture: torch.Tensor, device: torch.device,
                chunk_sec: float = 0.5, sr: int = 44100,
                overlap: float = 0.25) -> dict:
    chunk_len = int(chunk_sec * sr)
    hop_len   = int(chunk_len * (1 - overlap))
    T         = mixture.shape[-1]

    out   = {s: torch.zeros(2, T) for s in STEMS}
    wt    = torch.zeros(T)
    window = torch.hann_window(chunk_len)

    pos = 0
    model.eval()
    with torch.no_grad():
        while pos < T:
            end  = min(pos + chunk_len, T)
            chunk = mixture[:, pos:end]
            if chunk.shape[-1] < chunk_len:
                chunk = F.pad(chunk, (0, chunk_len - chunk.shape[-1]))

            mono = chunk.mean(0, keepdim=True).unsqueeze(0).to(device)
            preds = model(mono)

            w = window[:end - pos]
            for s in STEMS:
                out[s][:, pos:end] += preds[s][0, :, :end - pos].cpu() * w
            wt[pos:end] += w
            pos += hop_len

    wt = wt.clamp(min=1e-8)
    return {s: out[s] / wt.unsqueeze(0) for s in STEMS}


# ---------------------------------------------------------------------------
# Diagnostic Metrics
# ---------------------------------------------------------------------------

def crest_factor(wav: torch.Tensor) -> float:
    """Peak-to-RMS ratio in dB (higher = more transient-like)."""
    rms  = wav.pow(2).mean().sqrt().item()
    peak = wav.abs().max().item()
    if rms < 1e-8:
        return 0.0
    return 20 * np.log10(peak / rms)


def spectral_flatness(wav: torch.Tensor) -> float:
    """Spectral flatness in dB. 0 = tonal, higher = noise-like."""
    mono = wav.mean(0).numpy()
    spec = np.abs(np.fft.rfft(mono))
    spec = spec[spec > 1e-10]
    if len(spec) == 0:
        return 0.0
    geo  = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    return float(20 * np.log10(geo / arith + 1e-10))


def bleed_energy_db(pred: torch.Tensor, ground_truths: dict,
                    target_stem: str) -> dict:
    """
    Estimate how much energy from non-target stems leaked into the prediction.
    Returns dict: stem → dB of that stem's energy in the prediction.
    """
    result = {}
    pred_mono = pred.mean(0).float()
    for s, gt in ground_truths.items():
        if s == target_stem:
            continue
        gt_mono = gt.mean(0).float()
        n = min(pred_mono.shape[-1], gt_mono.shape[-1])
        overlap = (pred_mono[:n] * gt_mono[:n]).sum()
        gt_energy = (gt_mono[:n] ** 2).sum().sqrt()
        if gt_energy < 1e-8:
            result[s] = -99.0
        else:
            frac = (overlap.abs() / gt_energy).item()
            result[s] = float(20 * np.log10(frac + 1e-10))
    return result


def dominance_rank(stem: str, predictions: dict,
                   ground_truths: dict) -> tuple[float, int]:
    """
    Returns (SI-SDR of target stem, rank among all stems by SI-SDR).
    Rank 1 = dominant (best); rank 4 = worst. Ideal: rank 1.
    """
    scores = {}
    for s in STEMS:
        if s not in predictions or s not in ground_truths:
            continue
        p = predictions[s].unsqueeze(0)
        t = ground_truths[s].unsqueeze(0)
        scores[s] = compute_si_sdr(p, t).mean().item()

    target_score = scores.get(stem, -99)
    rank = sorted(scores.values(), reverse=True).index(target_score) + 1
    return target_score, rank


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

USABLE_THRESHOLD_DB = 10.0   # SI-SDR > 10dB = potentially usable
DOMINANT_THRESHOLD  = 1      # rank 1 = target is loudest/cleanest stem

def fmt(val: float, unit: str = "dB", width: int = 7) -> str:
    return f"{val:+{width}.1f} {unit}"

def pass_fail(condition: bool) -> str:
    return "[PASS]" if condition else "[FAIL]"

def bleed_label(db_val: float) -> str:
    if db_val > -12:  return "HEAVY"
    if db_val > -20:  return "moderate"
    if db_val > -30:  return "light"
    return "clean"


def print_stem_report(stem: str, pred: torch.Tensor, gt: torch.Tensor,
                      all_preds: dict, all_gts: dict):
    sdr, rank = dominance_rank(stem, all_preds, all_gts)
    cf_pred = crest_factor(pred)
    cf_gt   = crest_factor(gt)
    cf_diff = cf_pred - cf_gt
    sf_pred = spectral_flatness(pred)
    bleed   = bleed_energy_db(pred, all_gts, stem)

    usable    = sdr > USABLE_THRESHOLD_DB
    dominant  = rank <= DOMINANT_THRESHOLD
    transient = cf_diff > -3.0   # within 3dB of ground truth crest factor

    print(f"\n  ---- {stem.upper()} ----")
    print(f"  SI-SDR          : {fmt(sdr)}  {pass_fail(usable)} (>{USABLE_THRESHOLD_DB:.0f}dB = usable)")
    print(f"  Dominance rank  : {rank}/4      {pass_fail(dominant)} (rank 1 = target is dominant)")
    print(f"  Crest factor    : pred={cf_pred:+.1f}dB  gt={cf_gt:+.1f}dB  diff={cf_diff:+.1f}dB  "
          f"{pass_fail(transient)} (transients {'preserved' if transient else 'WEAK'})")
    print(f"  Spectral flat.  : {sf_pred:+.1f} dB (lower = more tonal)")
    print(f"  Bleed:")
    for src, b in bleed.items():
        print(f"    <- {src:<12}: {b:+.1f} dB  ({bleed_label(b)})")


def diagnose(sdr_by_stem: dict) -> list[str]:
    """
    Based on SI-SDR results, emit a diagnosis string.
    Maps observations to likely root causes.
    """
    findings = []
    avg_sdr = np.mean(list(sdr_by_stem.values()))

    if avg_sdr < -5:
        findings.append("DIAGNOSIS: All stems very blurry (avg SI-SDR < -5dB). "
                        "Likely a loss function issue — try 'hybrid' or 'mrstft'.")
    elif avg_sdr < 5:
        findings.append("DIAGNOSIS: Separation exists but weak (avg SI-SDR 0-5dB). "
                        "Model hasn't converged — train longer or increase model size.")
    else:
        findings.append("DIAGNOSIS: Reasonable separation (avg SI-SDR > 5dB). "
                        "Focus on bleed and transient detail next.")

    oh_sdr = sdr_by_stem.get("overheads", 0)
    kick_sdr = sdr_by_stem.get("kick", 0)
    if oh_sdr > kick_sdr + 6:
        findings.append("WARNING: Cymbal/overhead dominance — high frequencies masking "
                        "kick/snare. Try frequency-weighted MRSTFT or pre-emphasis EQ.")

    cf_issues = [s for s, v in sdr_by_stem.items() if v < 0]
    if cf_issues:
        findings.append(f"NOTE: Negative SI-SDR on {cf_issues} — output anticorrelated with "
                        "target. Possible phase inversion or architecture issue.")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)

    model = BleedDemucs(in_channels=1, out_channels=2, num_stems=4,
                        depth=ckpt.get("depth", 3),
                        channels=ckpt.get("channels", 16))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    data_dir = Path(args.data_dir)
    takes = sorted([d for d in data_dir.iterdir()
                    if d.is_dir() and all((d / f"{s}.wav").exists() for s in STEMS)])

    if not takes:
        print(f"ERROR: No valid takes in {data_dir}")
        return

    takes = takes[:args.max_takes]
    print(f"\nBleedDemucs Diagnostic Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Dataset    : {data_dir}  ({len(takes)} takes)")
    print(f"  Device     : {device}")
    print("=" * 60)

    all_sdrs: dict[str, list] = {s: [] for s in STEMS}

    for take in takes:
        print(f"\nTake: {take.name}")
        mixture, gt_stems = load_mixture(take, args.sample_rate)
        if mixture is None:
            print("  (skipped — missing stems)")
            continue

        preds = chunk_infer(model, mixture, device,
                            chunk_sec=args.chunk_sec, sr=args.sample_rate)

        for s in STEMS:
            if s not in gt_stems:
                continue
            sdr = compute_si_sdr(preds[s].unsqueeze(0), gt_stems[s].unsqueeze(0)).mean().item()
            all_sdrs[s].append(sdr)
            print_stem_report(s, preds[s], gt_stems[s], preds, gt_stems)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean SI-SDR across all takes)")
    mean_sdrs = {}
    for s in STEMS:
        if all_sdrs[s]:
            m = float(np.mean(all_sdrs[s]))
            mean_sdrs[s] = m
            usable = m > USABLE_THRESHOLD_DB
            print(f"  {s:<14}: {m:+.1f} dB  {pass_fail(usable)}")

    print()
    for finding in diagnose(mean_sdrs):
        print(f"  {finding}")

    overall = float(np.mean(list(mean_sdrs.values()))) if mean_sdrs else -99
    print(f"\n  Overall mean SI-SDR : {overall:+.1f} dB")
    verdict = ("VIABLE" if overall > 10
               else "IMPROVING" if overall > 0
               else "NEEDS WORK")
    print(f"  Pipeline verdict    : {verdict}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BleedDemucs stem quality evaluator")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--data_dir",     type=str, required=True)
    parser.add_argument("--sample_rate",  type=int, default=44100)
    parser.add_argument("--chunk_sec",    type=float, default=0.5)
    parser.add_argument("--max_takes",    type=int, default=5,
                        help="Max takes to evaluate (keep small for speed)")
    args = parser.parse_args()
    main(args)
