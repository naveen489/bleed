# Metrics Reference

Definitions for every metric reported in BleedDemucs evaluation runs.

---

## SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

**Formula**

```
SI-SDR = 10 * log10( ||proj||² / ||noise||² )

where:
  proj  = (<estimate, target> / ||target||²) * target   (projection of estimate onto target)
  noise = estimate - proj                                (residual / distortion)
```

Both signals are zero-mean before computation (removes DC offset).

**Interpretation**

| SI-SDR | Perceptual Quality |
|---|---|
| > 20 dB | Near-perfect separation |
| 10–20 dB | Usable in a mix with some cleanup |
| 5–10 dB | Target dominant but heavy bleed |
| 0–5 dB | Weak separation, mostly bleed |
| < 0 dB | Output anticorrelated with target — model failed |

**Scale invariance** means the metric is insensitive to overall gain — a stem output that
is 2× too loud still scores the same as one at the correct level.

**Implementation:** `python/training/losses.py → si_sdr()`

---

## Dominance Rank

For each stem output, compute SI-SDR against *all four* ground-truth stems.
Rank the stem that scores highest as rank 1.

- **Rank 1 (PASS):** The target element is the dominant content in the output.
- **Rank 2–4 (FAIL):** A different stem is more correlated with the output than the target.

A good model should produce rank 1 for every output stem.

---

## Crest Factor (Transient Preservation)

```
Crest Factor = 20 * log10( peak / RMS )   [dB]
```

High crest factor = signal is transient-heavy (many short peaks, low average energy).
Low crest factor = signal is more sustained / compressed.

**Comparison:** `CF_pred - CF_ground_truth`

- **Pass:** diff > -3 dB — transients are at least as sharp as ground truth
- **Fail:** diff < -3 dB — transients are being smeared/smoothed by the model

Drum transients (kick attack, snare crack) should have crest factors of +12 to +22 dB.
If the model output is below +9 dB, the "punch" of the drum is gone.

---

## Spectral Flatness

```
Spectral Flatness = 20 * log10( geometric_mean(|X(f)|) / arithmetic_mean(|X(f)|) )
```

- Close to 0 dB → noise-like (white noise = 0 dB)
- Very negative (< -10 dB) → highly tonal / harmonic

Used to detect **metallic or phasey artifacts**: if spectral flatness is anomalously
high (less negative) compared to the ground truth, the output contains unstructured
noise that wasn't in the original — a sign of model hallucination.

---

## Bleed Energy (dB)

For each output stem, estimate how much signal from *other* stems leaked in:

```
bleed(src → dst) = 20 * log10( |<pred_dst, gt_src>| / ||gt_src|| )
```

**Interpretation**

| Bleed Level | dB Range | Meaning |
|---|---|---|
| Clean | < -30 dB | Negligible bleed |
| Light | -30 to -20 dB | Audible only at high gain |
| Moderate | -20 to -12 dB | Noticeable, may affect mix |
| Heavy | > -12 dB | Severely degraded separation |

> **Note:** A model that outputs the raw mixture (no separation) will show HEAVY bleed
> everywhere. Use this as a baseline — all models start here at epoch 0.

---

## Loss Functions

### SI-SDR Loss
`loss = -mean_SI-SDR` across all stems and batch items.
Minimising this is equivalent to maximising SI-SDR. Goes negative as model improves.

### L1 Loss
`loss = mean(|pred - target|)` per stem. Simple, stable, but phase-sensitive.

### MSE Loss
`loss = mean((pred - target)²)` per stem. Penalises large errors quadratically.
Tends to produce over-smoothed outputs — bad for drum transients.

### MRSTFT Loss (Multi-Resolution STFT)
Computed at three resolutions:

| Window | Hop | Resolves |
|---|---|---|
| 256 samples (5.8 ms) | 64 | Fast transients (kick/snare attack) |
| 1024 samples (23 ms) | 256 | Mid-range content, body |
| 2048 samples (46 ms) | 512 | Low frequency, room tail |

Per resolution: `spectral_convergence + log_magnitude_L1`

- **Spectral convergence:** `||T - P||_F / ||T||_F` — shape mismatch across all bins
- **Log magnitude L1:** Perceptually weighted — errors at all frequencies count equally

> **Stability note:** `||T||_F < 1e-4` (silent chunk) skips spectral convergence to
> prevent division-by-near-zero explosions. Found in run_003.

### Hybrid Loss
`loss = 0.5 * MRSTFT + 0.5 * SI-SDR_loss`

Combines spectral shape accuracy (MRSTFT) with waveform-domain alignment (SI-SDR).
Currently the recommended loss for drum stem separation.
