# Diagnostic Findings & Recommendations

This document synthesises the findings from all diagnostic runs conducted on the BleedDemucs models (up to Epoch 200 of the overfit runs).

---

## 1. Executive Summary

Our diagnostic runs have verified that the baseline Demucs model (4-layer, 48 channels) with a hybrid L1/MRSTFT loss can successfully overfit a small dataset, but its rate of convergence and final metrics reveal important bottlenecks.

* **Key Breakthrough:** Changing the chunk size from `0.5s` to `2.0s` and switching to a `CosineAnnealingLR` scheduler improved the overall SI-SDR from **-2.6 dB** to **+1.2 dB** (a `+3.8 dB` improvement).
* **Major Instability Resolved:** Near-silent stems in the clean dataset caused the Multi-Resolution Short-Time Fourier Transform (MRSTFT) loss's spectral convergence term to explode due to division by zero. We have implemented an energy guard in the denominator ($||T||_F < 1e-4$) to prevent this instability.
* **Underperforming Stems:** While Kick separation has improved to **+7.4 dB**, Snare (+0.4 dB), Toms (-2.4 dB), and Overheads (-0.5 dB) are lagging behind. The spectral similarity between snare, toms, and overheads makes linear-like filters or shallow convolutional paths prone to leakage and phase cancellation.

---

## 2. Detailed Run Analysis

### Run 1: Sanity Check (Synthetic Data, 0.5s chunks, old LR scheduler)
* **Goal:** Verify that the training pipeline runs and saves checkpoints.
* **Loss:** Hybrid (L1 + MRSTFT)
* **Findings:**
  - The model runs, but the learning rate decayed too rapidly because the plateau scheduler was triggered by noisy batch-level losses.
  - Overall SI-SDR: **-2.6 dB**. Only Kick showed reasonable learning (+5.6 dB) because it occupies unique low frequencies.

### Run 2: Clean Dataset Stability Test (No-Bleed, 2.0s chunks)
* **Goal:** Intentionally overfit the model on clean (no-bleed) stems.
* **Findings:**
  - Catastrophic loss explosion on alternating epochs (loss values spiked above `19,000`).
  - **Root Cause:** In the clean dataset, several drum tracks (like Toms) contain long stretches of absolute silence. In the MRSTFT spectral convergence term:
    $$\frac{|| |STFT(x)| - |STFT(y)| ||_F}{|| |STFT(y)| ||_F}$$
    the denominator ($|| |STFT(y)| ||_F$) approaches zero, blowing up the gradient.
  - **Remedy:** Patched `python/training/losses.py` with an epsilon energy guard.

### Run 3: Best Overfit Run (Bleed Dataset, 2.0s chunks, Cosine LR, depth=4 ch=48)
* **Goal:** Overfit on a 15-take synthetic bleed dataset.
* **Findings:**
  - Loss smoothly decreased from `~32.0` down to `+0.0164` by epoch 200.
  - Overall SI-SDR reached **+1.2 dB** (Kick: +7.4 dB, Snare: +0.4 dB, Toms: -2.4 dB, Overheads: -0.5 dB).
  - Transients and rhythms are fully intact, but significant bleed remains in the Snare and Toms outputs.

---

## 3. Recommended Next Steps

1. **Move to Real Data:**
   Use the Cambridge Multitrack Dataset prep script to obtain real-world stems. Real bleed characteristics are far more complex than simple synthetic mixes, and our model needs realistic phase/frequency distributions to generalise.
2. **Increase Model Capacity:**
   Increase the Demucs architecture size (e.g., depth=5 or 6, channels=64) or explore HTDemucs (Hybrid Transformer Demucs) to better capture wideband drum mic correlations.
3. **Fine-tuning Approach:**
   Rather than training a complex multi-stem separator entirely from scratch on limited data, explore starting from a pre-trained music demixing checkpoint (e.g. Demucs v4) and fine-tuning on the drum stem dataset.
