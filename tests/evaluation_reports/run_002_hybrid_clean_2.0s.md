# Evaluation Report: Run 002 - Clean Dataset Stability Test

* **Date:** 2026-05-27
* **Config:** Depth 4, Channels 48, Chunk Size 2.0s, Hybrid Loss
* **Dataset:** Clean (zero-bleed) synthetic stems

---

## 1. Goal
Overfit the model on clean (no-bleed) drum tracks to verify if the architecture easily memorizes perfect dataset tracks.

## 2. Instability & Cause
During training, the loss values exploded erratically (spiking above 19,000 on alternating epochs).
* **Issue:** Several drum tracks (like Toms) have long stretches of complete silence.
* **Root Cause:** In the Multi-Resolution STFT spectral convergence loss, silence in the target stem causes the divisor (denominator) to approach zero, resulting in exploding gradients.

## 3. Resolution
We added an energy-guard threshold in `python/training/losses.py`. The spectral convergence term is now skipped when the target stem is near-silent ($||T||_F < 1e-4$), resolving the instability completely.
