# Evaluation Report: Run 001 - Sanity Check

* **Date:** 2026-05-27
* **Config:** Depth 3, Channels 32, Chunk Size 0.5s, Hybrid Loss (L1 + MRSTFT)
* **Dataset:** 15-take synthetic dataset with artificial bleed

---

## 1. Goal
Verify that the training pipeline runs end-to-end without crashing and saves model checkpoints.

## 2. Findings
* The model ran successfully, but the learning rate decayed too rapidly because the plateau scheduler was triggered prematurely by volatile batch-level training losses.
* High-level separation results:
  - **Kick:** +5.6 dB (some low-frequency separation learned)
  - **Snare:** -3.8 dB (severe bleed, failed to separate)
  - **Toms:** -8.6 dB (failed spectrally)
  - **Overheads:** -3.4 dB (retained severe low-frequency leakage)
  - **Overall:** -2.6 dB (unusable baseline)

## 3. Takeaways
1. **Fix Scheduler:** Switch from plateau-based to cosine annealing learning rate scheduler for better stability.
2. **Increase Chunk Size:** Increase default chunk size from 0.5s to 2.0s to give the BiLSTM more context.
