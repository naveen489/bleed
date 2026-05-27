# Evaluation Report: Run 001 - Sanity Check (Synthetic Data)

* **Date:** 2026-05-27
* **Experiment ID:** run_001_sanity_check
* **Model Configuration:** Depth: 3, Channels: 32, Chunk Size: 0.5s, Loss: Hybrid (L1 + MRSTFT)
* **Dataset:** 15-take synthetic dataset with artificial bleed

---

## 1. Objectives
* Establish that the training pipeline runs end-to-end without crashing.
* Test checkpoint saving, loss logging, and baseline separation on a small synthetic setup.

## 2. Experimental Setup
* **Command:**
  ```bash
  python python/training/train.py --data_dir data/debug --overfit_one --epochs 200 --batch_size 2 --depth 3 --channels 32 --chunk_sec 0.5 --loss hybrid --output_dir output/checkpoints
  ```
* **Optimizer:** Adam with dynamic plateau learning rate reduction scheduler.

## 3. Results & Metrics

| Stem | Mean SI-SDR | Transients | Verdict |
|---|---|---|---|
| Kick | **+5.6 dB** | ✅ Preserved | Learned low-frequency separation; target dominant. |
| Snare | **-3.8 dB** | ✅ Mostly OK | Failed; anti-correlated/wrong polarity, severe bleed. |
| Toms | **-8.6 dB** | ✅ Preserved | Failed; spectrally overlapped with cymbals/snare. |
| Overheads | **-3.4 dB** | ✅ Preserved | Severe leakage from low-frequency drum hits. |
| **Overall** | **-2.6 dB** | — | **Unusable in a mix** |

## 4. Key Takeaways
1. **LR Scheduler issue:** The `ReduceLROnPlateau` scheduler collapsed the learning rate too early because the batch-level training losses were highly volatile, causing the scheduler to think the model had plateaud.
2. **Chunk Size issue:** A 0.5-second chunk is too short to capture the rhythmic relationship and temporal envelopes of hits, leading to poor signal context for the BiLSTM bottleneck.
