# Evaluation Report: Run 002 - Clean Dataset Stability Test

* **Date:** 2026-05-27
* **Experiment ID:** run_002_hybrid_clean_2.0s
* **Model Configuration:** Depth: 4, Channels: 48, Chunk Size: 2.0s, Loss: Hybrid (L1 + MRSTFT)
* **Dataset:** `data/debug_clean` (synthetic stems generated with **zero bleed**)

---

## 1. Objectives
* Intentionally overfit the model on clean (no-bleed) drum tracks.
* Verify if the architecture can easily memorize a perfect dataset when bleed is absent.

## 2. Experimental Setup
* **Command:**
  ```bash
  python python/training/train.py --data_dir data/debug_clean --overfit_one --epochs 200 --batch_size 2 --depth 4 --channels 48 --chunk_sec 2.0 --loss hybrid --output_dir output/clean_checkpoints
  ```

## 3. Results & Numerical Instability
During training, the loss values started behaving erratically, exploding on alternating epochs:
* **Epoch 50:** Loss ~ 4.2
* **Epoch 51:** Loss = 19,205.12 (Catastrophic spike!)
* **Epoch 52:** Loss ~ 3.8
* **Epoch 53:** Loss = 9,454.21

This behavior indicated a severe numerical instability in backpropagation.

## 4. Root Cause Analysis
In the clean dataset, several drum tracks (particularly Toms and Snare) contain long durations of total silence (zero amplitude) when those instruments are not being struck.

The Multi-Resolution Short-Time Fourier Transform (MRSTFT) loss has a spectral convergence term:
$$L_{sc}(x, y) = \frac{|| |STFT(x)| - |STFT(y)| ||_F}{|| |STFT(y)| ||_F}$$

When the target stem $y$ is silent or near-silent:
1. The denominator $|| |STFT(y)| ||_F \approx 0$.
2. This division by zero causes the loss value and its gradients to explode to extreme values (or `NaN`).

## 5. Resolution
We modified `python/training/losses.py` to add an energy-guard threshold:
```python
# Skip the spectral convergence term if the target is near-silent
if target_norm < 1e-4:
    sc_loss = torch.zeros_like(l1_loss)
else:
    sc_loss = FrobeniusNorm(target_spec - pred_spec) / target_norm
```
This guarantees numerical stability even when training on clean multitracks containing sparse audio events.
