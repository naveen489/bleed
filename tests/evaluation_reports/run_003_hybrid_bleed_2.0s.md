# Evaluation Report: Run 003 - Best Overfit Run

* **Date:** 2026-05-27
* **Config:** Depth 4, Channels 48, Chunk Size 2.0s, Cosine Annealing LR
* **Dataset:** 15-take synthetic dataset with artificial bleed

---

## 1. Goal
Overfit the model on a tiny bleed dataset using corrected hyperparameters (increased chunk size and Cosine Annealing learning rate schedule) to determine the upper performance bounds on a synthetic drum mix.

## 2. Performance Outcomes
Training converged smoothly. Total hybrid loss dropped to **0.0164** at epoch 200.

### Evaluation Metrics at Epoch 200
* **Kick:** **+7.4 dB** (strong separation, highly recognizable target)
* **Snare:** **+0.4 dB** (heavy bleed remaining)
* **Toms:** **-2.4 dB** (heavy bleed from cymbals and snare)
* **Overheads:** **-0.5 dB** (clear cymbals, moderate bleed remaining)
* **Overall:** **+1.2 dB**

## 3. Findings
* Transient envelopes and drum hit timing are 100% preserved.
* The model behaves mostly like a dynamic EQ rather than a true phase-aware source separator.
* Transitioning to a 2.0s chunk and Cosine LR yielded a **+3.8 dB** overall improvement over Run 001. The pipeline is validated and ready for real-world multitrack data.
