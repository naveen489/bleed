# BleedDemucs — Test & Evaluation Framework

This directory contains all evaluation runs, diagnostic findings, and metric definitions
for the BleedDemucs drum stem separation project.

## Structure

```
tests/
├── README.md                          # This file
├── metrics_reference.md               # Definition of every metric used
├── diagnostic_findings.md             # Synthesised findings across all runs
└── evaluation_reports/
    ├── run_001_sanity_check.md         # Pipeline smoke test (synthetic data)
    ├── run_002_hybrid_bleed_0.5s.md    # First diagnostic overfit (bleed, 0.5s chunks)
    ├── run_003_hybrid_clean_2.0s.md    # Clean data stability test (failed — MRSTFT explosion)
    └── run_004_hybrid_bleed_2.0s.md    # Best run to date (bleed, 2.0s chunks, cosine LR)
```

## How to reproduce any evaluation

```bash
# 1. Generate the debug dataset (with bleed)
python python/scripts/generate_debug_dataset.py \
    --output_dir data/debug --num_takes 15 --duration 10

# 2. Train
python python/training/train.py \
    --data_dir data/debug --overfit_one \
    --epochs 200 --depth 4 --channels 48 \
    --chunk_sec 2.0 --loss hybrid \
    --export_epochs 5,10,20,50,100,200 \
    --output_dir output/checkpoints

# 3. Evaluate
python python/scripts/evaluate.py \
    --checkpoint models/checkpoints/final.pt \
    --data_dir data/debug \
    --chunk_sec 2.0 --max_takes 5
```

## Current best result

| Stem      | SI-SDR   | Transients | Verdict      |
|-----------|----------|------------|--------------|
| Kick      | +7.4 dB  | Preserved  | IMPROVING    |
| Snare     | +0.4 dB  | Preserved  | NEEDS WORK   |
| Toms      | -2.4 dB  | Preserved  | NEEDS WORK   |
| Overheads | -0.5 dB  | Preserved  | NEEDS WORK   |
| **Overall** | **+1.2 dB** | — | **IMPROVING** |

> Target for production quality: SI-SDR > 10 dB on all stems.
