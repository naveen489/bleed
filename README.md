# Bleed (Active Work in Progress 🚧)

AI-powered drum multitrack stem separation and bleed attenuation.

This project implements a deep learning source separation pipeline (based on the Demucs architecture) to isolate a drum mixture into four distinct stems: **Kick**, **Snare**, **Toms**, and **Cymbals/Overheads**. This enables post-recording balance adjustments, bleed cleanup, and isolated stem processing.

---

## 🚀 Key Features

* **Advanced Demucs Architecture:** Deep convolutional encoder-decoder network with BiLSTM bottlenecks and custom skip-connections for high-fidelity waveform separation.
* **Hybrid Loss Function:** Optimized using a combination of L1 Time-Domain Loss and Multi-Resolution Short-Time Fourier Transform (MRSTFT) frequency-domain loss with strict stability guards for near-silent regions.
* **Robust Diagnostic & Evaluation Framework:** Automated script to calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), SDR, SIR, SAR, and Crest Factor (Transient preservation).
* **Synthetic & Real-World Dataset Generation:** Pipeline to construct controllable synthetic mixes with customizable bleed, and batch utilities to process real-world multitracks.
* **Hybrid DSP/Plugin Ready:** Structured with a C++ JUCE plugin harness in `plugin/` for real-time inference using exported ONNX models.

---

## 📂 Project Structure

```
bleed/
├── python/
│   ├── training/          # PyTorch models, hybrid losses, and training loops
│   ├── scripts/           # Dataset generators, evaluations, and ONNX exporters
│   └── requirements.txt   # Python dependency specification
├── tests/
│   ├── README.md          # Overview of test suite & diagnostic commands
│   ├── metrics_reference.md # Complete reference of SDR, SI-SDR, etc.
│   ├── diagnostic_findings.md # Analysis of architectural performance
│   └── evaluation_reports/ # Detailed logs of major experimental runs
├── plugin/                # C++ JUCE real-time inference plugin codebase
├── research/              # Jupyter notebooks and exploratory analysis
└── models/                # Saved checkpoints and production-ready ONNX models
```

---

## 🛠️ Quick Start

### 1. Installation
Ensure you have Python 3.10+ installed. Navigate to the root directory and install dependencies:
```bash
pip install -r python/requirements.txt
```

### 2. Generate a Synthetic Diagnostic Dataset
To test if the pipeline is running correctly on your environment:
```bash
python python/scripts/generate_debug_dataset.py \
    --output_dir data/debug \
    --num_takes 15 \
    --duration 10
```

### 3. Run Overfitting Test
To verify the network capacity and guarantee that the model can learn separation:
```bash
python python/training/train.py \
    --data_dir data/debug \
    --overfit_one \
    --epochs 200 \
    --depth 4 \
    --channels 48 \
    --chunk_sec 2.0 \
    --loss hybrid \
    --output_dir output/checkpoints
```

### 4. Evaluate Checkpoint Performance
Once training is complete, compute standard source separation metrics:
```bash
python python/scripts/evaluate.py \
    --checkpoint models/checkpoints/final.pt \
    --data_dir data/debug \
    --chunk_sec 2.0
```

---

## 📊 Latest Diagnostic Benchmark

Our best diagnostic run using a **4-layer / 48-channel** model on our synthetic bleed dataset yielded the following results (200 epochs):

| Stem | Mean SI-SDR | Transients | Verdict |
|---|---|---|---|
| **Kick** | **+7.4 dB** | ✅ Preserved | Strong isolation, punchy and intact |
| **Snare** | **+0.4 dB** | ✅ Preserved | Heavy bleed remaining |
| **Toms** | **-2.4 dB** | ✅ Preserved | Notable leakage from cymbals |
| **Overheads** | **-0.5 dB** | ✅ Preserved | Intact highs, minor low-end bleed |
| **Overall** | **+1.2 dB** | — | **Progressing Well** |

For deep mathematical definitions of our metrics and a breakdown of performance findings, consult the [Tests Directory](file:///c:/Users/Naveen/Documents/Code/bleed/tests/README.md).

---

## 🎯 Next Milestone Roadmap

1. **Integrate Real-World Data:** Prepare and run training on the *Cambridge Multitrack Dataset* to expose the network to realistic acoustic phase changes.
2. **Increase Architectural Capacity:** Expand to a 5/6-layer depth with 64–96 channels or explore pre-trained Demucs v4 fine-tuning.
3. **Real-time JUCE Engine:** Build out the C++ plug-in processor in `plugin/` to load ONNX runs dynamically inside the DAW.
