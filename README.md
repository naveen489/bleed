# Bleed

AI drum rebalancing from single mic recordings.

## Project Structure

- `python/`: PyTorch research, training, and ONNX export.
- `plugin/`: C++ JUCE plugin codebase.
- `research/`: Jupyter notebooks and exploratory data analysis.
- `models/`: Exported ONNX models.
- `tests/`: Unit and integration tests.

## Goal

Separate a drum mix into Kick, Snare, Toms, and Cymbals/Overheads to allow users to rebalance volumes post-recording.
