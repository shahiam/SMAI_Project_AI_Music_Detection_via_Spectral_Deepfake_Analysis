# AI-Generated Music Detection via Spectral Deepfake Analysis

Project proposal implementation for SMAI, Team Sarcosuchus.

## Mid-evaluation status

This repository now contains a reproducible two-branch baseline:

1. **Classical branch (30%)**: MFCCs and spectral centroid features with an SVM.
2. **Modern branch (70%)**: log-Mel spectrogram images and a transfer-learned ResNet-18.

Both branches use the same manifest and patient/group-aware train/validation/test split. The
evaluation report contains precision, recall, F1, ROC-AUC, and a confusion matrix. The modern
branch also includes Grad-CAM for localizing suspicious time-frequency regions.

The code is intentionally runnable on a small synthetic dataset for a smoke test before the M6
dataset is available. Replace the manifest with the M6 subset for the actual experiment.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Creates short WAV files and a manifest with the expected M6-compatible schema.
python -m music_detection.cli make-demo-data --output data\demo

# Trains/evaluates the CPU-friendly classical branch.
python -m music_detection.cli classical --manifest data\demo\manifest.csv --output runs\classical

# Trains/evaluates the CNN branch (use --epochs 1 for a smoke test).
python -m music_detection.cli modern --manifest data\demo\manifest.csv --output runs\modern --epochs 1 --no-pretrained
```

For M6, create a CSV with columns `path,label,track_id,split`, where `label` is `0` for human
and `1` for AI-generated. Keep all segments from one source track in one split to prevent
leakage. Paths may be absolute or relative to the manifest location.

## Checkpoints

These are the intended commit/push points:

```text
checkpoint-1-scaffold       data contract, preprocessing, demo data, CLI
checkpoint-2-classical     MFCC + centroid SVM and reproducible metrics
checkpoint-3-modern        ResNet-18, checkpointing, evaluation, Grad-CAM
checkpoint-4-mid-eval      results table, plots, and presentation narrative
```

Commit each checkpoint after running the corresponding command and push it to the team
repository. Do not claim a metric until it has been produced on the held-out M6 test split.

## Project layout

```text
src/music_detection/
  cli.py          command-line entry points
  data.py         manifest validation, split handling, audio loading
  features.py     log-Mel, MFCC, and spectral-centroid extraction
  classical.py    SVM training and evaluation
  modern.py       ResNet-18 training, evaluation, and Grad-CAM
  synthetic.py    deterministic demo dataset generator
tests/            focused smoke tests
docs/mid_eval.md  presentation-ready speaking notes and work plan
```

## Scientific safeguards

- The split is track-aware and can be supplied explicitly in the manifest.
- Audio is resampled to 44.1 kHz and converted to mono.
- The same log-Mel configuration is used by both branches where applicable.
- Metrics are generated from held-out predictions, not training output.
- Demo data is for pipeline validation only; it is not evidence for the research hypothesis.