# Sarcospectus: AI-Generated Music Detection via Spectral Deepfake Analysis

Project implementation for SMAI, Team Sarcosuchus.

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
make install
make test
make dashboard
```

If `make` is not installed on Windows, run the equivalent commands directly:

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

# Launch the upload dashboard (processing/results are intentionally staged for the next checkpoint).
python app.py
# Equivalent standard Streamlit command:
# py -m streamlit run app.py
```

The `Makefile` creates a local `.venv` and uses it for all project commands, avoiding the
PEP 668 externally-managed-environment error on modern WSL/Linux distributions. It also
automatically uses `python3` on WSL/Linux when `python` is unavailable. On Windows, override
it with `make PYTHON=py`. It deliberately does not download the approximately 28 GB M6 archive
automatically.

To reuse an existing virtual environment instead of `.venv`, pass its path as `VENV`, e.g.
`make install VENV=../../SMAI_env` (Windows: `make install PYTHON=py VENV=.venv-win`). To avoid
typing it each time, put `VENV := ../../SMAI_env` in a `local.mk` file in the repo root; it is
gitignored, so each teammate can keep their own. All paths are quoted, so project and environment
folders containing spaces are supported.

For M6, create a CSV with columns `path,label,track_id,split`, where `label` is `0` for human
and `1` for AI-generated. Keep all segments from one source track in one split to prevent
leakage. Paths may be absolute or relative to the manifest location.

The public M6 archive is approximately 28 GB and should not be committed to this repository.
After downloading and extracting it locally, generate a manifest with:

```powershell
python -m music_detection.cli build-m6-manifest `
  --root data\M6_database `
  --output data\m6_manifest.csv `
  --include-duration
```

The command labels files below an `ai` directory as AI-generated, identifies `AMG`, `MGPT`,
and `MG` generators from filename prefixes, records genre metadata, and assigns deterministic
80/10/10 train/validation/test splits by `track_id`. Review the generated class and split counts
before training.

With the archive downloaded as `data\M6_database.tar.gz`, the full workflow is:

```powershell
make extract-m6
make m6-manifest
make m6-train
```

`m6-train` writes independent artifacts to `runs\m6_classical` and `runs\m6_modern`.
The 10-epoch ResNet run is expected to take hours on an 8 GB laptop GPU, depending on storage
speed and the number of M6 clips; the SVM feature extraction is CPU-heavy and may also take
hours. Start with `make m6-classical` and a short modern run before committing to the full run.

## Band-limited experiment (main result)

M6 AI generators output audio at 16 kHz (AMG) or 32 kHz (MG, MGPT), while human clips are
22-48 kHz. At 44.1 kHz the empty high band alone reveals the label: a rule that uses only the
native sample rate scores F1 0.94 on the test split. The main experiment therefore resamples all
audio to 16 kHz (`--sample-rate 16000`, now the default), trains on random 5 s crops with
augmentation, and evaluates full tracks (mean over 5 s windows). Thresholds and the ensemble
weight are tuned on validation, and the dashboard loads them from the model files.

```powershell
make sr-baseline   # shortcut check -> runs/sr_baseline.json
make m6-bl         # band-limited SVM/RF, augmented ResNet-18, validated ensemble
```

See `docs/mid_eval.md` for results and the mid-evaluation guide.

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

Both pipelines share one implementation (same architecture); only preprocessing is dataset-specific.

```text
src/music_detection/
  cli.py                        command-line entry point (python -m music_detection.cli <command>)
  shared/                       used by the M6 and SONICS pipelines
    data/        manifest.py    manifest contract + validation (path, label, track_id, split)
                 audio.py       mono/resampled loading, segment reads, durations
    features/    spectral.py    log-Mel, MFCC + centroid, windowing, band-limited sample rate
                 augmentations.py  volume, noise, MP3, pitch, stretch, SpecAugment
    training/    classical.py   SVM (+ Random Forest) on 42 MFCC/centroid features
                 modern.py      ResNet-18 on log-Mel spectrograms (random crops, augmentation)
    tuning/      thresholds.py  decision-threshold selection (low-FPR, F1, Youden, cost)
                 calibrate.py   re-tune the threshold of an existing model (tune-threshold command)
                 ensemble.py    validated modern/classical blend weight (ensemble command)
    evaluation/  metrics.py     metrics.json + confusion matrix
                 track_level.py window index, per-clip aggregation, per-generator breakdown
                 shortcut_checks.py  sample-rate-only baseline (sr-baseline command)
    output/      predict.py     full-track prediction (predict command, dashboard)
                 gradcam.py     Grad-CAM heatmaps
                 dashboard.py   Streamlit dashboard (launched via app.py)
  m6/preprocessing/
                 build_manifest.py  M6 manifest (build-m6-manifest command)
                 demo_data.py       synthetic smoke-test data (make-demo-data command)
  sonics/                       SONICS pipeline, see src/music_detection/sonics/README.md
    preprocessing/  fetch.py, download_youtube.py, prepare_clips.py, profile_audio.py, utils.py
tests/
  shared/  m6/  sonics/         unit tests mirroring the package layout
app.py                          launcher: `streamlit run app.py` / `make dashboard`
docs/                           mid-evaluation notes and slides
```

All CLI commands, Makefile targets and run folders are unchanged by this layout. M6 runs write to
`runs/m6_*`, SONICS runs to `runs/sonics_*`.

## SONICS pipeline (MP3, Suno/Udio)

```bash
make sonics-plan && make sonics-fakes && make sonics-reals   # data in ~/Datasets/SONICS
make sonics-profile && make sonics-clips && make sonics-train
```

See `src/music_detection/sonics/README.md` for details and the shortcut checks.

## Scientific safeguards

- The split is track-aware and can be supplied explicitly in the manifest.
- Audio is converted to mono and resampled (16 kHz by default, the band-limited pipeline; 44.1 kHz = original).
- The same log-Mel configuration is used by both branches where applicable.
- Metrics are generated from held-out predictions, not training output.
- Demo data is for pipeline validation only; it is not evidence for the research hypothesis.

## Run command windows:
.\.venv-win\Scripts\streamlit.exe run app.py