# Mid-evaluation presentation notes

## 1. Work completed so far

- Defined the research question as binary classification of human versus AI-generated audio.
- Implemented the shared data contract: 44.1 kHz mono audio, explicit track-aware splits, and binary labels.
- Implemented log-Mel spectrogram extraction using STFT and Mel filterbanks.
- Implemented the 30% classical branch: MFCCs plus spectral centroid statistics and an SVM.
- Implemented the 70% modern branch: transfer-learned ResNet-18 on the same spectrogram representation.
- Added held-out metrics and confusion matrices. The modern branch is ready for Grad-CAM integration after the first real M6 run.
- Added a synthetic smoke dataset so the full software path can be demonstrated without pretending demo metrics are research results.

## 2. Algorithms and implementation details

The waveform is resampled to 44.1 kHz, converted to mono, transformed with an STFT, mapped to 128 Mel bands,
and converted to decibels. The classical model aggregates 20 MFCCs and spectral-centroid statistics over time,
standardizes them, and fits a class-balanced SVM. The modern model resizes the log-Mel image to 224x224, repeats
the channel to match ImageNet-pretrained ResNet-18, replaces the final layer with two logits, and fine-tunes with
AdamW. The final report will compare precision, recall, F1, and ROC-AUC on the held-out M6 test set.

## 3. Challenges and mitigations

| Challenge | Mitigation |
|---|---|
| Codec or file-format shortcut | Use the uncompressed M6 WAV subset and matched preprocessing. |
| Segment leakage between splits | Validate that each `track_id` belongs to one split only. |
| Small initial compute budget | Start with ResNet-18 and bounded clips; use Ada/Colab for the full run. |
| Demo data can mislead | Clearly label synthetic runs as smoke tests, not scientific evidence. |

## 4. Remaining work and expected outcome

- Populate the manifest from the approved M6 subset and verify class/generator/genre balance.
- Train with multiple seeds and save the exact configuration and test predictions.
- Add Grad-CAM overlays and inspect whether highlighted regions align with known AI-generated sections.
- Compare the classical and CNN F1 scores; explain errors using confusion matrices and heatmaps.
- Freeze the test set before final tuning and write the final report.

## 5. Checkpoint narrative

1. `checkpoint-1-scaffold`: reproducible project and data contract.
2. `checkpoint-2-classical`: first measurable baseline and feature-extraction result.
3. `checkpoint-3-modern`: modern model and direct comparison.
4. `checkpoint-4-mid-eval`: figures, limitations, and presentation evidence.
