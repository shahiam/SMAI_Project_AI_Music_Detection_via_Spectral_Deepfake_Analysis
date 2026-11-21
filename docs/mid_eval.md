# Mid-evaluation preparation guide (Team Sarcosuchus)

The TA email asks for four things. This guide follows the same order, so it can be used directly
as the outline for slides and speaking notes:

1. Work completed so far
2. Algorithms, models, techniques and implementation details
3. Challenges faced and how they were addressed
4. Remaining work, future plans and expected outcomes for the final evaluation (1 December)

At the end there are likely examiner questions with answers, a demo script, and a checklist for
tonight. **Every number below comes from a file in `runs/`.** If you are asked where a number
comes from, open that file.

---

## 0. The 30-second story (memorise this)

> We detect AI-generated music by turning audio into log-Mel spectrograms and classifying them
> with a classical MFCC + SVM baseline (30%) and a transfer-learned ResNet-18 (70%), trained on
> the M6 benchmark. Our first M6 run gave 98.9% F1 for the CNN. We then audited the data and found
> a shortcut: every AI generator in M6 outputs audio at 16 or 32 kHz, while human recordings are
> 22-48 kHz. A rule that looks *only at the file's sample rate* already gets F1 0.94. We fixed it
> by band-limiting all audio to 16 kHz, so both classes share the same 0-8 kHz content. We then
> retrained with random crops and augmentation and evaluated full tracks, not just the first 5
> seconds. The SVM drops from F1 0.95 to 0.80, while the CNN holds at 0.98: it beats the
> classical baseline by 18 F1 points, and on the hardest AI clips (44.1 kHz full songs) it is
> right 79% of the time vs 7% for the SVM. Generalisation beyond M6 is the focus for the final
> evaluation.

This story turns a weakness into the strongest part of the presentation: it shows we understand
our data rather than just reporting a high number.

---

## 1. Work completed so far

| Area | Status | Where |
|---|---|---|
| M6 dataset downloaded (≈28 GB), manifest built (13,292 clips: 4,098 human, 9,194 AI) | Done | `data/m6_manifest.csv`, `m6_manifest.py` |
| Track-aware, deterministic 80/10/10 split (hash of `track_id`) | Done | `m6_manifest.py`, checked in `data.py` |
| Shared preprocessing: mono, resampling, STFT → Mel → dB | Done | `features.py` |
| Classical branch: MFCC + spectral centroid → SVM, plus Random Forest | Done | `classical.py` |
| Modern branch: ResNet-18 (ImageNet weights) fine-tuned on spectrograms | Done | `modern.py` |
| Augmentation: volume, noise, MP3 compression, pitch shift, time stretch, SpecAugment | Done and used in the band-limited run | `augmentations.py` |
| Full-track analysis: 5 s windows, mean of window probabilities | Done (training eval + dashboard) | `features.py`, `evaluation.py` |
| Decision-threshold tuning on validation (low false-positive criterion) | Done | `metrics.py` |
| **Shortcut audit + sample-rate baseline + band-limited retraining** | Done | `evaluation.py`, `runs/sr_baseline.json` |
| Validated ensemble (weight + threshold fitted on validation) | Done | `evaluation.py`, `runs/m6_bl_ensemble/` |
| Streamlit dashboard: upload, per-window timeline, model choice | Done | `app.py` |
| Grad-CAM function | Implemented, not yet evaluated | `modern.py::save_gradcam` |
| Unit tests (19) | Passing | `tests/` |

### Dataset facts to quote

- 13,292 WAV clips. Human: 4,098. AI: 9,194 (AMG 4,998 · MG 1,666 · MGPT 1,650 · unlabelled
  generator "unknown_ai" 880, the lyrics/long-song/general folders).
- Split: train 10,647 · val 1,317 · test 1,328 clips. The test set has 6,473 five-second windows.
- Class imbalance ≈ 2.2 : 1 (AI : human). This is handled with class-weighted loss and
  `class_weight="balanced"`.

---

## 2. Algorithms, models and implementation details

### 2.1 Signal pipeline (shared by both branches)

1. Load audio, convert to mono, **resample to 16 kHz** (band-limited experiment; the original
   pipeline used 44.1 kHz).
2. Cut each track into **5-second windows** (hop = 5 s; the last partial window is aligned to the
   end of the track).
3. STFT with n_fft = 1024 (64 ms), hop = 256 (16 ms), Hann window.
4. Map onto 128 Mel bands → power to dB (`ref = max`, clipped to [-80, 0] dB).
5. A track's score is the **mean AI probability over all its windows**. The dashboard uses the
   same rule, so the reported numbers match what the demo does.

### 2.2 Classical branch (30%)

- Features per window (42-dim): mean and standard deviation of 20 MFCCs over time, plus mean and
  standard deviation of the spectral centroid.
- Models: StandardScaler → RBF-SVM (`class_weight=balanced`, Platt-scaled probabilities), and a
  500-tree Random Forest for comparison.
- Training: 2 random 5 s crops per training clip (21,294 feature vectors).
- Why it should be weaker: MFCCs summarise the spectral *envelope*. Averaging over time throws
  away fine time-frequency texture, which is exactly where generator artifacts are hypothesised
  to live.

### 2.3 Modern branch (70%)

- ResNet-18 pretrained on ImageNet. The final layer is replaced by 2 logits. The 1-channel
  spectrogram is resized to 224×224 and repeated to 3 channels.
- Training: AdamW (lr 1e-4, weight decay 1e-4), cosine LR schedule, batch 32, 12 epochs,
  class-weighted cross-entropy.
- **Random crops:** every epoch takes a new random 5 s window from anywhere in each clip (not
  always the first 5 s).
- **Augmentation:** volume ±6 dB, Gaussian noise at 20-40 dB SNR, real MP3 encode/decode,
  pitch ±2 semitones, time stretch ±8%, and SpecAugment (2 frequency + 2 time masks).
- Model selection: the epoch with the best **validation ROC-AUC** (threshold-free), so a lucky
  threshold cannot pick the epoch.
- Grad-CAM on the last residual block (`layer4`) shows which time-frequency regions drove the
  decision.

### 2.4 Decision threshold

- Criterion `low_fpr`: on the **validation** split, choose the threshold that maximises recall
  while keeping the false-positive rate ≤ 2%.
- Why: a false positive means a human artist is accused of using AI, which is the costlier error
  for this application.
- The threshold is saved inside the model file. The dashboard's `bandlimited` profile reads it
  from there. **The dashboard currently runs the `legacy` profile** (original 44.1 kHz models,
  cutoffs 0.70 / 0.65), because the band-limited models miss modern Suno-style covers (see 4.4).

### 2.5 Ensemble

- p = w·p_CNN + (1−w)·p_SVM. The weight w is chosen on the validation split by ROC-AUC (grid
  0, 0.05, …, 1). The threshold uses the same low-FPR rule. Both are stored in
  `runs/m6_bl_ensemble/ensemble.json` (used by the dashboard's `bandlimited` profile).
- The `legacy` profile (current dashboard default) keeps the hand-picked 40/60 blend at 0.55,
  which has no validation behind it.

### 2.6 Evaluation protocol

- Metrics: precision, recall, F1, false-positive rate, ROC-AUC, confusion matrix, and
  **per-generator accuracy**.
- The test split is used once, after all choices (epoch, threshold, ensemble weight) are made on
  validation.

---

## 3. Challenges and how we addressed them

### 3.1 The bandwidth shortcut (the main finding — spend the most time here)

What we found when we read the native sample rate of every file:

| Group | Native sample rate |
|---|---|
| Human (4,098) | 44.1 kHz (2,872), 22.05 kHz (1,000), 24 kHz (168), 48 kHz (56), 2 others |
| AMG (4,998) | **all 16 kHz** |
| MG + MGPT (3,316) | **all 32 kHz** |
| unknown_ai (880) | 44.1 kHz |

Resampling to 44.1 kHz cannot create frequencies that were never recorded. An AMG clip therefore
has an empty band above 8 kHz and an MG/MGPT clip has nothing above 16 kHz. A CNN sees that empty
band instantly.

**Proof:** a classifier that only knows `P(AI | native sample rate)` from the training split
(`runs/sr_baseline.json`) scores on the test set:

| Model | F1 | ROC-AUC | Notes |
|---|---|---|---|
| Sample-rate rule (no audio at all) | **0.940** | **0.961** | 100% on AMG/MG/MGPT/human, 0% on unknown_ai |
| Our original SVM (44.1 kHz) | 0.947 | 0.974 | about the same as the rule |
| Our original CNN (44.1 kHz) | 0.989 | 0.999 | |

**Fix:** resample everything to 16 kHz, the lowest rate in the data. Both classes then share
exactly the same 0-8 kHz band, and the information left is about *content and texture*, not file
format. This mirrors the proposal's own argument about MP3-vs-WAV codec bias (Section 3), applied
to sample rate.

### 3.2 Only the first 5 seconds were used

The earlier training and test code loaded only the first 5 s of each clip, while the dashboard
analysed full tracks. Now training uses random crops, and val/test score every window of every
track, consistent with the dashboard.

### 3.3 Thresholds were hard-coded in the dashboard

The dashboard used 0.70 / 0.65 / 0.55 and a 40/60 ensemble with no validation behind them. Now
all thresholds and the ensemble weight are fitted on the validation split and loaded from the
model files.

### 3.4 Data leakage between splits

Splits are assigned by a hash of `track_id`. `load_manifest` refuses a manifest in which a track
appears in two splits. Open item: near-duplicate prompts from the same generator could still be
spread across splits (see Section 4).

### 3.5 Engineering issues

- The 28 GB archive and slow feature extraction: added spectrogram caching, parallel feature
  extraction (joblib) and checkpoint/resume for the CNN.
- Windows DataLoader workers oversubscribed CPU threads, making augmentation about 15× slower.
  Fixed by pinning each worker to one BLAS/OpenMP thread.
- One malformed WAV in M6 (no RIFF header) is skipped explicitly.
- Codec bias (MP3 vs WAV) from the proposal: M6 is all WAV, and MP3 compression is applied as an
  augmentation to *both* classes.

---

## 4. Results

### 4.1 Before vs after the fix (test split)

Thresholds are tuned on validation, except the sample-rate rule and the original SVM, which use the
default 0.5 (the original SVM predates threshold tuning).

| Model | Sample rate | Precision | Recall | F1 | FPR | ROC-AUC | Source |
|---|---|---|---|---|---|---|---|
| Sample-rate rule | – | 1.000 | 0.886 | 0.940 | 0.000 | 0.961 | `runs/sr_baseline.json` |
| SVM (original) | 44.1 kHz, first 5 s | 0.948 | 0.947 | 0.947 | – | 0.974 | `runs/m6_classical/metrics.json` |
| CNN (original) | 44.1 kHz, first 5 s | 0.985 | 0.993 | 0.989 | 0.034 | 0.999 | `runs/m6_modern/metrics.json` |
| SVM (band-limited) | 16 kHz, full track | 0.974 | 0.684 | 0.804 | 0.042 | 0.960 | `runs/m6_bl_classical/metrics.json` |
| Random Forest (band-limited) | 16 kHz, full track | 0.973 | 0.631 | 0.765 | 0.039 | 0.951 | same file, `random_forest` |
| **CNN (band-limited, augmented)** | 16 kHz, full track | 0.988 | 0.976 | **0.982** | 0.027 | **0.998** | `runs/m6_bl_modern/metrics.json` |
| Ensemble (band-limited) | 16 kHz, full track | 0.988 | 0.976 | 0.982 | 0.027 | 0.998 | `runs/m6_bl_ensemble/ensemble.json` |

### 4.2 Per-generator accuracy, band-limited (test)

| Generator (n) | SVM | CNN |
|---|---|---|
| Human (407) — true-negative rate | 95.8% | 97.3% |
| AMG (520) | 86.2% | 100% |
| MG (153) | 54.2% | 100% |
| MGPT (143) | 64.3% | 100% |
| unknown_ai (105) | 6.7% | **79.0%** |

### 4.3 How to read these results

- **The fix exposed a real gap.** The SVM fell from F1 0.947 to 0.804. Much of its earlier score
  came from bandwidth, not from anything about how the music was generated.
- **The SVM's tuned threshold is strict (0.905).** At the default 0.5 it reaches F1 0.942 but
  flags 17% of human tracks. The low-FPR rule trades recall for protecting human artists. Say
  this explicitly; it is a design choice, not an accident.
- **unknown_ai is the hard case.** These are 44.1 kHz full songs with vocals (closest to
  Suno/Udio output). The sample-rate rule gets 0% on them by construction, and they are where a
  real detector has to work.
- **The most informative Random Forest feature is `mfcc0_mean`**, which tracks overall
  loudness/energy. That hints at a mastering/loudness difference between the classes, a possible
  second shortcut to examine for the final evaluation.
- **The CNN beats the SVM by 17.8 F1 points (0.982 vs 0.804)**, which meets the proposal's
  ≥10-point target. Significance testing (seeds, bootstrap CIs, McNemar) is still pending. Its
  AUC barely changed after the fix (0.999 → 0.998). So, unlike the SVM, the CNN was *not* relying
  only on bandwidth: it finds real cues inside 0-8 kHz.
- **The CNN gets unknown_ai right 79% of the time vs 7% for the SVM**, the clearest evidence that
  the learned features capture something the MFCC summary cannot.
- **The ensemble chose 100% CNN weight on validation** (`ensemble.json`: modern_weight = 1.0).
  Adding the SVM never improved validation AUC, so the "hybrid" is in practice the CNN. Report
  this honestly; it is a result, not a bug.
- **Caveat to raise yourselves:** AMG/MG/MGPT are 100% even after band-limiting. Part of that may
  still be dataset construction (e.g. the AI `a_instruments` clips are clean single-instrument
  renders, while the human ones include room recordings). Leave-one-generator-out and the
  external test set (Section 5) are designed to test exactly this.
- Training curve (`runs/m6_bl_modern/history.json`): val AUC 0.986 after epoch 1 → 0.999 at
  epoch 12 (best epoch); the val FPR stays ≤ 1.9% throughout.

### 4.4 Out-of-distribution check: six real YouTube tracks

Mean AI probability over full tracks (the legacy ensemble is 0.4·CNN + 0.6·SVM, cutoff 0.55):

| Track | Old CNN | Old SVM | Old ensemble | New CNN | New SVM |
|---|---|---|---|---|---|
| PUBLIC – Make You Mine (human) | 0.03 | 0.04 | 0.03 | 0.29 | 0.21 |
| Billie Jean (Country AI Cover) | 0.81 | 0.77 | 0.79 | 0.05 | 0.18 |
| Gangsta's Paradise 1950s Soul (AI) | 0.32 | 0.72 | 0.563 | 0.01 | 0.18 |
| KURBAAN HUA (Rock Version) | 0.19 | 0.62 | 0.45 | 0.04 | 0.35 |
| Selena Gomez – Good For You (human) | 0.68 | 0.46 | 0.547 | 0.08 | 0.58 |
| KFC Pigeon Yodel (AI) | 0.75 | 0.71 | 0.73 | 0.19 | 0.19 |

- The band-limited models miss modern Suno/Udio-style covers: every AI track scores < 0.2.
  M6's generators are older research models, and band-limiting removes everything above 8 kHz.
- The legacy ensemble classifies all six "correctly", but two tracks sit 0.003 and 0.013 from the
  hand-set 0.55 cutoff. The old CNN alone gets Gangsta's Paradise wrong. These cutoffs were
  chosen while looking at these songs, so the result is optimistic.
- Conclusion: **neither model yet generalises to modern AI covers**. Fixes for the final:
  (1) add modern-generator training data (e.g. SONICS: Suno/Udio + matched human songs);
  (2) train at 44.1 kHz but randomly low-pass *human* clips during training, so a missing high
  band stops meaning "AI" while real high-frequency cues are kept. These six songs (and more)
  become an external test set that is never used for tuning.

---

## 5. Remaining work, plans and expected outcomes (to 1 December)

| Week | Work | Expected outcome |
|---|---|---|
| 23-26 Nov | **External test set**: 20-50 human and 20-50 AI tracks (Suno/Udio/MusicGen + songs we own or that are CC-licensed) never used in training | Real-world precision/recall/FPR; the true test of generalisation |
| 23-26 Nov | **Loudness check**: normalise to a fixed LUFS before features and retrain the SVM; see whether `mfcc0` stops dominating | Know whether loudness is a second shortcut |
| 23-28 Nov | **Statistics**: 3 seeds per model, bootstrap 95% CIs on F1/AUC, McNemar test CNN vs SVM | A "statistically meaningful margin", as the proposal promises |
| 24-28 Nov | **Leave-one-generator-out**: train without AMG (or MG/MGPT), test on the left-out generator | Evidence of whether the model learns artifacts in general or one system's fingerprint |
| 25-29 Nov | **Grad-CAM evaluation**: synthetic splices (human → AI → human at known times); measure whether the window probabilities and heatmaps peak on the AI segment | Turns the localisation claim into a number |
| 26-29 Nov | **Dashboard**: show both model probabilities and an "uncertain" label when the models disagree | Honest output for borderline tracks |
| 29 Nov-1 Dec | Final report and slides: before/after table, per-generator analysis, Grad-CAM figures, limitations | |

Proposal promises and their status:

- Precision/Recall/F1/ROC-AUC per pipeline: **done**.
- ≥10-point F1 gap CNN vs classical: **met (17.8 points)**, significance testing pending.
- Grad-CAM localisation validated on tracks with known AI sections: **pending** (synthetic splices).
- SVM **and** Random Forest: **done**.

---

## 6. Likely examiner questions

**Q: If a sample-rate rule gets F1 0.94, how do we know the CNN learned anything real?**
After band-limiting to 16 kHz, both classes share the same bandwidth and the rule no longer
applies. The SVM dropped to 0.80; the CNN stayed at 0.98. And on unknown_ai (44.1 kHz AI songs,
where the rule scores 0%) the CNN is right 79% of the time.

**Q: Why does the ensemble use only the CNN?**
We fitted the blend weight on validation; every amount of SVM weight lowered validation AUC,
so the optimum is 100% CNN. The SVM is still useful as the interpretable baseline.

**Q: Doesn't band-limiting to 8 kHz throw away the high-frequency artifacts your proposal talked
about?**
Yes, and that is the trade-off. In M6 the 8-16 kHz band is *only* present for some classes, so a
model cannot learn generator artifacts there without also learning the sample rate. With a
dataset where both classes are full-band (our planned external set), the upper band can be
reintroduced.

**Q: How do you know there is no leakage between train and test?**
Splits are assigned by a hash of the track ID, and the loader rejects any track ID that appears
in two splits. We have not yet checked near-duplicate prompts within a generator; that is on the
plan.

**Q: Why tune the threshold for low false positives instead of using 0.5?**
Wrongly accusing a human artist is the costlier error. We cap the validation FPR at 2% and
maximise recall under that cap. The threshold is chosen on validation, never on test.

**Q: Why ResNet-18 and ImageNet weights for spectrograms?**
Spectrograms are not natural images, but the early filters (edges, textures, periodic patterns)
still transfer, and pretrained weights converge much faster on ~10k clips. ResNet-18 is small
enough to train on a laptop GPU (RTX 4060) in about 30 minutes.

**Q: Why mean aggregation over windows?**
It is robust to a single noisy window and matches the training unit (5 s). Max-aggregation would
better catch partly-AI tracks; we will compare both on the spliced-track experiment.

**Q: What about class imbalance?**
2.2:1 AI:human. We use class-weighted cross-entropy (CNN) and `class_weight="balanced"` (SVM/RF),
and report F1 and FPR rather than accuracy alone.

**Q: What is "unknown_ai"?**
M6 AI clips whose file names carry no generator prefix (lyrics, long songs, general). They are
44.1 kHz, so they are the only AI clips the sample-rate shortcut cannot catch, which makes them
the most realistic part of the test set.

**Q: Who did what?** ⟨Fill in honestly before the eval: e.g. Aaryan — data pipeline, M6
manifest, CNN training, dashboard; Soham — classical branch, augmentation, evaluation,
threshold tuning. Both — shortcut analysis, report.⟩

---

## 7. Demo plan (2-3 minutes)

1. `.\.venv-win\Scripts\streamlit.exe run app.py` serves the **legacy** (original 44.1 kHz)
   models by default. To demo the band-limited models instead:
   `$env:SARCOSPECTUS_PROFILE = "bandlimited"` before launching.
2. Upload a few clips and show the per-window timeline, the cutoff in the caption, and switch
   between ResNet, SVM and Ensemble.
3. If asked why the demo uses the older models, use the answer in 4.4 / Section 6.
4. Have the backup ready: screenshots of the dashboard and confusion matrices in case the laptop
   or demo fails.

---

## 8. Checklist for tonight

- [ ] Both members can tell the 30-second story (Section 0) without notes.
- [ ] Slides built (see `docs/ppt_prompt.md`), with the confusion matrices from
      `runs/m6_bl_*/confusion_matrix.png`.
- [ ] Demo clips chosen and copied to the desktop; dashboard tested once end to end.
- [ ] Contribution split agreed (Section 6, last question).
- [ ] Laptop charged; the venv activates; `runs/m6_bl_*` folders present on the demo machine.

## Reproducing everything

```powershell
python -m music_detection.cli sr-baseline --manifest data\m6_manifest.csv --output runs\sr_baseline.json
python -m music_detection.cli classical --manifest data\m6_manifest.csv --output runs\m6_bl_classical --sample-rate 16000
python -m music_detection.cli modern --manifest data\m6_manifest.csv --output runs\m6_bl_modern --sample-rate 16000 --epochs 12 --augment --audio-augment --workers 14
python -m music_detection.cli ensemble --classical runs\m6_bl_classical --modern runs\m6_bl_modern --output runs\m6_bl_ensemble
```

(or `make sr-baseline` and `make m6-bl`). Pass `--sample-rate 44100` to reproduce the original
shortcut-affected pipeline.
