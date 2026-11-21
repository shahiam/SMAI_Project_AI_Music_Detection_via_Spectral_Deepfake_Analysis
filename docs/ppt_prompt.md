# Prompt for generating the mid-evaluation slide deck

Paste everything below the line into Claude Code, run from the repository root.

---

Build a mid-evaluation slide deck (.pptx) for our SMAI course project "AI-Generated Music
Detection via Spectral Deepfake Analysis" (Team Sarcosuchus: Aaryan Shah 2024113014, Soham
Acharya 2024113012). The evaluation is a ~10 minute talk with Q&A, given to the professor and TAs.

**Sources of truth: read these first and take every number from them, never from memory:**
- `docs/mid_eval.md`: narrative, results tables, challenges, plan, Q&A
- `runs/sr_baseline.json`: sample-rate-only baseline
- `runs/m6_classical/metrics.json`, `runs/m6_modern/metrics.json`: original 44.1 kHz runs
- `runs/m6_bl_classical/metrics.json` (includes `random_forest`), `runs/m6_bl_modern/metrics.json`,
  `runs/m6_bl_modern/history.json`, `runs/m6_bl_ensemble/ensemble.json`: band-limited runs
- `runs/m6_bl_*/confusion_matrix.png`: confusion matrices to embed
- The original proposal (`Sarcosuchus_Project_Proposal.pdf`, if available) for the stated goals

If any value is missing or a ⟨placeholder⟩ in `docs/mid_eval.md` is still unfilled, read it
from the JSON files. If it is not there either, put a visible "TBD" on the slide and tell me,
rather than inventing a number.

**The TA email requires these four sections. Use them as the deck's spine, in this order:**
1. Work completed so far
2. Algorithms, models, techniques and implementation details
3. Challenges faced and how they were addressed
4. Remaining work, future plans and expected outcomes for the final evaluation

**Slide plan (about 14 slides):**
1. Title: project name, team, course, date.
2. Problem and motivation: why detect AI music; framing as image-style forensics on spectrograms.
3. Pipeline diagram: audio → resample (16 kHz) → 5 s windows → STFT → Mel → dB → two branches
   (MFCC+centroid → SVM/RF, and ResNet-18) → window mean → threshold → label + per-window timeline.
   Draw it as shapes, not a screenshot.
4. Dataset: M6 composition (13,292 clips; human 4,098; AMG / MG / MGPT / unknown_ai counts),
   track-aware 80/10/10 split, class imbalance.
5. Work completed: compact checklist table (from Section 1 of the guide).
6. Classical branch details: 42-dim features, SVM + RF, why we expect it to be weaker.
7. Modern branch details: ResNet-18 transfer learning, random crops, augmentation list,
   AUC-based epoch selection, low-FPR threshold.
8. **Key finding: the bandwidth shortcut.** Table of native sample rate by class, and a bar chart
   comparing F1 of the sample-rate-only rule vs the original SVM vs the original CNN. Headline:
   "A rule that never listens to the audio gets F1 0.94."
9. The fix: band-limit everything to 16 kHz. Before/after spectrogram sketch (empty band above
   8 / 16 kHz for AI vs full band for human), and why this mirrors the proposal's codec-bias argument.
10. Results after the fix: the main before/after table, plus the band-limited CNN and SVM
    confusion matrices side by side.
11. Per-generator accuracy chart (grouped bars: SVM vs CNN for human / AMG / MG / MGPT /
    unknown_ai). Call out unknown_ai as the hard, realistic case.
12. Other challenges: first-5-seconds evaluation, hard-coded thresholds, unvalidated ensemble
    weights, Windows worker thread oversubscription, malformed WAV. One line each: problem → fix.
13. Remaining work timeline to 1 December (table from Section 5) and the proposal-promise
    status list.
14. Demo + thank you / questions.

Put 3-5 bullet speaking notes in the **speaker notes** of every slide, taken from the guide, so
either team member can present any slide. Put the likely Q&A answers from Section 6 of the guide
into the notes of a final hidden backup slide.

**Design:**
- Clean, technical, dark or light but consistent; one accent colour; large readable numbers.
- At most ~40 words of body text per slide; tables are fine.
- Charts must be real charts built from the JSON values (python-pptx native charts or matplotlib
  PNGs), with axis labels and units.
- Label every result with its split ("test", threshold tuned on validation) and its run folder.
- Present the drop from 98.9% F1 as a finding, not a failure.

**Output:** save as `docs/Sarcosuchus_MidEval.pptx`. Then render every slide to an image, check
for overflowing text, overlapping shapes and unreadable charts, fix any you find, and report which
numbers came from which file.
