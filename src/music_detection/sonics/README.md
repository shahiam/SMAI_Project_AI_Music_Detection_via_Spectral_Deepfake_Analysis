# SONICS pipeline (parallel to M6)

SONICS = Suno/Udio AI songs (MP3, from Hugging Face) + matched human songs (YouTube). This pipeline only
adds **preprocessing**; training, tuning, evaluation and prediction use the same shared code as M6
(`music_detection.shared`), so the SONICS models have the same architecture but are trained and saved
separately (`runs/sonics_*`).

| Module (`music_detection.sonics.preprocessing.*`) | What it does |
|---|---|
| `fetch` | `plan`: metadata + stratified, class-balanced subset sized by `--gb-per-class`. `fakes`: AI zip parts, keeps only selected songs. `reals`: matching human songs per split from YouTube. `status`: progress. Default root `/Volumes/BACKUP/SONICS`. |
| `download_youtube` | yt-dlp downloader used by `fetch reals` (parallel jobs, per-split targets, replaces failed videos, backs off when rate-limited, resumable). |
| `profile_audio` | Codec, sample rate, channels, bitrate, duration and measured bandwidth per class. |
| `prepare_clips` | K clips per song, identical encoding for both classes (codec parity + same bandwidth), SONICS split kept; `--balance` = equal songs and clips per split, AI side keeps generator x label proportions. |
| `utils` | Shared helpers (split names, stratified allocation). |

Requirements: `pip install -U yt-dlp huggingface_hub` (plus the project requirements) and ffmpeg/ffprobe
on PATH (macOS `brew install ffmpeg`, Windows `winget install ffmpeg`).

## Steps (repo root; on macOS prefix long steps with `caffeinate -i`)

```bash
make sonics-plan                 # SONICS_GB=15 per class by default
make sonics-fakes                # terminal 1
make sonics-reals                # terminal 2, at the same time (SONICS_JOBS=8)
make sonics-status
make sonics-profile              # decide SONICS_CLIP_RATE (2 x lowest bandwidth; default 32000)
make sonics-clips                # -> data/sonics_manifest.csv
make sonics-profile-clips        # both classes must now look identical
make sonics-sr-baseline          # should be near chance on the clips
make sonics-train                # runs/sonics_classical, runs/sonics_modern, runs/sonics_ensemble
```

Equivalent module calls: `python -m music_detection.sonics.preprocessing.fetch plan --root /Volumes/BACKUP/SONICS --gb-per-class 15`, etc.
On Windows set `SONICS_ROOT := D:/SONICS` (or similar) in `local.mk`.

Keep all audio outside the repository. YouTube audio is copyrighted: use it only for this project and do
not redistribute it. SONICS itself is CC BY-NC 4.0 (non-commercial).
