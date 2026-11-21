"""Profile audio files per class: header sample rate, codec, bitrate, channels, duration,
and the *effective* bandwidth (highest frequency that actually contains energy).

Why: on M6 the AI clips were 16/32 kHz and the human clips 22-48 kHz, so a model could
detect "AI" from the empty top of the spectrogram. Run this on any new dataset (and on
the prepared clips) to check that both classes look the same in these respects.

Requirements: numpy, pandas, ffmpeg + ffprobe on PATH.

Examples:
    python -m music_detection.sonics.preprocessing.profile_audio \
        --dir fake=~/Datasets/SONICS/fake_songs --dir real=~/Datasets/SONICS/real_songs \
        --sample 200 --csv data/sonics_profile_raw.csv

    python -m music_detection.sonics.preprocessing.profile_audio \
        --dir fake=~/Datasets/SONICS/clips/fake --dir real=~/Datasets/SONICS/clips/real
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import AUDIO_EXTS



def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,bit_rate:format=duration,bit_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        fmt = info.get("format", {})
    except (ValueError, KeyError, IndexError):
        return {}
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0)) or None,
        "channels": stream.get("channels"),
        "bitrate_kbps": round(int(bit_rate) / 1000) if bit_rate else None,
        "duration_s": round(float(fmt.get("duration", "nan")), 1),
    }


def effective_bandwidth(path: Path, sample_rate: int, seconds: float = 30.0, drop_db: float = 50.0) -> float | None:
    """Highest frequency (kHz) whose average level is within drop_db of the 2-6 kHz level.

    Codec low-passes and upsampling leave a steep cliff, so this finds where the content stops."""
    if not sample_rate:
        return None
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", "10", "-t", str(seconds), "-i", str(path),
         "-ac", "1", "-f", "f32le", "-"],
        capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < 8192:  # short file: retry from the start
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-t", str(seconds), "-i", str(path), "-ac", "1", "-f", "f32le", "-"],
            capture_output=True,
        )
        audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < 8192:
        return None
    n_fft, hop = 4096, 2048
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop]
    power = (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(power + 1e-20)
    # Smooth over ~100 Hz so single tonal peaks or notches do not dominate.
    width = max(1, int(100 / (sample_rate / n_fft)))
    db = np.convolve(db, np.ones(width) / width, mode="same")
    freqs = np.arange(db.size) * sample_rate / n_fft
    reference = np.median(db[(freqs >= 2000) & (freqs <= 6000)])
    above = np.nonzero(db >= reference - drop_db)[0]
    if above.size == 0:
        return None
    return round(above[-1] * sample_rate / n_fft / 1000, 2)


def profile_file(label: str, path: Path) -> dict:
    row = {"class": label, "file": path.name, **probe(path)}
    row["bandwidth_khz"] = effective_bandwidth(path, row.get("sample_rate") or 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile sample rate, codec, bitrate and real bandwidth per class.")
    parser.add_argument("--dir", action="append", required=True, metavar="LABEL=PATH",
                        help="Class label and folder, e.g. fake=/path/to/fake_songs (repeatable)")
    parser.add_argument("--sample", type=int, default=200, help="Random files per class (0 = all)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Optional per-file output CSV")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH (brew install ffmpeg / winget install ffmpeg)")

    jobs = []
    rng = random.Random(args.seed)
    for spec in args.dir:
        label, _, folder = spec.partition("=")
        files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._")]
        if not files:
            print(f"Warning: no audio files under {folder}")
        if args.sample and len(files) > args.sample:
            files = rng.sample(files, args.sample)
        jobs += [(label, f) for f in files]

    print(f"Profiling {len(jobs)} files...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: profile_file(*job), jobs))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("Nothing profiled.")

    pd.set_option("display.width", 140)
    for label, group in frame.groupby("class"):
        print(f"\n=== {label}: {len(group)} files ===")
        print("codec:        ", group["codec"].value_counts().to_dict())
        print("sample rate:  ", group["sample_rate"].value_counts().sort_index().to_dict())
        print("channels:     ", group["channels"].value_counts().to_dict())
        print("bitrate kbps: ", group["bitrate_kbps"].describe(percentiles=[0.1, 0.5, 0.9])[["min", "10%", "50%", "90%", "max"]].round(0).to_dict())
        print("duration s:   ", group["duration_s"].describe(percentiles=[0.5])[["min", "50%", "max"]].round(1).to_dict())
        print("bandwidth kHz:", group["bandwidth_khz"].describe(percentiles=[0.05, 0.5, 0.95])[["min", "5%", "50%", "95%", "max"]].round(1).to_dict())

    lows = frame.groupby("class")["bandwidth_khz"].quantile(0.05)
    print("\nLowest typical bandwidth per class (5th percentile, kHz):", lows.round(1).to_dict())
    print(f"Suggested common sample rate for prepare_clips.py: <= {int(lows.min() * 2000)} Hz "
          "(2 x the lowest bandwidth), e.g. 32000 or 16000")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"Per-file results: {args.csv}")


if __name__ == "__main__":
    main()
