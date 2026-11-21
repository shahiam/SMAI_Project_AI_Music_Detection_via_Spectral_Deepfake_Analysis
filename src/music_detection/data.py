from __future__ import annotations

from pathlib import Path
from typing import Final

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

REQUIRED_COLUMNS: Final = {"path", "label", "track_id", "split"}
SPLITS: Final = {"train", "val", "test"}


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest without silently repairing invalid labels."""
    manifest_path = Path(manifest_path).resolve()
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest must contain at least one audio file.")
    if not set(frame["split"]).issubset(SPLITS):
        raise ValueError(f"split must contain only {sorted(SPLITS)}.")
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("label must be binary: 0=human, 1=AI-generated.")
    if frame["track_id"].duplicated().any():
        split_counts = frame.groupby("track_id")["split"].nunique()
        if (split_counts > 1).any():
            raise ValueError("A track_id must not occur in multiple splits.")
    frame["path"] = frame["path"].map(
        lambda value: str((manifest_path.parent / value).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    missing_files = [path for path in frame["path"] if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Audio files missing from manifest: {missing_files[:3]}")
    return frame


def load_audio(path: str | Path, sample_rate: int = 44_100, duration: float | None = 5.0):
    """Load mono audio at the project sample rate."""
    return librosa.load(path, sr=sample_rate, mono=True, duration=duration)


def load_full_audio(path: str | Path, sample_rate: int = 44_100):
    """Load the complete mono audio file without duration truncation."""
    return librosa.load(path, sr=sample_rate, mono=True, duration=None)


def get_audio_duration(path: str | Path) -> float:
    """Get duration of audio file in seconds."""
    return float(librosa.get_duration(path=path))


def load_segment(path: str | Path, sample_rate: int, start_sec: float, duration: float) -> np.ndarray:
    """Read only ``duration`` seconds starting at ``start_sec`` and resample to ``sample_rate``.

    Much faster than decoding the whole file when training on random crops or scoring
    individual windows. Falls back to librosa for formats soundfile cannot seek.
    """
    try:
        with sf.SoundFile(str(path)) as handle:
            native_sr = handle.samplerate
            handle.seek(min(int(start_sec * native_sr), max(handle.frames - 1, 0)))
            audio = handle.read(int(round(duration * native_sr)), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if native_sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sample_rate)
    except (RuntimeError, sf.LibsndfileError):
        audio, _ = librosa.load(path, sr=sample_rate, mono=True, offset=start_sec, duration=duration)
    return np.asarray(audio, dtype=np.float32)


def native_duration(path: str | Path) -> float:
    """Track duration in seconds from the file header (no decoding)."""
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except (RuntimeError, sf.LibsndfileError):
        return get_audio_duration(path)
