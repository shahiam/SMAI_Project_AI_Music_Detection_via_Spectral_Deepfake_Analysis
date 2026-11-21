"""Log-Mel spectrograms, MFCC/centroid features, windowing and the band-limited sample rate."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import librosa

from ..data.audio import load_audio, load_full_audio

SAMPLE_RATE = 44_100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# All audio is band-limited to this rate for the main experiment. M6 AI generators are
# natively 16 kHz (AMG) or 32 kHz (MG/MGPT) while human clips are 22-48 kHz, so at
# 44.1 kHz the empty high band alone reveals the label. Resampling everything to 16 kHz
# gives both classes the same 0-8 kHz bandwidth.
BANDLIMITED_SAMPLE_RATE = 16_000


def stft_params(sr: int) -> tuple[int, int]:
    """Return (n_fft, hop_length) keeping roughly constant time resolution across rates."""
    n_fft = N_FFT if sr >= 32_000 else 1024
    return n_fft, n_fft // 4


def window_starts(
    total_seconds: float, window_duration: float = 5.0, hop_duration: float = 5.0
) -> list[float]:
    """Window start times (seconds) matching ``slice_audio_windows`` for a track length."""
    if total_seconds <= window_duration:
        return [0.0]
    starts = list(np.arange(0.0, total_seconds - window_duration + 1e-9, hop_duration))
    if starts[-1] + window_duration < total_seconds - 0.5:
        starts.append(total_seconds - window_duration)
    return [float(start) for start in starts]


def _pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or tile audio to target sample length without edge discontinuities."""
    if len(audio) >= target_length:
        return audio[:target_length]
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)
    repeats = int(np.ceil(target_length / len(audio)))
    return np.tile(audio, repeats)[:target_length].astype(np.float32)


def slice_audio_windows(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_duration: float = 5.0,
    hop_duration: float = 5.0,
) -> list[dict[str, Any]]:
    """Slice full audio waveform into uniform temporal windows with timestamps.
    
    Returns a list of dicts with keys:
    - 'audio': 1D numpy array of length window_samples
    - 'start_sec': start timestamp in seconds
    - 'end_sec': end timestamp in seconds
    - 'window_index': 0-indexed window number
    """
    win_samples = int(window_duration * sr)
    hop_samples = max(1, int(hop_duration * sr))
    total_samples = len(audio)

    if total_samples <= win_samples:
        padded = _pad_audio(audio, win_samples)
        duration_sec = total_samples / sr
        return [
            {
                "audio": padded,
                "start_sec": 0.0,
                "end_sec": max(duration_sec, window_duration),
                "window_index": 0,
            }
        ]

    starts = list(range(0, total_samples - win_samples + 1, hop_samples))
    # If the tail has uncovered samples (at least 0.5s), include a final window ending at total_samples
    if starts[-1] + win_samples < total_samples:
        tail_samples = total_samples - (starts[-1] + win_samples)
        if tail_samples >= int(0.5 * sr):
            starts.append(total_samples - win_samples)

    windows = []
    for idx, st in enumerate(starts):
        slice_audio = audio[st : st + win_samples]
        windows.append(
            {
                "audio": slice_audio.astype(np.float32),
                "start_sec": round(st / sr, 3),
                "end_sec": round((st + win_samples) / sr, 3),
                "window_index": idx,
            }
        )
    return windows


def log_mel_spectrogram(source: str | Path | np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute log-Mel spectrogram from an audio file path or an in-memory numpy array."""
    if isinstance(source, (str, Path)):
        audio, _ = load_audio(source, sr)
    else:
        audio = source
    n_fft, hop_length = stft_params(sr)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def classical_features(source: str | Path | np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC and spectral centroid features from an audio path or in-memory array."""
    if isinstance(source, (str, Path)):
        audio, _ = load_audio(source, sr)
    else:
        audio = source
    n_fft, hop_length = stft_params(sr)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length
    )
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), centroid.mean(axis=1), centroid.std(axis=1)])


def extract_windowed_spectrograms(
    source: str | Path | np.ndarray,
    window_duration: float = 5.0,
    hop_duration: float = 5.0,
    sr: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Extract log-Mel spectrograms for all windows across a full audio track."""
    if isinstance(source, (str, Path)):
        audio, _ = load_full_audio(source, sr)
    else:
        audio = source

    windows = slice_audio_windows(audio, sr=sr, window_duration=window_duration, hop_duration=hop_duration)
    results = []
    for win in windows:
        spec = log_mel_spectrogram(win["audio"], sr=sr)
        results.append(
            {
                "spectrogram": spec,
                "start_sec": win["start_sec"],
                "end_sec": win["end_sec"],
                "window_index": win["window_index"],
            }
        )
    return results


def extract_windowed_classical_features(
    source: str | Path | np.ndarray,
    window_duration: float = 5.0,
    hop_duration: float = 5.0,
    sr: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Extract classical features (MFCC + centroid) for all windows across a full audio track."""
    if isinstance(source, (str, Path)):
        audio, _ = load_full_audio(source, sr)
    else:
        audio = source

    windows = slice_audio_windows(audio, sr=sr, window_duration=window_duration, hop_duration=hop_duration)
    results = []
    for win in windows:
        feats = classical_features(win["audio"], sr=sr)
        results.append(
            {
                "features": feats,
                "start_sec": win["start_sec"],
                "end_sec": win["end_sec"],
                "window_index": win["window_index"],
            }
        )
    return results
