from pathlib import Path

import numpy as np
import soundfile as sf

from music_detection.shared.data.audio import load_segment
from music_detection.shared.features.spectral import (
    log_mel_spectrogram,
    slice_audio_windows,
    stft_params,
    window_starts,
)


def test_window_starts_match_slice_audio_windows():
    sr = 16_000
    for seconds in (3.0, 5.0, 10.0, 12.7, 30.0):
        audio = np.zeros(int(seconds * sr), dtype=np.float32)
        expected = [w["start_sec"] for w in slice_audio_windows(audio, sr=sr)]
        assert np.allclose(window_starts(seconds), expected, atol=1e-3), seconds


def test_load_segment_resamples_and_crops(tmp_path: Path):
    path = tmp_path / "tone.wav"
    sr = 32_000
    t = np.arange(int(8.0 * sr)) / sr
    sf.write(path, np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    segment = load_segment(path, 16_000, start_sec=2.0, duration=5.0)
    assert abs(len(segment) - 5 * 16_000) <= 2


def test_bandlimited_spectrogram_shape():
    assert stft_params(44_100) == (2048, 512)
    assert stft_params(16_000) == (1024, 256)
    spec = log_mel_spectrogram(np.random.randn(5 * 16_000).astype(np.float32), sr=16_000)
    assert spec.shape[0] == 128 and np.isfinite(spec).all()
