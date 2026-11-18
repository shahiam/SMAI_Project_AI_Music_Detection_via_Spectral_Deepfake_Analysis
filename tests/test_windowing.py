from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from music_detection.features import (
    SAMPLE_RATE,
    classical_features,
    extract_windowed_classical_features,
    extract_windowed_spectrograms,
    log_mel_spectrogram,
    slice_audio_windows,
)


def test_slice_audio_windows_short_track_padded():
    # 2 seconds track when window is 5 seconds
    sr = 44_100
    audio = np.random.randn(int(2.0 * sr)).astype(np.float32)
    windows = slice_audio_windows(audio, sr=sr, window_duration=5.0, hop_duration=5.0)

    assert len(windows) == 1
    assert len(windows[0]["audio"]) == int(5.0 * sr)
    assert windows[0]["start_sec"] == 0.0
    assert windows[0]["end_sec"] >= 2.0
    assert windows[0]["window_index"] == 0


def test_slice_audio_windows_long_track_covers_full_duration():
    # 13.5 seconds track with 5.0s window and 5.0s hop
    sr = 44_100
    audio = np.random.randn(int(13.5 * sr)).astype(np.float32)
    windows = slice_audio_windows(audio, sr=sr, window_duration=5.0, hop_duration=5.0)

    # Starts should be: 0.0, 5.0, and tail window ending at 13.5 (8.5 to 13.5)
    assert len(windows) == 3
    assert windows[0]["start_sec"] == 0.0
    assert windows[1]["start_sec"] == 5.0
    assert windows[-1]["end_sec"] == 13.5
    for w in windows:
        assert len(w["audio"]) == int(5.0 * sr)


def test_extract_windowed_spectrograms_from_file(tmp_path: Path):
    sr = 44_100
    duration = 11.0
    audio = np.random.randn(int(duration * sr)).astype(np.float32)
    file_path = tmp_path / "test_track.wav"
    sf.write(file_path, audio, sr)

    windows = extract_windowed_spectrograms(file_path, window_duration=5.0, hop_duration=5.0, sr=sr)
    assert len(windows) >= 2
    for w in windows:
        spec = w["spectrogram"]
        assert spec.shape[0] == 128
        assert not np.isnan(spec).any()
        assert "start_sec" in w and "end_sec" in w


def test_extract_windowed_classical_features_from_file(tmp_path: Path):
    sr = 44_100
    duration = 10.5
    audio = np.random.randn(int(duration * sr)).astype(np.float32)
    file_path = tmp_path / "test_track.wav"
    sf.write(file_path, audio, sr)

    windows = extract_windowed_classical_features(file_path, window_duration=5.0, hop_duration=5.0, sr=sr)
    assert len(windows) >= 2
    for w in windows:
        feats = w["features"]
        assert len(feats) == 42
        assert not np.isnan(feats).any()
