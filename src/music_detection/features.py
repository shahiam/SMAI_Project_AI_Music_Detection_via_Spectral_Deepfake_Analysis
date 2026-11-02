from __future__ import annotations

import numpy as np
import librosa

from .data import load_audio

SAMPLE_RATE = 44_100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def log_mel_spectrogram(path: str) -> np.ndarray:
    audio, _ = load_audio(path, SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def classical_features(path: str) -> np.ndarray:
    audio, _ = load_audio(path, SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), centroid.mean(axis=1), centroid.std(axis=1)])
