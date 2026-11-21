from __future__ import annotations

import io
from typing import Sequence
import numpy as np
import soundfile as sf
import librosa
from scipy import signal


def volume_change(
    y: np.ndarray,
    gain_db_range: tuple[float, float] = (-6.0, 6.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly scale audio amplitude within a dB range."""
    rng = rng or np.random.default_rng()
    gain_db = rng.uniform(gain_db_range[0], gain_db_range[1])
    gain_linear = 10.0 ** (gain_db / 20.0)
    return np.clip(y * gain_linear, -1.0, 1.0).astype(np.float32)


def additive_noise(
    y: np.ndarray,
    snr_db_range: tuple[float, float] = (20.0, 40.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise at a random signal-to-noise ratio (SNR) in dB."""
    rng = rng or np.random.default_rng()
    snr_db = rng.uniform(snr_db_range[0], snr_db_range[1])
    signal_power = np.mean(y**2) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return np.clip(y + noise, -1.0, 1.0).astype(np.float32)


def mp3_compression(
    y: np.ndarray,
    sr: int = 44_100,
    bitrates: Sequence[int] = (64, 128, 192, 256, 320),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply MP3 lossy compression artifacts using in-memory encoding or lowpass filtering."""
    rng = rng or np.random.default_rng()
    orig_len = len(y)

    # 1. Attempt genuine MP3 compression via soundfile in-memory buffer
    try:
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="MP3")
        buf.seek(0)
        compressed, comp_sr = sf.read(buf, dtype="float32")
        if comp_sr != sr:
            compressed = librosa.resample(compressed, orig_sr=comp_sr, target_sr=sr)
        if len(compressed) >= orig_len:
            return compressed[:orig_len].astype(np.float32)
        return np.pad(compressed, (0, orig_len - len(compressed)), mode="reflect").astype(np.float32)
    except Exception:
        pass

    # 2. High-fidelity fallback: simulate MP3 codec spectral notch / high-frequency roll-off
    cutoff_freq = rng.uniform(14_000, 17_500)
    nyquist = 0.5 * sr
    normalized_cutoff = min(0.95, cutoff_freq / nyquist)
    b, a = signal.butter(6, normalized_cutoff, btype="low")
    filtered = signal.filtfilt(b, a, y)
    quantized = np.round(filtered * 256.0) / 256.0
    return np.clip(quantized, -1.0, 1.0).astype(np.float32)


def pitch_shift(
    y: np.ndarray,
    sr: int = 44_100,
    steps_range: tuple[float, float] = (-2.0, 2.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly pitch shift audio by semitone intervals."""
    rng = rng or np.random.default_rng()
    steps = rng.uniform(steps_range[0], steps_range[1])
    if abs(steps) < 0.2:
        return y.astype(np.float32)
    shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=steps)
    if len(shifted) >= len(y):
        return shifted[:len(y)].astype(np.float32)
    return np.pad(shifted, (0, len(y) - len(shifted)), mode="reflect").astype(np.float32)


def time_stretch(
    y: np.ndarray,
    rate_range: tuple[float, float] = (0.92, 1.08),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly time-stretch audio while keeping pitch unchanged."""
    rng = rng or np.random.default_rng()
    rate = rng.uniform(rate_range[0], rate_range[1])
    if abs(rate - 1.0) < 0.02:
        return y.astype(np.float32)
    stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    target_len = len(y)
    if len(stretched) >= target_len:
        return stretched[:target_len].astype(np.float32)
    return np.pad(stretched, (0, target_len - len(stretched)), mode="reflect").astype(np.float32)


def spec_augment(
    spectrogram: np.ndarray,
    max_freq_mask: int = 16,
    max_time_mask: int = 24,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    mask_value: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment (frequency and time masking) to a 2D log-Mel spectrogram."""
    rng = rng or np.random.default_rng()
    aug = spectrogram.copy()
    n_mels, time_steps = aug.shape
    fill_val = mask_value if mask_value is not None else float(aug.min())

    # Frequency masking
    for _ in range(num_freq_masks):
        if n_mels <= 2:
            break
        f = int(rng.integers(1, min(max_freq_mask, n_mels // 2) + 1))
        f0 = int(rng.integers(0, max(1, n_mels - f)))
        aug[f0 : f0 + f, :] = fill_val

    # Time masking
    for _ in range(num_time_masks):
        if time_steps <= 2:
            break
        t = int(rng.integers(1, min(max_time_mask, time_steps // 2) + 1))
        t0 = int(rng.integers(0, max(1, time_steps - t)))
        aug[:, t0 : t0 + t] = fill_val

    return aug.astype(np.float32)


class AudioAugmentor:
    """Chained audio and spectrogram data augmentor for deepfake detection training."""

    def __init__(
        self,
        p_volume: float = 0.5,
        p_noise: float = 0.5,
        p_mp3: float = 0.5,
        p_pitch: float = 0.35,
        p_stretch: float = 0.35,
        p_spec_augment: float = 0.5,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        snr_db_range: tuple[float, float] = (20.0, 40.0),
        pitch_steps_range: tuple[float, float] = (-2.0, 2.0),
        stretch_rate_range: tuple[float, float] = (0.92, 1.08),
        seed: int | None = None,
    ):
        self.p_volume = p_volume
        self.p_noise = p_noise
        self.p_mp3 = p_mp3
        self.p_pitch = p_pitch
        self.p_stretch = p_stretch
        self.p_spec_augment = p_spec_augment
        self.gain_db_range = gain_db_range
        self.snr_db_range = snr_db_range
        self.pitch_steps_range = pitch_steps_range
        self.stretch_rate_range = stretch_rate_range
        self.rng = np.random.default_rng(seed)

    def apply_audio(self, y: np.ndarray, sr: int = 44_100) -> np.ndarray:
        """Apply random audio-domain augmentations sequentially."""
        out = y.copy().astype(np.float32)

        if self.rng.random() < self.p_volume:
            out = volume_change(out, self.gain_db_range, rng=self.rng)

        if self.rng.random() < self.p_noise:
            out = additive_noise(out, self.snr_db_range, rng=self.rng)

        if self.rng.random() < self.p_mp3:
            out = mp3_compression(out, sr=sr, rng=self.rng)

        if self.rng.random() < self.p_pitch:
            out = pitch_shift(out, sr=sr, steps_range=self.pitch_steps_range, rng=self.rng)

        if self.rng.random() < self.p_stretch:
            out = time_stretch(out, rate_range=self.stretch_rate_range, rng=self.rng)

        return out

    def apply_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Apply random spectrogram-domain augmentations (SpecAugment)."""
        if self.rng.random() < self.p_spec_augment:
            return spec_augment(spec, rng=self.rng)
        return spec
