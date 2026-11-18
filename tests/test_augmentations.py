from pathlib import Path
import numpy as np
import pytest

from music_detection.augmentations import (
    AudioAugmentor,
    additive_noise,
    mp3_compression,
    pitch_shift,
    spec_augment,
    time_stretch,
    volume_change,
)


@pytest.fixture
def sample_audio():
    sr = 44_100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


@pytest.fixture
def sample_spectrogram():
    rng = np.random.default_rng(42)
    return rng.normal(-40.0, 10.0, size=(128, 256)).astype(np.float32)


def test_volume_change_preserves_shape_and_range(sample_audio):
    audio, _ = sample_audio
    aug = volume_change(audio, gain_db_range=(-6.0, 6.0))
    assert aug.shape == audio.shape
    assert aug.dtype == np.float32
    assert np.max(np.abs(aug)) <= 1.0
    assert not np.isnan(aug).any()


def test_additive_noise_preserves_shape_and_adds_variation(sample_audio):
    audio, _ = sample_audio
    aug = additive_noise(audio, snr_db_range=(20.0, 30.0))
    assert aug.shape == audio.shape
    assert aug.dtype == np.float32
    assert not np.array_equal(audio, aug)
    assert not np.isnan(aug).any()


def test_mp3_compression_preserves_shape_and_stability(sample_audio):
    audio, sr = sample_audio
    aug = mp3_compression(audio, sr=sr)
    assert aug.shape == audio.shape
    assert aug.dtype == np.float32
    assert not np.isnan(aug).any()


def test_pitch_shift_preserves_shape(sample_audio):
    audio, sr = sample_audio
    aug = pitch_shift(audio, sr=sr, steps_range=(1.0, 2.0))
    assert aug.shape == audio.shape
    assert aug.dtype == np.float32
    assert not np.isnan(aug).any()


def test_time_stretch_preserves_shape(sample_audio):
    audio, _ = sample_audio
    aug = time_stretch(audio, rate_range=(0.95, 1.05))
    assert aug.shape == audio.shape
    assert aug.dtype == np.float32
    assert not np.isnan(aug).any()


def test_spec_augment_masks_frequency_and_time(sample_spectrogram):
    aug = spec_augment(
        sample_spectrogram,
        max_freq_mask=8,
        max_time_mask=12,
        num_freq_masks=2,
        num_time_masks=2,
        mask_value=-80.0,
    )
    assert aug.shape == sample_spectrogram.shape
    assert aug.dtype == np.float32
    # Check that mask value is present
    assert np.isclose(aug, -80.0).any()


def test_audio_augmentor_pipeline(sample_audio, sample_spectrogram):
    audio, sr = sample_audio
    augmentor = AudioAugmentor(
        p_volume=1.0,
        p_noise=1.0,
        p_mp3=1.0,
        p_pitch=0.5,
        p_stretch=0.5,
        p_spec_augment=1.0,
        seed=42,
    )

    aug_audio = augmentor.apply_audio(audio, sr=sr)
    assert aug_audio.shape == audio.shape
    assert not np.isnan(aug_audio).any()

    aug_spec = augmentor.apply_spectrogram(sample_spectrogram)
    assert aug_spec.shape == sample_spectrogram.shape
    assert not np.isnan(aug_spec).any()
