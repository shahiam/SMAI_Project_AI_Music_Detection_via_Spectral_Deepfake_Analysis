from pathlib import Path

import pandas as pd
import pytest

from music_detection.data import load_manifest


def test_manifest_resolves_relative_paths(tmp_path: Path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "clip.wav", "label": 0, "track_id": "clip", "split": "test"}]).to_csv(manifest, index=False)
    assert load_manifest(manifest).iloc[0]["path"] == str(audio.resolve())


def test_manifest_rejects_cross_split_track(tmp_path: Path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"path": "clip.wav", "label": 0, "track_id": "same", "split": "train"},
        {"path": "clip.wav", "label": 0, "track_id": "same", "split": "test"},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="multiple splits"):
        load_manifest(manifest)
