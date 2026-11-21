from pathlib import Path

import numpy as np

from music_detection.classical import predict_files


def test_predict_files_returns_batch_schema(tmp_path: Path, monkeypatch):
    class FakeModel:
        def predict_proba(self, features):
            assert features.shape == (2, 42)
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr("music_detection.classical.joblib.load", lambda path: FakeModel())
    monkeypatch.setattr(
        "music_detection.classical.classical_features",
        lambda path, **kwargs: np.zeros(42, dtype=np.float32),
    )

    results = predict_files(model_path, [tmp_path / "one.wav", tmp_path / "two.wav"])
    assert list(results["prediction"]) == ["Human", "AI-generated"]
    assert list(results.columns) == ["filename", "ai_probability", "prediction"]
