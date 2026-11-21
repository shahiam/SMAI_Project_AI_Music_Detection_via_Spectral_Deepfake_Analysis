"""Re-tune the decision threshold of an already trained model on one manifest split."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import torch

from ..data.manifest import load_manifest
from ..evaluation.track_level import aggregate_clip_probabilities, build_window_index
from ..features.spectral import SAMPLE_RATE
from ..training.classical import _window_features
from ..training.modern import build_model, track_probabilities
from .thresholds import tune_threshold


def calibrate_svm_threshold(
    model_path: str | Path,
    manifest_path: str | Path,
    split: str = "val",
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
) -> dict[str, Any]:
    """Re-tune the decision threshold of an existing classical model on track-level scores."""
    manifest = load_manifest(manifest_path)
    frame = manifest[manifest["split"].eq(split)]
    if frame.empty:
        frame = manifest[manifest["split"].eq("train")]
    frame = frame.reset_index(drop=True)

    loaded = joblib.load(model_path)
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    sample_rate = int(getattr(model, "sample_rate_", SAMPLE_RATE))
    duration = float(getattr(model, "window_duration_", 5.0))

    windows = build_window_index(frame, duration, duration)
    window_probs = model.predict_proba(_window_features(windows, sample_rate, duration, -1))[:, 1]
    probs = aggregate_clip_probabilities(windows, window_probs, len(frame))
    tuning = tune_threshold(frame["label"].to_numpy(), probs, criterion=criterion, target_fpr=target_fpr)
    model.threshold_ = float(tuning["optimal_threshold"])
    joblib.dump(model, model_path)
    return tuning


def calibrate_checkpoint_threshold(
    model_path: str | Path,
    manifest_path: str | Path,
    split: str = "val",
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
    batch_size: int = 64,
    workers: int = 0,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-tune the decision threshold of an existing checkpoint on track-level scores."""
    manifest = load_manifest(manifest_path)
    frame = manifest[manifest["split"] == split]
    if frame.empty:
        frame = manifest[manifest["split"] == "train"]
    frame = frame.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    sample_rate = int(checkpoint.get("sample_rate", SAMPLE_RATE))
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    cache = Path(cache_dir) if cache_dir else Path(model_path).parent / "window_cache" / split
    probabilities = track_probabilities(model, frame, device, sample_rate, cache, batch_size, workers)
    tuning = tune_threshold(frame["label"].to_numpy(), probabilities, criterion=criterion, target_fpr=target_fpr)
    checkpoint["threshold"] = float(tuning["optimal_threshold"])
    checkpoint["threshold_tuning"] = {k: v for k, v in tuning.items() if k != "threshold_curve"}
    torch.save(checkpoint, model_path)
    return tuning
