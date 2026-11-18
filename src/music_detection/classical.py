from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_manifest
from .features import SAMPLE_RATE, classical_features, extract_windowed_classical_features
from .metrics import classification_metrics, save_metrics, tune_threshold


def run(
    manifest_path: str,
    output_dir: str,
    threshold_criterion: str = "low_fpr",
    target_fpr: float = 0.02,
) -> dict[str, float]:
    """Train classical SVM baseline and tune decision threshold on validation set."""
    manifest = load_manifest(manifest_path)
    features = np.vstack([classical_features(path) for path in manifest["path"]])
    train = manifest["split"].eq("train").to_numpy()
    val = manifest["split"].eq("val").to_numpy()
    test = manifest["split"].eq("test").to_numpy()

    model = make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42))
    model.fit(features[train], manifest.loc[train, "label"])

    # Threshold calibration on validation split
    tuning_split = val if val.any() else train
    val_probs = model.predict_proba(features[tuning_split])[:, 1]
    tuning = tune_threshold(
        manifest.loc[tuning_split, "label"].to_numpy(),
        val_probs,
        criterion=threshold_criterion,
        target_fpr=target_fpr,
    )
    tuned_threshold = float(tuning["optimal_threshold"])
    model.threshold_ = tuned_threshold

    # Evaluate on held-out test split
    test_probs = model.predict_proba(features[test])[:, 1]
    test_labels = manifest.loc[test, "label"].to_numpy()
    metrics_0_5 = classification_metrics(test_labels, (test_probs >= 0.5).astype(int), test_probs, 0.5)
    metrics_tuned = classification_metrics(
        test_labels,
        (test_probs >= tuned_threshold).astype(int),
        test_probs,
        tuned_threshold,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "svm.joblib")

    full_metrics = {
        **metrics_tuned,
        "default_0_5": metrics_0_5,
        "tuned": metrics_tuned,
        "tuned_threshold": tuned_threshold,
        "tuning_criterion": threshold_criterion,
    }
    save_metrics(
        output,
        full_metrics,
        test_labels,
        (test_probs >= tuned_threshold).astype(int),
        threshold=tuned_threshold,
    )
    return metrics_tuned


def calibrate_svm_threshold(
    model_path: str | Path,
    manifest_path: str | Path,
    split: str = "val",
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
) -> dict[str, Any]:
    """Calibrate decision threshold on validation set for an existing SVM artifact."""
    manifest = load_manifest(manifest_path)
    mask = manifest["split"].eq(split).to_numpy()
    if not mask.any():
        mask = manifest["split"].eq("train").to_numpy()

    loaded = joblib.load(model_path)
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

    features = np.vstack([classical_features(path) for path in manifest.loc[mask, "path"]])
    probs = model.predict_proba(features)[:, 1]
    labels = manifest.loc[mask, "label"].to_numpy()

    tuning = tune_threshold(labels, probs, criterion=criterion, target_fpr=target_fpr)
    model.threshold_ = float(tuning["optimal_threshold"])
    joblib.dump(model, model_path)
    return tuning


def predict_files(
    model_path: str | Path,
    audio_paths: list[str | Path],
    window_duration: float = 5.0,
    hop_duration: float = 5.0,
    aggregation: str = "mean",
    threshold: float | None = None,
    detailed: bool = False,
) -> pd.DataFrame:
    """Predict human/AI labels for uploaded audio files using SVM with full-track window aggregation."""
    if not audio_paths:
        raise ValueError("At least one audio file is required.")

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        model_threshold = loaded.get("threshold", 0.5)
    else:
        model = loaded
        model_threshold = getattr(model, "threshold_", 0.5)

    cutoff = float(threshold) if threshold is not None else float(model_threshold)

    filenames = []
    file_window_ranges = []
    file_segments_list = []
    all_features = []

    for path in audio_paths:
        path_obj = Path(path)
        filenames.append(path_obj.name)

        # Attempt full-track windowed extraction if real file exists on disk
        win_feats = None
        if path_obj.is_file():
            try:
                win_feats = extract_windowed_classical_features(
                    path_obj,
                    window_duration=window_duration,
                    hop_duration=hop_duration,
                    sr=SAMPLE_RATE,
                )
            except Exception:
                win_feats = None

        if win_feats is None:
            # Fallback for mocked tests or synthetic clips
            feats = classical_features(path)
            win_feats = [{"features": feats, "start_sec": 0.0, "end_sec": window_duration, "window_index": 0}]

        start_idx = len(all_features)
        for w in win_feats:
            all_features.append(w["features"])
        end_idx = len(all_features)

        file_window_ranges.append((start_idx, end_idx))
        file_segments_list.append(win_feats)

    # Batch prediction across all windows
    feature_matrix = np.vstack(all_features)
    all_window_probs = model.predict_proba(feature_matrix)[:, 1]

    aggregated_probabilities = []
    predictions = []
    confidences = []
    flagged_window_counts = []
    total_window_counts = []
    all_segments = []

    for (start_idx, end_idx), win_feats in zip(file_window_ranges, file_segments_list):
        window_probs = all_window_probs[start_idx:end_idx]

        if aggregation == "mean":
            agg_prob = float(np.mean(window_probs))
        elif aggregation == "median":
            agg_prob = float(np.median(window_probs))
        elif aggregation == "max":
            agg_prob = float(np.max(window_probs))
        elif aggregation == "min":
            agg_prob = float(np.min(window_probs))
        else:
            agg_prob = float(np.mean(window_probs))

        aggregated_probabilities.append(agg_prob)
        is_ai = bool(agg_prob >= cutoff)
        predictions.append("AI-generated" if is_ai else "Human")

        if agg_prob >= cutoff:
            conf = (agg_prob - cutoff) / max(1.0 - cutoff, 1e-6)
        else:
            conf = (cutoff - agg_prob) / max(cutoff, 1e-6)
        confidences.append(float(np.clip(conf, 0.0, 1.0)))

        flagged_count = int(np.sum(window_probs >= cutoff))
        flagged_window_counts.append(flagged_count)
        total_window_counts.append(len(window_probs))

        segments = [
            {
                "window_index": i,
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "probability": float(window_probs[i]),
                "flagged": bool(window_probs[i] >= cutoff),
            }
            for i, w in enumerate(win_feats)
        ]
        all_segments.append(segments)

    probabilities_array = np.asarray(aggregated_probabilities)
    if not detailed:
        return pd.DataFrame(
            {
                "filename": filenames,
                "ai_probability": probabilities_array,
                "prediction": predictions,
            }
        )

    return pd.DataFrame(
        {
            "filename": filenames,
            "ai_probability": probabilities_array,
            "prediction": predictions,
            "confidence": np.asarray(confidences),
            "flagged_windows": flagged_window_counts,
            "total_windows": total_window_counts,
            "threshold": [cutoff] * len(filenames),
            "segments": all_segments,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-criterion", default="low_fpr", choices=["low_fpr", "f1", "youden", "cost_sensitive"])
    parser.add_argument("--target-fpr", type=float, default=0.02)
    args = parser.parse_args()
    print(run(args.manifest, args.output, threshold_criterion=args.threshold_criterion, target_fpr=args.target_fpr))
