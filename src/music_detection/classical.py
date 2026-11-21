from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_manifest, load_segment, native_duration
from .evaluation import (
    aggregate_clip_probabilities,
    build_window_index,
    save_predictions,
    summarize,
)
from .features import (
    BANDLIMITED_SAMPLE_RATE,
    SAMPLE_RATE,
    _pad_audio,
    classical_features,
    extract_windowed_classical_features,
)
from .metrics import save_metrics, tune_threshold


def _segment_features(path: str, sample_rate: int, start: float, duration: float) -> np.ndarray:
    audio = _pad_audio(load_segment(path, sample_rate, start, duration), int(duration * sample_rate))
    return classical_features(audio, sr=sample_rate)


def _window_features(windows: pd.DataFrame, sample_rate: int, duration: float, jobs: int) -> np.ndarray:
    rows = Parallel(n_jobs=jobs, batch_size=32)(
        delayed(_segment_features)(path, sample_rate, start, duration)
        for path, start in zip(windows["path"], windows["start_sec"])
    )
    return np.vstack(rows)


def _random_crop_windows(frame: pd.DataFrame, duration: float, crops: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for position, path in enumerate(frame["path"]):
        latest = max(0.0, native_duration(path) - duration)
        for _ in range(crops):
            rows.append({"clip": position, "path": path, "start_sec": float(rng.uniform(0, latest)) if latest else 0.0})
    return pd.DataFrame(rows)


def run(
    manifest_path: str,
    output_dir: str,
    threshold_criterion: str = "low_fpr",
    target_fpr: float = 0.02,
    sample_rate: int = BANDLIMITED_SAMPLE_RATE,
    window_duration: float = 5.0,
    train_crops: int = 2,
    jobs: int = -1,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the MFCC + spectral-centroid SVM (and a Random Forest for comparison).

    Training uses ``train_crops`` random windows per training clip. Validation and test are
    scored at track level: the mean probability over every window of each clip, exactly as
    the dashboard does.
    """
    manifest = load_manifest(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = {split: manifest[manifest["split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}

    train_windows = _random_crop_windows(frames["train"], window_duration, train_crops, seed)
    x_train = _window_features(train_windows, sample_rate, window_duration, jobs)
    y_train = frames["train"]["label"].to_numpy()[train_windows["clip"].to_numpy()]
    print(f"train features: {x_train.shape}", flush=True)

    eval_windows, eval_features = {}, {}
    for split in ("val", "test"):
        eval_windows[split] = build_window_index(frames[split], window_duration, window_duration)
        eval_features[split] = _window_features(eval_windows[split], sample_rate, window_duration, jobs)
        print(f"{split} window features: {eval_features[split].shape}", flush=True)

    models = {
        "svm": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", min_samples_leaf=2, n_jobs=jobs, random_state=seed
        ),
    }
    results: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)

        def clip_probs(split: str) -> np.ndarray:
            window_probs = model.predict_proba(eval_features[split])[:, 1]
            return aggregate_clip_probabilities(eval_windows[split], window_probs, len(frames[split]))

        val_probs, test_probs = clip_probs("val"), clip_probs("test")
        tuning = tune_threshold(
            frames["val"]["label"].to_numpy(), val_probs, criterion=threshold_criterion, target_fpr=target_fpr
        )
        threshold = float(tuning["optimal_threshold"])
        model.threshold_ = threshold
        model.sample_rate_ = sample_rate
        model.window_duration_ = window_duration
        filename = "svm.joblib" if name == "svm" else "random_forest.joblib"
        joblib.dump(model, output / filename)

        metrics = summarize(frames["test"], test_probs, threshold)
        results[name] = {
            **metrics,
            "default_0_5": summarize(frames["test"], test_probs, 0.5),
            "tuned_threshold": threshold,
        }
        if name == "svm":
            save_predictions(output, "val", frames["val"], val_probs)
            save_predictions(output, "test", frames["test"], test_probs)
            primary = metrics
            primary_preds = (test_probs >= threshold).astype(int)
            primary_threshold = threshold
        else:
            feature_names = [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
            feature_names += ["centroid_mean", "centroid_std"]
            order = np.argsort(model.feature_importances_)[::-1][:10]
            results[name]["top_features"] = {feature_names[i]: float(model.feature_importances_[i]) for i in order}
        print(f"{name}: f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} threshold={threshold:.3f}", flush=True)

    config = {
        "sample_rate": sample_rate,
        "window_duration": window_duration,
        "train_crops": train_crops,
        "threshold_criterion": threshold_criterion,
        "target_fpr": target_fpr,
        "seed": seed,
        "manifest": str(manifest_path),
        "evaluation": "track-level: mean probability over all windows of each clip",
    }
    full_metrics = {**results["svm"], "random_forest": results["random_forest"], "config": config}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_metrics(output, full_metrics, frames["test"]["label"].to_numpy(), primary_preds, threshold=primary_threshold)
    return primary


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
    sample_rate = int(getattr(model, "sample_rate_", SAMPLE_RATE))

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
                    sr=sample_rate,
                )
            except Exception:
                win_feats = None

        if win_feats is None:
            # Fallback for mocked tests or synthetic clips
            feats = classical_features(path, sr=sample_rate)
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
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
