"""Classical branch: MFCC + spectral-centroid features -> SVM (plus a Random Forest for comparison).

Training uses random 5 s crops per clip; validation and test are scored at track level.
"""
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

from ..data.audio import load_segment, native_duration
from ..data.manifest import load_manifest
from ..evaluation.metrics import save_metrics
from ..evaluation.track_level import aggregate_clip_probabilities, build_window_index, save_predictions, summarize
from ..features.spectral import BANDLIMITED_SAMPLE_RATE, _pad_audio, classical_features
from ..tuning.thresholds import tune_threshold


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
