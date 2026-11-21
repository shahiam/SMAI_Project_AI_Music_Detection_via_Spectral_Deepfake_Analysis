"""Shared track-level evaluation helpers used by both branches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import native_duration
from .metrics import classification_metrics


def build_window_index(
    frame: pd.DataFrame, window_duration: float = 5.0, hop_duration: float = 5.0
) -> pd.DataFrame:
    """One row per (clip, window). ``clip`` is the positional index of the clip in ``frame``."""
    from .features import window_starts

    rows = []
    for position, path in enumerate(frame["path"]):
        for index, start in enumerate(window_starts(native_duration(path), window_duration, hop_duration)):
            rows.append({"clip": position, "window_index": index, "start_sec": start, "path": path})
    return pd.DataFrame(rows)


def aggregate_clip_probabilities(
    windows: pd.DataFrame, window_probabilities: np.ndarray, n_clips: int, how: str = "mean"
) -> np.ndarray:
    """Collapse window probabilities to one probability per clip."""
    series = pd.Series(np.asarray(window_probabilities, dtype=float)).groupby(windows["clip"].to_numpy())
    reduced = series.median() if how == "median" else series.mean()
    return reduced.reindex(range(n_clips)).to_numpy()


def cache_key(path: str, *parts: Any) -> str:
    return hashlib.sha1("|".join([str(path), *map(str, parts)]).encode("utf-8")).hexdigest()


def group_breakdown(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, column: str = "generator"
) -> dict[str, dict[str, float]]:
    """Per-group accuracy: recall for AI groups, true-negative rate for human groups."""
    if column not in frame.columns:
        return {}
    table = frame[[column, "label"]].copy()
    table["probability"] = probabilities
    table["predicted"] = (table["probability"] >= threshold).astype(int)
    result = {}
    for name, group in table.groupby(column):
        result[str(name)] = {
            "n": int(len(group)),
            "accuracy": float((group["predicted"] == group["label"]).mean()),
            "mean_ai_probability": float(group["probability"].mean()),
        }
    return result


def save_predictions(output: Path, split: str, frame: pd.DataFrame, probabilities: np.ndarray) -> None:
    columns = [c for c in ("path", "label", "track_id", "generator", "genre") if c in frame.columns]
    table = frame[columns].copy()
    table["ai_probability"] = probabilities
    table.to_csv(output / f"predictions_{split}.csv", index=False)


def summarize(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = frame["label"].to_numpy()
    return {
        **classification_metrics(labels, (probabilities >= threshold).astype(int), probabilities, threshold),
        "per_generator": group_breakdown(frame, probabilities, threshold),
    }


def sample_rate_baseline(manifest: pd.DataFrame) -> dict[str, Any]:
    """'Dumb' baseline: predict the label from the file's native sample rate only.

    Learns P(AI | native sample rate) on the train split and applies it to test. If this
    scores well, the dataset contains a bandwidth shortcut.
    """
    import soundfile as sf

    table = manifest.copy()
    table["native_sr"] = [sf.info(str(path)).samplerate for path in table["path"]]
    train = table[table["split"] == "train"]
    lookup = train.groupby("native_sr")["label"].mean()
    prior = float(train["label"].mean())
    test = table[table["split"] == "test"]
    probabilities = test["native_sr"].map(lookup).fillna(prior).to_numpy()
    crosstab = pd.crosstab(table["label"].map({0: "human", 1: "ai"}), table["native_sr"])
    return {
        "rule": "P(AI | native sample rate) learned on train split",
        "train_lookup": {str(int(k)): float(v) for k, v in lookup.items()},
        "test": summarize(test, probabilities, 0.5),
        "sample_rate_by_class": json.loads(crosstab.to_json()),
    }


def fit_ensemble(
    classical_dir: str | Path,
    modern_dir: str | Path,
    output_dir: str | Path,
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
) -> dict[str, Any]:
    """Choose the modern/classical blend weight and threshold on validation only, report test.

    Reads predictions_{val,test}.csv written by both training runs. The weight is picked by
    validation ROC-AUC; the threshold by the same criterion the single models use.
    """
    from sklearn.metrics import roc_auc_score

    from .metrics import tune_threshold

    def load(split: str) -> pd.DataFrame:
        classical = pd.read_csv(Path(classical_dir) / f"predictions_{split}.csv")
        modern = pd.read_csv(Path(modern_dir) / f"predictions_{split}.csv")
        merged = classical.merge(modern[["path", "ai_probability"]], on="path", suffixes=("_classical", "_modern"))
        if len(merged) != len(classical):
            raise ValueError(f"{split}: classical and modern predictions cover different clips")
        return merged

    val, test = load("val"), load("test")
    weights = np.round(np.linspace(0.0, 1.0, 21), 2)

    def blend(frame: pd.DataFrame, weight: float) -> np.ndarray:
        return weight * frame["ai_probability_modern"].to_numpy() + (1 - weight) * frame[
            "ai_probability_classical"
        ].to_numpy()

    val_auc = {float(w): float(roc_auc_score(val["label"], blend(val, w))) for w in weights}
    best_weight = max(val_auc, key=lambda w: (round(val_auc[w], 4), w))
    tuning = tune_threshold(val["label"].to_numpy(), blend(val, best_weight), criterion=criterion, target_fpr=target_fpr)
    threshold = float(tuning["optimal_threshold"])
    result = {
        "modern_weight": best_weight,
        "classical_weight": round(1.0 - best_weight, 2),
        "threshold": threshold,
        "criterion": criterion,
        "target_fpr": target_fpr,
        "val_auc_by_modern_weight": val_auc,
        "val": {k: v for k, v in tuning["metrics_at_optimal"].items()},
        "test": summarize(test, blend(test, best_weight), threshold),
        "classical_dir": str(classical_dir),
        "modern_dir": str(modern_dir),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ensemble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
