"""Track-level evaluation shared by both branches: window index, aggregation, summaries, predictions."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.audio import native_duration
from ..features.spectral import window_starts
from .metrics import classification_metrics


def build_window_index(
    frame: pd.DataFrame, window_duration: float = 5.0, hop_duration: float = 5.0
) -> pd.DataFrame:
    """One row per (clip, window). ``clip`` is the positional index of the clip in ``frame``."""
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
