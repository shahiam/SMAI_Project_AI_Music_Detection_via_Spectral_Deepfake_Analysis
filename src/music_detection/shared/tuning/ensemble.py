"""Validated ensemble: choose the modern/classical blend weight and threshold on validation only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.track_level import summarize
from .thresholds import tune_threshold


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
