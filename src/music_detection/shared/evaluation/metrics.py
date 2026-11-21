"""Binary classification metrics and the metrics.json / confusion-matrix files of a run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute comprehensive binary classification metrics including FPR and confusion counts."""
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(predictions, dtype=int)
    probs_arr = np.asarray(probabilities, dtype=float)

    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    result = {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "false_positive_rate": fpr,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "threshold": float(threshold),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels_arr, probs_arr))
        if len(set(labels_arr)) == 2
        else float("nan")
    )
    return result


def save_metrics(
    output_dir: str | Path,
    metrics: dict[str, Any],
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    threshold: float = 0.5,
    filename: str = "metrics.json",
    image_name: str = "confusion_matrix.png",
) -> None:
    """Save metrics JSON and confusion matrix plot with labeled threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=["Human", "AI"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(output_dir / image_name, dpi=160)
    plt.close()
