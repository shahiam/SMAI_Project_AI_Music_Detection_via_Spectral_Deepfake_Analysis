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


def tune_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
    cost_fp: float = 3.0,
    cost_fn: float = 1.0,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 181,
) -> dict[str, Any]:
    """Tune binary decision threshold on validation predictions.
    
    Supported criteria:
    - 'low_fpr': Finds threshold that keeps FPR <= target_fpr (e.g. 2%) while maximizing Recall.
    - 'f1': Maximizes F1-score across validation data.
    - 'youden': Maximizes Youden's J statistic (Sensitivity + Specificity - 1).
    - 'cost_sensitive': Minimizes cost = cost_fp * FP + cost_fn * FN.
    - 'balanced_accuracy': Maximizes average of TPR and TNR.
    """
    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(probabilities, dtype=float)

    if len(set(labels_arr)) < 2:
        return {
            "optimal_threshold": 0.5,
            "criterion": criterion,
            "metrics_at_0_5": classification_metrics(labels_arr, (probs_arr >= 0.5).astype(int), probs_arr, 0.5),
            "metrics_at_optimal": classification_metrics(labels_arr, (probs_arr >= 0.5).astype(int), probs_arr, 0.5),
            "warning": "Fewer than 2 classes present; defaulting to 0.5 threshold.",
        }

    threshold_grid = np.linspace(min_threshold, max_threshold, num_steps)
    records = []

    for t in threshold_grid:
        preds = (probs_arr >= t).astype(int)
        cm = confusion_matrix(labels_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        p = float(precision_score(labels_arr, preds, zero_division=0))
        r = float(recall_score(labels_arr, preds, zero_division=0))
        f1 = float(f1_score(labels_arr, preds, zero_division=0))
        cost = float(cost_fp * fp + cost_fn * fn)
        youden = tpr - fpr
        balanced_acc = 0.5 * (tpr + (1.0 - fpr))

        records.append(
            {
                "threshold": float(t),
                "precision": p,
                "recall": r,
                "f1": f1,
                "fpr": fpr,
                "fnr": 1.0 - tpr,
                "cost": cost,
                "youden": youden,
                "balanced_accuracy": balanced_acc,
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            }
        )

    # Select optimal threshold
    if criterion in ("low_fpr", "target_fpr"):
        # Select candidates with FPR <= target_fpr
        valid_candidates = [r for r in records if r["fpr"] <= target_fpr]
        if valid_candidates:
            # Among valid, maximize Recall, then F1, then higher threshold
            best = max(valid_candidates, key=lambda r: (r["recall"], r["f1"], r["threshold"]))
        else:
            # Fallback to candidate with absolute lowest FPR
            best = min(records, key=lambda r: (r["fpr"], -r["recall"]))
    elif criterion == "f1":
        best = max(records, key=lambda r: (r["f1"], -r["fpr"], r["threshold"]))
    elif criterion == "youden":
        best = max(records, key=lambda r: (r["youden"], -r["fpr"]))
    elif criterion == "cost_sensitive":
        best = min(records, key=lambda r: (r["cost"], -r["f1"]))
    elif criterion == "balanced_accuracy":
        best = max(records, key=lambda r: (r["balanced_accuracy"], -r["fpr"]))
    else:
        raise ValueError(f"Unknown threshold tuning criterion: {criterion}")

    optimal_threshold = float(best["threshold"])
    metrics_0_5 = classification_metrics(labels_arr, (probs_arr >= 0.5).astype(int), probs_arr, 0.5)
    metrics_opt = classification_metrics(labels_arr, (probs_arr >= optimal_threshold).astype(int), probs_arr, optimal_threshold)

    return {
        "optimal_threshold": optimal_threshold,
        "criterion": criterion,
        "target_fpr": target_fpr,
        "metrics_at_0_5": metrics_0_5,
        "metrics_at_optimal": metrics_opt,
        "fp_reduction": metrics_0_5["false_positives"] - metrics_opt["false_positives"],
        "fpr_reduction": metrics_0_5["false_positive_rate"] - metrics_opt["false_positive_rate"],
        "threshold_curve": records,
    }


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
