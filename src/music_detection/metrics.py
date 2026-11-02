from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels, predictions, probabilities) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else float("nan")
    return result


def save_metrics(output_dir: str | Path, metrics: dict[str, float], labels, predictions) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=["human", "AI"])
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
