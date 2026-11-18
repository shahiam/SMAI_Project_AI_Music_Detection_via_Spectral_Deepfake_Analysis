import numpy as np
import pytest

from music_detection.metrics import classification_metrics, tune_threshold


def test_classification_metrics_contains_all_keys():
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 0, 1]
    probabilities = [0.1, 0.7, 0.3, 0.9]

    metrics = classification_metrics(labels, predictions, probabilities, threshold=0.5)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "threshold"):
        assert key in metrics
    assert metrics["false_positive_rate"] == 0.5


def test_tune_threshold_low_fpr_reduces_false_positives():
    np.random.seed(42)
    # 50 human, 50 AI
    labels = np.array([0] * 50 + [1] * 50)
    # Probabilities with overlap around 0.5
    human_probs = np.random.uniform(0.0, 0.55, 50)
    ai_probs = np.random.uniform(0.45, 1.0, 50)
    probs = np.concatenate([human_probs, ai_probs])

    result = tune_threshold(labels, probs, criterion="low_fpr", target_fpr=0.04)

    assert "optimal_threshold" in result
    assert result["optimal_threshold"] > 0.50
    assert result["metrics_at_optimal"]["false_positive_rate"] <= 0.04
    assert result["metrics_at_optimal"]["false_positives"] < result["metrics_at_0_5"]["false_positives"]


def test_tune_threshold_f1_maximizes_f1():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.45, 0.48, 0.52, 0.7, 0.8, 0.9])

    result = tune_threshold(labels, probs, criterion="f1")
    assert result["metrics_at_optimal"]["f1"] >= result["metrics_at_0_5"]["f1"]


def test_tune_threshold_cost_sensitive_penalizes_false_positives():
    labels = np.array([0] * 20 + [1] * 20)
    human_probs = np.random.uniform(0.1, 0.6, 20)
    ai_probs = np.random.uniform(0.5, 0.95, 20)
    probs = np.concatenate([human_probs, ai_probs])

    # 5x penalty on FP
    result = tune_threshold(labels, probs, criterion="cost_sensitive", cost_fp=5.0, cost_fn=1.0)
    assert result["metrics_at_optimal"]["false_positives"] <= result["metrics_at_0_5"]["false_positives"]
