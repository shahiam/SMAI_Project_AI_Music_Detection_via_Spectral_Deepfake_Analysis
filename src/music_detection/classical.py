from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_manifest
from .features import classical_features
from .metrics import classification_metrics, save_metrics


def run(manifest_path: str, output_dir: str) -> dict[str, float]:
    manifest = load_manifest(manifest_path)
    features = np.vstack([classical_features(path) for path in manifest["path"]])
    train = manifest["split"].eq("train").to_numpy()
    test = manifest["split"].eq("test").to_numpy()
    model = make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42))
    model.fit(features[train], manifest.loc[train, "label"])
    probabilities = model.predict_proba(features[test])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(manifest.loc[test, "label"], predictions, probabilities)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "svm.joblib")
    save_metrics(output, metrics, manifest.loc[test, "label"], predictions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.manifest, args.output))
