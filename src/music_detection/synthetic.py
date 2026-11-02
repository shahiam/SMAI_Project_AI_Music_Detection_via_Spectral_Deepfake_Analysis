from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def create_demo(output_dir: str, samples_per_class: int = 6, duration: float = 1.0) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    rate = 44_100
    time = np.linspace(0, duration, int(rate * duration), endpoint=False)
    for label in (0, 1):
        for index in range(samples_per_class):
            frequency = 220 + index * 13 + label * 90
            signal = 0.25 * np.sin(2 * np.pi * frequency * time)
            if label:
                signal += 0.04 * np.sign(np.sin(2 * np.pi * (frequency * 2.5) * time))
            path = output / f"{'ai' if label else 'human'}_{index}.wav"
            sf.write(path, signal.astype(np.float32), rate)
            split = "test" if index == samples_per_class - 1 else ("val" if index == samples_per_class - 2 else "train")
            rows.append({"path": path.name, "label": label, "track_id": path.stem, "split": split})
    manifest = output / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(create_demo(args.output))
