from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import librosa
import pandas as pd


GENERATOR_PREFIXES = ("AMG_", "MGPT_", "MG_")


def _generator(filename: str, label: int) -> str:
    if not label:
        return "human"
    for prefix in GENERATOR_PREFIXES:
        if filename.startswith(prefix):
            return prefix[:-1]
    return "unknown_ai"


def _genre(filename: str) -> str:
    name = Path(filename).stem
    for prefix in GENERATOR_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return re.sub(r"\d+$", "", name).rstrip("_") or "unknown"


def _split(track_id: str) -> str:
    bucket = int(hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def build_manifest(dataset_root: str | Path, output_path: str | Path, include_duration: bool = False) -> Path:
    """Build a deterministic, track-aware M6 manifest from an extracted archive."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"M6 directory does not exist: {root}")
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found below {root}")

    rows = []
    for path in files:
        relative = path.relative_to(root)
        parts = {part.lower() for part in relative.parts}
        label = int("ai" in parts)
        rows.append(
            {
                "path": str(relative),
                "label": label,
                "track_id": path.stem,
                "split": _split(path.stem),
                "generator": _generator(path.name, label),
                "genre": _genre(path.name),
            }
        )
        if include_duration:
            rows[-1]["duration_seconds"] = round(float(librosa.get_duration(path=str(path))), 3)

    frame = pd.DataFrame(rows)
    if set(frame["label"]) != {0, 1}:
        raise ValueError("The extracted M6 directory must contain both AI and human WAV files.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame["path"] = frame["path"].map(
        lambda value: os.path.relpath(root / value, output.parent)
    )
    frame.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest from an extracted M6 archive.")
    parser.add_argument("--root", required=True, help="Extracted M6_database directory.")
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument("--include-duration", action="store_true")
    args = parser.parse_args()
    output = build_manifest(args.root, args.output, args.include_duration)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
