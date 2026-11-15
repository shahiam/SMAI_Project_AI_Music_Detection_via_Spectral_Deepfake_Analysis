from pathlib import Path

import pandas as pd

from music_detection.m6_manifest import build_manifest


def test_m6_manifest_infers_labels_generators_and_splits(tmp_path: Path):
    ai = tmp_path / "M6_database" / "ai" / "d_genres"
    human = tmp_path / "M6_database" / "human" / "genres"
    ai.mkdir(parents=True)
    human.mkdir(parents=True)
    (ai / "AMG_jazz12.wav").write_bytes(b"RIFF")
    (human / "human_jazz12.wav").write_bytes(b"RIFF")

    output = build_manifest(tmp_path / "M6_database", tmp_path / "manifest.csv")
    frame = pd.read_csv(output)

    assert set(frame["label"]) == {0, 1}
    assert frame.loc[frame["label"].eq(1), "generator"].item() == "AMG"
    assert frame.loc[frame["label"].eq(1), "genre"].item() == "jazz"
    assert set(frame["split"]).issubset({"train", "val", "test"})
