"""Shortcut audits: can the label be predicted without listening to the music?"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .track_level import summarize


def sample_rate_baseline(manifest: pd.DataFrame) -> dict[str, Any]:
    """'Dumb' baseline: predict the label from the file's native sample rate only.

    Learns P(AI | native sample rate) on the train split and applies it to test. If this
    scores well, the dataset contains a bandwidth shortcut.
    """
    import soundfile as sf

    table = manifest.copy()
    table["native_sr"] = [sf.info(str(path)).samplerate for path in table["path"]]
    train = table[table["split"] == "train"]
    lookup = train.groupby("native_sr")["label"].mean()
    prior = float(train["label"].mean())
    test = table[table["split"] == "test"]
    probabilities = test["native_sr"].map(lookup).fillna(prior).to_numpy()
    crosstab = pd.crosstab(table["label"].map({0: "human", 1: "ai"}), table["native_sr"])
    return {
        "rule": "P(AI | native sample rate) learned on train split",
        "train_lookup": {str(int(k)): float(v) for k, v in lookup.items()},
        "test": summarize(test, probabilities, 0.5),
        "sample_rate_by_class": json.loads(crosstab.to_json()),
    }
