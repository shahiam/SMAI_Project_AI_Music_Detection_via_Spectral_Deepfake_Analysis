"""Small helpers shared by the SONICS preprocessing modules."""
from __future__ import annotations

AUDIO_EXTS = (".mp3", ".m4a", ".opus", ".webm", ".wav", ".ogg", ".flac")


def normalise_split(value: object) -> str:
    """Map SONICS split names onto the manifest contract (train / val / test)."""
    text = str(value).strip().lower()
    return {"valid": "val", "validation": "val", "dev": "val"}.get(text, text)


def allocate(total: int, counts: dict, caps: dict | None = None) -> dict:
    """Split `total` across strata proportionally to `counts` (largest remainder), never above caps.

    Used for stratified sampling: every stratum keeps its share of the full dataset.
    """
    caps = caps or counts
    keys = [k for k in counts if counts[k] > 0]
    weight = sum(counts[k] for k in keys)
    if weight == 0 or total <= 0:
        return {k: 0 for k in counts}
    raw = {k: total * counts[k] / weight for k in keys}
    out = {k: min(int(raw[k]), caps[k]) for k in keys}
    leftover = min(total, sum(caps[k] for k in keys)) - sum(out.values())
    order = sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    while leftover > 0:
        progressed = False
        for k in order:
            if leftover > 0 and out[k] < caps[k]:
                out[k] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return out
