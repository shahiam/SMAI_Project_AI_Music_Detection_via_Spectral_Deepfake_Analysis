"""Small helpers shared by the SONICS preprocessing modules."""
from __future__ import annotations

import re

AUDIO_EXTS = (".mp3", ".m4a", ".opus", ".webm", ".wav", ".ogg", ".flac")

# Columns that may hold a genre label, in order of preference.
GENRE_COLUMNS = ("genre", "tag", "genres", "style")

# Coarse genre buckets used to balance AI and human songs per genre. SONICS AI genres come from
# free-text prompts ("lo-fi hip hop", "synthwave", ...) while human genres come from lyrics-site
# tags ("rap", "rb", "pop", ...), so both are mapped onto the same buckets. First match wins.
COARSE_GENRES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("hiphop", r"hip[\s-]?hop|\brap\b|\btrap\b|\bdrill\b|\bgrime\b|boom[\s-]?bap"),
        ("rnb_soul", r"r\s?&\s?b|\brnb\b|\br'?n'?b\b|\brb\b|\bsoul|\bfunk|motown|gospel"),
        ("metal", r"metal|djent|thrash|grindcore"),
        ("rock", r"rock|punk|grunge|\bemo\b|shoegaze|alternative"),
        ("country_folk", r"country|folk|bluegrass|americana|acoustic|singer[\s-]?songwriter"),
        ("jazz_blues", r"jazz|blues|swing|bebop|bossa"),
        ("classical", r"classical|orchestra|symphon|\bopera\b|baroque|cinematic|soundtrack|\bscore\b|piano"),
        ("latin_reggae", r"latin|reggae|salsa|samba|bachata|\bska\b|afro|dancehall|cumbia|flamenco"),
        ("electronic", r"electr|\bedm\b|house|techno|trance|dubstep|drum\s?(and|&|n)\s?bass|\bdnb\b|synth"
                       r"|dance|disco|ambient|lo-?fi|chillwave|vaporwave|\bidm\b|breakbeat|hyperpop|garage"),
        ("pop", r"pop|ballad|\bindie\b"),
    )
)


def normalise_split(value: object) -> str:
    """Map SONICS split names onto the manifest contract (train / val / test)."""
    text = str(value).strip().lower()
    return {"valid": "val", "validation": "val", "dev": "val"}.get(text, text)


def coarse_genre(value: object) -> str:
    """Map a free-text genre/tag onto a coarse bucket ('unknown' if missing, 'other' if unmatched)."""
    if value is None or (isinstance(value, float) and value != value):
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in ("nan", "none"):
        return "unknown"
    for name, pattern in COARSE_GENRES:
        if pattern.search(text):
            return name
    return "other"


def genre_column(frame, min_fraction: float = 0.3) -> str | None:
    """First GENRE_COLUMNS entry present in `frame` with at least `min_fraction` non-empty values."""
    for name in GENRE_COLUMNS:
        if name in frame.columns and len(frame) and frame[name].notna().mean() >= min_fraction:
            return name
    return None


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
