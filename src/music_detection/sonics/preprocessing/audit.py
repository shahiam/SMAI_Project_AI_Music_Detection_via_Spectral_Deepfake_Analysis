"""Audit the SONICS subset before building clips (safe to run while downloads continue).

For AI and human songs it reports what could become a shortcut or a bias:
  * genre mix (raw labels and coarse buckets) and how many songs survive equal-per-genre balancing
  * vocals vs instrumentals, durations, release era, artists spanning splits, duplicate songs,
    AI prompts reused across splits, AI bitrates, and other metadata columns
It reads only the CSVs and the file names on disk (no audio decoding). Tables go to --out.

    python -m music_detection.sonics.preprocessing.audit --root ~/Datasets/SONICS --out data/sonics_audit
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .utils import AUDIO_EXTS, coarse_genre, genre_column, normalise_split

KEY_COLUMNS = ("genre", "tag", "style", "no_vocal", "year", "artist", "title", "duration", "bit_rate",
               "artist_overlap", "skip_time", "algorithm", "label", "split", "mood", "topic", "source")


def stems_on_disk(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._")}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def with_percent(table: pd.DataFrame) -> pd.DataFrame:
    shares = (100 * table / table.sum()).round(1).add_suffix(" %")
    return pd.concat([table, shares], axis=1)


def add_genres(frame: pd.DataFrame) -> str | None:
    name = genre_column(frame)
    raw = frame[name] if name else pd.Series(None, index=frame.index, dtype=object)
    frame["genre_raw"] = raw.fillna("").astype(str).str.strip().str.lower()
    frame["genre"] = raw.map(coarse_genre)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit genre mix and possible shortcuts in the SONICS subset.")
    parser.add_argument("--root", type=Path, default=Path.home() / "Datasets" / "SONICS")
    parser.add_argument("--out", type=Path, default=Path("data/sonics_audit"))
    parser.add_argument("--top", type=int, default=20, help="How many raw genre labels to list")
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    fakes_all = pd.read_csv(args.root / "fake_songs.csv", low_memory=False)
    reals_all = pd.read_csv(args.root / "real_songs.csv", low_memory=False)
    name_col = "filepath" if "filepath" in fakes_all.columns else "filename"
    fakes_all["stem"] = fakes_all[name_col].astype(str).map(lambda v: Path(v).stem)
    reals_all = reals_all[reals_all["youtube_id"].notna()].copy()
    reals_all["youtube_id"] = reals_all["youtube_id"].astype(str).str.strip()
    for frame in (fakes_all, reals_all):
        frame["split"] = frame["split"].map(normalise_split) if "split" in frame.columns else "unknown"

    fake_genre_col = add_genres(fakes_all)
    real_genre_col = add_genres(reals_all)
    fakes = fakes_all[fakes_all["stem"].isin(stems_on_disk(args.root / "fake_songs"))].copy()
    reals = reals_all[reals_all["youtube_id"].isin(stems_on_disk(args.root / "real_songs"))].copy()
    fakes["class"], reals["class"] = "ai", "human"
    both = pd.concat([fakes, reals], ignore_index=True)

    section("Metadata columns (non-empty % in the full CSVs)")
    for label, frame in (("AI (fake_songs.csv)", fakes_all), ("human (real_songs.csv)", reals_all)):
        present = {c: f"{100 * frame[c].notna().mean():.0f}%" for c in KEY_COLUMNS if c in frame.columns}
        print(f"{label}: {len(frame)} rows; {present}")
    print(f"Genre column used: AI -> {fake_genre_col}, human -> {real_genre_col}")
    if real_genre_col is None or fake_genre_col is None:
        print("WARNING: one class has no usable genre labels, so equal-per-genre balancing is impossible "
              "from metadata alone (send me this output).")

    section("Songs on disk per split")
    print(pd.crosstab(both["split"], both["class"], margins=True))

    section(f"Raw genre labels (top {args.top}, songs on disk)")
    for label, frame in (("AI", fakes), ("human", reals)):
        print(f"{label}: {frame['genre_raw'].replace('', '(missing)').value_counts().head(args.top).to_dict()}")

    section("Coarse genres on disk (counts and share of each class)")
    coarse = pd.crosstab(both["genre"], both["class"])
    print(with_percent(coarse))
    feasible = 0
    per_split = pd.crosstab([both["split"], both["genre"]], both["class"])
    if {"ai", "human"} <= set(per_split.columns):
        feasible = int(per_split[["ai", "human"]].min(axis=1).sum())
    print(f"\nSongs per class after equal AI/human per (split x genre): {feasible} "
          f"(on disk now: {len(fakes)} AI, {len(reals)} human)")
    if len(reals_all):
        print("All human candidates in real_songs.csv (what further downloads could add):",
              (100 * reals_all["genre"].value_counts(normalize=True)).round(1).to_dict())
    unmapped = both.loc[both["genre"] == "other", "genre_raw"].value_counts()
    if len(unmapped):
        print(f"Labels mapped to 'other' (tell me if any deserve a bucket): {unmapped.head(15).to_dict()}")

    section("AI songs: coarse genre x generator, and SONICS label")
    if len(fakes):
        print(pd.crosstab(fakes["genre"], fakes.get("algorithm", pd.Series("?", index=fakes.index)), margins=True))
        if "label" in fakes.columns:
            print(pd.crosstab(fakes["genre"], fakes["label"]))

    section("Vocals (no_vocal = instrumental)")
    for label, frame in (("AI", fakes), ("human", reals)):
        if "no_vocal" in frame.columns:
            flag = frame["no_vocal"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            print(f"{label}: {int(flag.sum())} instrumental of {len(frame)} ({100 * flag.mean():.1f}%)")
        else:
            print(f"{label}: no 'no_vocal' column")

    section("Song duration in seconds (from the CSVs)")
    for label, frame in (("AI", fakes), ("human", reals)):
        if "duration" in frame.columns and len(frame):
            print(f"{label}: {pd.to_numeric(frame['duration'], errors='coerce').describe().round(1).to_dict()}")
    if "algorithm" in fakes.columns and "duration" in fakes.columns and len(fakes):
        print(fakes.groupby("algorithm")["duration"].median().round(0).rename("median s").to_dict())

    section("Release era of human songs (year)")
    if "year" in reals.columns and len(reals):
        years = pd.to_numeric(reals["year"], errors="coerce")
        bins = pd.cut(years, [0, 1969, 1979, 1989, 1999, 2009, 2019, 2100],
                      labels=["<1970", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])
        print(bins.value_counts(dropna=False).sort_index().to_dict())
    else:
        print("no 'year' column")

    section("Artists and duplicates (human songs on disk)")
    if "artist" in reals.columns and len(reals):
        artist = reals["artist"].astype(str).str.strip().str.lower()
        spans = reals.assign(a=artist).groupby("a")["split"].nunique()
        print(f"{artist.nunique()} artists; top: {artist.value_counts().head(5).to_dict()}")
        print(f"artists appearing in more than one split: {int((spans > 1).sum())}")
        if "title" in reals.columns:
            key = artist + " | " + reals["title"].astype(str).str.strip().str.lower()
            print(f"duplicate (artist, title) rows: {int(key.duplicated().sum())}")
    if "artist_overlap" in reals.columns:
        print(f"artist_overlap values: {reals['artist_overlap'].value_counts(dropna=False).head(5).to_dict()}")

    section("AI prompt reuse across splits")
    if len(fakes):
        prompt = fakes["stem"].str.extract(r"fake_(\d+)_", expand=False)
        spans = fakes.assign(p=prompt).dropna(subset=["p"]).groupby("p")["split"].nunique()
        print(f"{prompt.nunique()} prompt ids on disk; ids spanning more than one split: {int((spans > 1).sum())}")

    section("AI bitrate (bit_rate column) and other metadata")
    if "bit_rate" in fakes.columns and len(fakes):
        print(f"bit_rate: {pd.to_numeric(fakes['bit_rate'], errors='coerce').describe().round(0).to_dict()}")
    for col in ("lyrics_feature", "source", "mood", "topic"):
        for label, frame in (("AI", fakes), ("human", reals)):
            if col in frame.columns and frame[col].notna().any():
                print(f"{label} {col}: {frame[col].value_counts().head(6).to_dict()}")

    args.out.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(args.out / "genre_by_class.csv")
    per_split.to_csv(args.out / "genre_by_split_class.csv")
    fakes["genre_raw"].value_counts().to_csv(args.out / "raw_genres_ai.csv")
    reals["genre_raw"].value_counts().to_csv(args.out / "raw_genres_human.csv")
    unmapped.to_csv(args.out / "unmapped_genres.csv")
    print(f"\nTables written to {args.out}/")


if __name__ == "__main__":
    main()
