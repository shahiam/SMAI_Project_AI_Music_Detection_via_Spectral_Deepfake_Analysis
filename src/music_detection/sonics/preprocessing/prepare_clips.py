"""Turn SONICS songs into uniform MP3 clips plus a manifest for the existing pipeline.

Input
  * SONICS AI songs (extracted from fake_songs/part_*.zip) + fake_songs.csv
  * Human songs downloaded by download_youtube.py (40 s sections) + real_songs.csv

Output
  * <out-dir>/fake/*.mp3 and <out-dir>/real/*.mp3 and a manifest CSV with the columns the existing
    code expects (path, label, track_id, split) plus metadata (genre, generator, loudness, ...).

Shortcut controls (each one removes a way to tell AI from human without listening to the music):
  * Format parity: every clip of both classes is encoded identically (codec, bitrate, channels).
  * Bandwidth parity: a common sample rate (default 32 kHz) caps both classes at the same frequency.
  * Genre balance (--balance): equal numbers of AI and human songs, and then clips, in every
    split x coarse genre; the AI side keeps its generator x SONICS-label mix within each genre.
  * Position parity: AI clips come from the same part of the song as the human sections
    (40 s starting at 25% of the song), so intros/outros are not class-specific.
  * Loudness parity: each clip is gain-normalised to the same RMS level (peak-limited, no compression).
  * Silence filter: near-silent clips are dropped for both classes.
  * Instrumentals: AI songs flagged no_vocal are dropped (human songs are vocal songs).
  * Era (optional --min-year): drop old human recordings whose production style could give them away.

The manifest plugs straight into the existing CLI (same architecture as M6, separate models):

    python -m music_detection.cli classical --manifest data/sonics_manifest.csv --output runs/sonics_classical
    python -m music_detection.cli modern    --manifest data/sonics_manifest.csv --output runs/sonics_modern

Requirements: pandas, ffmpeg + ffprobe on PATH.

Example:
    python -m music_detection.sonics.preprocessing.prepare_clips \
        --sonics-root ~/Datasets/SONICS \
        --real-dir ~/Datasets/SONICS/real_songs \
        --out-dir ~/Datasets/SONICS/clips \
        --manifest data/sonics_manifest.csv --balance
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .utils import AUDIO_EXTS, allocate, coarse_genre, genre_column, normalise_split

FAKE_STRATA = ["generator", "sonics_label"]
VOLUME_PATTERNS = {
    "mean": re.compile(r"mean_volume:\s*(-?[\d.]+|-inf) dB"),
    "max": re.compile(r"max_volume:\s*(-?[\d.]+|-inf) dB"),
}


def column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    """Return frame[name], or a column filled with default if the CSV lacks it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_genres(frame: pd.DataFrame) -> None:
    name = genre_column(frame)
    raw = frame[name] if name else pd.Series(None, index=frame.index, dtype=object)
    frame["genre_raw"] = raw.fillna("").astype(str).str.strip().str.lower()
    frame["genre"] = raw.map(coarse_genre)


def probe_duration(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def segment_starts(duration: float, skip: float, k: int, seg: float, frac_start: float, frac_end: float) -> list[float]:
    """Evenly spaced clip start times inside the middle of a region (avoids its edges)."""
    if duration <= seg:
        return [0.0]
    lo = max(skip, duration * frac_start)
    hi = min(duration, max(duration * frac_end, lo + seg))
    if hi - lo < seg:
        lo, hi = max(0.0, duration - seg), duration
    last = hi - seg
    if k == 1 or last <= lo:
        return [round(lo, 2)]
    step = (last - lo) / (k - 1)
    starts = [lo + i * step for i in range(k)]
    # Drop clips that would overlap heavily when the usable region is short.
    kept = [starts[0]]
    for start in starts[1:]:
        if start - kept[-1] >= seg * 0.5:
            kept.append(start)
    return [round(s, 2) for s in kept]


def clip_region(duration: float, skip: float, is_section_file: bool, section_start: float,
                section_seconds: float) -> tuple[float, float]:
    """Part of the song clips are taken from: the same 'section' rule for AI songs and human downloads."""
    if is_section_file or section_seconds <= 0:
        return 0.0, duration
    start = max(skip, duration * section_start)
    start = min(start, max(0.0, duration - section_seconds))
    return round(start, 2), round(min(duration, start + section_seconds), 2)


def parse_volumedetect(stderr: str) -> tuple[float | None, float | None]:
    values = []
    for key in ("mean", "max"):
        match = VOLUME_PATTERNS[key].search(stderr)
        if match is None:
            values.append(None)
        else:
            values.append(float("-inf") if match.group(1) == "-inf" else float(match.group(1)))
    return values[0], values[1]


def index_audio(folder: Path) -> dict[str, Path]:
    """Map file stem -> path for every audio file below folder."""
    index: dict[str, Path] = {}
    if folder.is_dir():
        for path in folder.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("._"):
                index.setdefault(path.stem, path)
    return index


def load_fakes(root: Path, algorithms: list[str] | None, labels: list[str] | None,
               keep_instrumentals: bool) -> pd.DataFrame:
    frame = pd.read_csv(root / "fake_songs.csv", low_memory=False)
    name_col = "filepath" if "filepath" in frame.columns else "filename"
    frame["stem"] = frame[name_col].astype(str).map(lambda v: Path(v).stem)
    index = index_audio(root / "fake_songs") or index_audio(root)
    frame["source"] = frame["stem"].map(index)
    missing = int(frame["source"].isna().sum())
    frame = frame[frame["source"].notna()].copy()
    if missing:
        print(f"AI songs: {missing} rows in fake_songs.csv have no file on disk (not selected) - skipped")
    frame["split"] = column(frame, "split", "unknown").map(normalise_split)
    frame["generator"] = column(frame, "algorithm", "unknown_ai").fillna("unknown_ai").astype(str)
    frame["sonics_label"] = column(frame, "label", "fake").fillna("fake").astype(str)
    if algorithms:
        frame = frame[frame["generator"].str.lower().isin([a.lower() for a in algorithms])]
    if labels:
        frame = frame[frame["sonics_label"].str.lower().isin([l.lower() for l in labels])]
    if not keep_instrumentals and "no_vocal" in frame.columns:
        instrumental = truthy(frame["no_vocal"])
        print(f"AI songs: dropped {int(instrumental.sum())} instrumentals (no_vocal)")
        frame = frame[~instrumental]
    frame["skip"] = pd.to_numeric(column(frame, "skip_time", 0.0), errors="coerce").fillna(0.0)
    frame["year"] = float("nan")
    frame["key"] = "fake_" + frame["stem"]
    frame["label_int"] = 1
    add_genres(frame)
    return frame


def load_reals(root: Path, real_dir: Path, keep_instrumentals: bool, min_year: int | None) -> pd.DataFrame:
    frame = pd.read_csv(root / "real_songs.csv", low_memory=False)
    frame = frame[frame["youtube_id"].notna()].copy()
    frame["youtube_id"] = frame["youtube_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("youtube_id")
    frame["source"] = frame["youtube_id"].map(index_audio(real_dir))
    frame = frame[frame["source"].notna()].copy()
    frame["split"] = column(frame, "split", "unknown").map(normalise_split)
    frame["generator"] = "human"
    frame["sonics_label"] = "real"
    frame["year"] = pd.to_numeric(column(frame, "year", float("nan")), errors="coerce")
    if min_year is not None:
        keep = frame["year"] >= min_year
        print(f"Human songs: dropped {int((~keep).sum())} released before {min_year} or with unknown year")
        frame = frame[keep]
    if not keep_instrumentals and "no_vocal" in frame.columns:
        frame = frame[~truthy(frame["no_vocal"])]
    # Downloaded sections start at 0 inside the file, so the song's skip_time does not apply.
    frame["skip"] = 0.0
    frame["key"] = "real_" + frame["youtube_id"]
    frame["label_int"] = 0
    add_genres(frame)
    return frame


def sample_strata(pool: pd.DataFrame, n: int, strata: list[str], seed: int) -> pd.DataFrame:
    """n rows from pool keeping the proportions of the `strata` columns (all rows if n >= len)."""
    if n <= 0 or pool.empty:
        return pool.head(0)
    if n >= len(pool):
        return pool
    if not strata:
        return pool.sample(n=n, random_state=seed)
    parts = []
    for key, k in allocate(n, pool.groupby(strata).size().to_dict()).items():
        if k <= 0:
            continue
        key = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=pool.index)
        for col, value in zip(strata, key):
            mask &= pool[col] == value
        parts.append(pool[mask].sample(n=k, random_state=seed))
    return pd.concat(parts) if parts else pool.head(0)


def balance_groups(reals: pd.DataFrame, fakes: pd.DataFrame, keys: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal numbers of human and AI rows in every group (e.g. split x genre)."""
    def groups(frame: pd.DataFrame) -> dict:
        return {k if isinstance(k, tuple) else (k,): g for k, g in frame.groupby(keys)}

    real_groups, fake_groups = groups(reals), groups(fakes)
    kept_real, kept_fake = [reals.head(0)], [fakes.head(0)]
    for key in sorted(set(real_groups) | set(fake_groups), key=str):
        real = real_groups.get(key, reals.head(0))
        fake = fake_groups.get(key, fakes.head(0))
        n = min(len(real), len(fake))
        kept_real.append(sample_strata(real, n, [], seed))
        kept_fake.append(sample_strata(fake, n, FAKE_STRATA, seed))
    return pd.concat(kept_real), pd.concat(kept_fake)


def limit_frame(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    fractions = frame["split"].value_counts(normalize=True)
    return pd.concat(
        sample_strata(frame[frame["split"] == s], max(1, round(limit * f)), [], seed) for s, f in fractions.items()
    )


def _input_args(source: Path, start: float, args: argparse.Namespace) -> list[str]:
    return ["-ss", f"{start:.2f}", "-t", f"{args.segment_seconds:.2f}", "-i", str(source)]


def measure_clip(source: Path, start: float, args: argparse.Namespace) -> tuple[float | None, float | None]:
    """RMS (mean_volume) and peak (max_volume) of the clip as it will be encoded (rate, channels)."""
    chain = [f"lowpass=f={args.lowpass}"] if args.lowpass else []
    chain.append(f"aresample={args.sample_rate}")
    if args.channels == 1:
        chain.append("aformat=channel_layouts=mono")
    chain.append("volumedetect")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info",
               *_input_args(source, start, args), "-af", ",".join(chain), "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True)
    return parse_volumedetect(result.stderr)


def encode_clip(source: Path, target: Path, start: float, args: argparse.Namespace) -> dict:
    if target.is_file() and target.stat().st_size > 0:
        return {"status": "ok", "mean_db": None, "gain_db": None}
    mean_db, max_db = measure_clip(source, start, args)
    if mean_db is None or max_db is None:
        return {"status": "failed", "error": "could not measure loudness (unreadable audio?)"}
    if mean_db < args.min_rms_db:
        return {"status": "silent", "mean_db": mean_db}
    gain = 0.0
    if args.loudness_target is not None:
        gain = args.loudness_target - mean_db
        if max_db + gain > args.peak_ceiling:  # never push peaks above the ceiling (no limiter, no clipping)
            gain = args.peak_ceiling - max_db
    chain = [f"lowpass=f={args.lowpass}"] if args.lowpass else []
    if abs(gain) >= 0.01:
        chain.append(f"volume={gain:.2f}dB")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *_input_args(source, start, args)]
    if chain:
        command += ["-af", ",".join(chain)]
    command += ["-ac", str(args.channels), "-ar", str(args.sample_rate)]
    command += ["-c:a", "libmp3lame", "-b:a", f"{args.bitrate}k"] if args.codec == "mp3" else ["-c:a", "pcm_s16le"]
    tmp = target.with_name(target.stem + ".part" + target.suffix)
    result = subprocess.run(command + [str(tmp)], capture_output=True, text=True)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return {"status": "failed", "error": result.stderr.strip()[:200] or "ffmpeg failed"}
    tmp.replace(target)
    return {"status": "ok", "mean_db": round(mean_db, 2), "gain_db": round(gain, 2)}


def plan_song(row: pd.Series, args: argparse.Namespace) -> list[dict]:
    source = Path(row["source"])
    duration = probe_duration(source)
    if duration is None or duration < 1.0:
        return []
    is_section = bool(row["label_int"] == 0 and duration <= 1.5 * args.section_seconds)
    lo, hi = clip_region(duration, float(row["skip"]), is_section, args.section_start, args.section_seconds)
    skip = 0.0 if args.section_seconds > 0 else float(row["skip"])
    starts = [round(lo + s, 2) for s in segment_starts(hi - lo, skip, args.segments_per_song, args.segment_seconds,
                                                      args.start_fraction, args.end_fraction)]
    ext = ".mp3" if args.codec == "mp3" else ".wav"
    folder = args.out_dir / ("fake" if row["label_int"] else "real")
    return [
        {
            "source": source,
            "target": folder / f"{row['key']}_s{i}{ext}",
            "start": start,
            "label": int(row["label_int"]),
            "track_id": row["key"],
            "split": row["split"],
            "genre": row["genre"],
            "genre_raw": row["genre_raw"],
            "generator": row["generator"],
            "sonics_label": row["sonics_label"],
            "year": row["year"],
        }
        for i, start in enumerate(starts)
    ]


def genre_table(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    table = pd.crosstab(frame["genre"], frame[label_col].map({0: "human", 1: "ai"}))
    return table.assign(total=table.sum(axis=1)).sort_values("total", ascending=False).drop(columns="total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build uniform SONICS clips and a pipeline manifest.")
    parser.add_argument("--sonics-root", required=True, type=Path, help="Folder with fake_songs.csv, real_songs.csv, fake_songs/")
    parser.add_argument("--real-dir", required=True, type=Path, help="Folder written by download_youtube.py")
    parser.add_argument("--out-dir", required=True, type=Path, help="Where the clips go (outside the repo)")
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest CSV to write, e.g. data/sonics_manifest.csv")
    parser.add_argument("--balance", action="store_true",
                        help="Equal AI and human songs AND clips per split x genre (AI keeps generator x label mix)")
    parser.add_argument("--no-genre-balance", action="store_true", help="With --balance, balance per split only")
    parser.add_argument("--limit-real", type=int, help="Max human songs (stratified by split)")
    parser.add_argument("--limit-fake", type=int, help="Max AI songs (stratified by split); ignored with --balance")
    parser.add_argument("--algorithms", nargs="+", help="Keep only these generators, e.g. suno_v3.5 udio_130")
    parser.add_argument("--labels", nargs="+", help="Keep only these SONICS labels, e.g. 'full fake' 'mostly fake'")
    parser.add_argument("--keep-instrumentals", action="store_true", help="Keep AI songs flagged no_vocal")
    parser.add_argument("--min-year", type=int, help="Drop human songs released before this year (era shortcut)")
    parser.add_argument("--segments-per-song", type=int, default=3)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--section-start", type=float, default=0.25,
                        help="AI clips come from this point of the song, like the human downloads (0.25 = 25%%)")
    parser.add_argument("--section-seconds", type=float, default=40.0,
                        help="Length of that region; must match the downloader. 0 = use the whole song")
    parser.add_argument("--start-fraction", type=float, default=0.10, help="Ignore the first part of the region")
    parser.add_argument("--end-fraction", type=float, default=0.90, help="Ignore the last part of the region")
    parser.add_argument("--codec", choices=["mp3", "wav"], default="mp3")
    parser.add_argument("--bitrate", type=int, default=128, help="MP3 kbps, identical for both classes")
    parser.add_argument("--sample-rate", type=int, default=32000,
                        help="Common rate for BOTH classes. Resampling is a steep low-pass at rate/2, so 32000 "
                             "caps every clip at 16 kHz. Set it to 2x the lowest bandwidth profile_audio reports")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 2], help="1 = mono (the pipeline uses mono anyway)")
    parser.add_argument("--lowpass", type=int, help="Optional extra (gentle) low-pass in Hz for BOTH classes")
    parser.add_argument("--loudness-target", type=float, default=-20.0, help="RMS level (dBFS) every clip is set to")
    parser.add_argument("--no-loudness-norm", action="store_true", help="Keep original levels")
    parser.add_argument("--peak-ceiling", type=float, default=-1.0, help="Gain never pushes peaks above this (dBFS)")
    parser.add_argument("--min-rms-db", type=float, default=-45.0, help="Drop clips quieter than this (dBFS RMS)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show counts and a few planned clips, encode nothing")
    args = parser.parse_args()
    if args.no_loudness_norm:
        args.loudness_target = None
    pd.set_option("display.width", 160)

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH (brew install ffmpeg / winget install ffmpeg)")

    reals = limit_frame(load_reals(args.sonics_root, args.real_dir, args.keep_instrumentals, args.min_year),
                        args.limit_real, args.seed)
    fakes = load_fakes(args.sonics_root, args.algorithms, args.labels, args.keep_instrumentals)
    reals = reals[reals["split"].isin(["train", "val", "test"])]
    fakes = fakes[fakes["split"].isin(["train", "val", "test"])]

    balance_keys = ["split"]
    if args.balance:
        known = lambda f: (f["genre"] != "unknown").mean() if len(f) else 0.0  # noqa: E731
        if not args.no_genre_balance and known(reals) >= 0.5 and known(fakes) >= 0.5:
            balance_keys = ["split", "genre"]
        elif not args.no_genre_balance:
            print("WARNING: genre labels missing for one class; balancing per split only (run the audit).")
        before = (len(reals), len(fakes))
        reals, fakes = balance_groups(reals, fakes, balance_keys, args.seed)
        print(f"Balanced per {' x '.join(balance_keys)}: human {before[0]} -> {len(reals)}, AI {before[1]} -> {len(fakes)}")
    else:
        fakes = limit_frame(fakes, args.limit_fake, args.seed)
    songs = pd.concat([reals, fakes], ignore_index=True)
    if songs.empty:
        raise SystemExit("No songs found. Check --sonics-root / --real-dir.")
    print("Songs per split (0 = human, 1 = AI):")
    print(pd.crosstab(songs["split"], songs["label_int"]))
    print("Songs per coarse genre:")
    print(genre_table(songs, "label_int"))
    print("AI generators:", fakes["generator"].value_counts().to_dict())

    if args.dry_run:
        for _, row in songs.sample(n=min(6, len(songs)), random_state=args.seed).iterrows():
            for clip in plan_song(row, args):
                print(f"  {clip['split']:5s} label={clip['label']} {clip['genre']:12s} start={clip['start']:7.2f}s  "
                      f"{clip['target'].name}")
        return

    (args.out_dir / "fake").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "real").mkdir(parents=True, exist_ok=True)

    print(f"Planning clips for {len(songs)} songs (reading durations)...", flush=True)
    clips: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for planned in pool.map(lambda r: plan_song(r[1], args), songs.iterrows()):
            clips.extend(planned)

    print(f"Encoding {len(clips)} clips with {args.workers} workers...", flush=True)
    counts = {"failed": 0, "silent": 0}
    done_clips = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(encode_clip, c["source"], c["target"], c["start"], args): c for c in clips}
        for count, future in enumerate(as_completed(futures), start=1):
            clip, result = futures[future], future.result()
            if result["status"] == "ok":
                done_clips.append({**clip, "mean_db": result.get("mean_db"), "gain_db": result.get("gain_db")})
            else:
                counts[result["status"]] += 1
                if result["status"] == "failed":
                    print(f"  failed {clip['target'].name}: {result.get('error')}")
            if count % 500 == 0:
                print(f"  {count}/{len(clips)} clips", flush=True)

    manifest = pd.DataFrame([
        {
            "path": clip["target"].resolve().as_posix(),
            "label": clip["label"],
            "track_id": clip["track_id"],
            "split": clip["split"],
            "genre": clip["genre"],
            "genre_raw": clip["genre_raw"],
            "generator": clip["generator"],
            "sonics_label": clip["sonics_label"],
            "year": clip["year"],
            "segment_start": clip["start"],
            "segment_seconds": args.segment_seconds,
            "mean_volume_db": clip["mean_db"],
            "gain_db": clip["gain_db"],
            "source_file": Path(clip["source"]).name,
        }
        for clip in done_clips
    ])
    if args.balance and not manifest.empty:
        # Short songs and silent clips leave uneven clip counts: trim the larger class per group.
        real_clips, fake_clips = balance_groups(manifest[manifest["label"] == 0], manifest[manifest["label"] == 1],
                                                balance_keys, args.seed)
        dropped = len(manifest) - len(real_clips) - len(fake_clips)
        manifest = pd.concat([real_clips, fake_clips])
        if dropped:
            print(f"Balanced clip counts per {' x '.join(balance_keys)}: dropped {dropped} surplus clips (files stay on disk)")
    manifest = manifest.sort_values(["split", "label", "track_id", "segment_start"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} clips to {args.manifest} "
          f"({counts['failed']} failed, {counts['silent']} dropped as silent)")
    print(pd.crosstab(manifest["split"], manifest["label"]))
    print(genre_table(manifest, "label"))


if __name__ == "__main__":
    main()
