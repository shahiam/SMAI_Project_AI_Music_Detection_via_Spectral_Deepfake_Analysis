"""Turn SONICS songs into uniform MP3 clips plus a manifest for the existing pipeline.

Input
  * SONICS fake songs (extracted from fake_songs/part_*.zip) + fake_songs.csv
  * Real songs downloaded by download_youtube.py + real_songs.csv

Output
  * <out-dir>/fake/*.mp3 and <out-dir>/real/*.mp3: K clips per song, every clip encoded
    with the SAME settings (codec, bitrate, sample rate, channels) for both classes, so the
    model cannot tell real from fake by file format alone ("codec parity").
  * The common sample rate (default 32 kHz) also gives both classes the same frequency
    ceiling (16 kHz), which prevents the bandwidth shortcut found on M6. Run
    profile_audio.py first and lower --sample-rate if either class stops below 16 kHz.
  * A manifest CSV with the columns the existing code expects (path, label, track_id, split)
    plus extra metadata (generator, sonics_label, segment_start, ...).

The manifest plugs straight into the existing CLI, so the SONICS model is the same
architecture as the M6 model but trained and saved separately:

    python -m music_detection.cli classical --manifest data/sonics_manifest.csv --output runs/sonics_classical
    python -m music_detection.cli modern    --manifest data/sonics_manifest.csv --output runs/sonics_modern --epochs 10

Requirements: pandas, ffmpeg + ffprobe on PATH.

Example:
    python -m music_detection.sonics.preprocessing.prepare_clips \
        --sonics-root /Volumes/BACKUP/SONICS \
        --real-dir /Volumes/BACKUP/SONICS/real_songs \
        --out-dir /Volumes/BACKUP/SONICS/clips \
        --manifest data/sonics_manifest.csv --balance
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .utils import AUDIO_EXTS, allocate, normalise_split



def stratified_by(frame: pd.DataFrame, per_split: dict[str, int], strata: list[str], seed: int) -> pd.DataFrame:
    """Per split, take n rows keeping the proportions of the `strata` columns."""
    parts = []
    for split, n in per_split.items():
        pool = frame[frame["split"] == split]
        if pool.empty or n <= 0:
            continue
        if not strata:
            parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
            continue
        for key, k in allocate(n, pool.groupby(strata).size().to_dict()).items():
            key = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=pool.index)
            for col, val in zip(strata, key):
                mask &= pool[col] == val
            parts.append(pool[mask].sample(n=k, random_state=seed))
    return pd.concat(parts) if parts else frame.head(0)


def column(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    """Return frame[name], or a column filled with default if the CSV lacks it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


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
    """Evenly spaced clip start times inside the middle of the song (avoids intros/outros)."""
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


def index_audio(folder: Path) -> dict[str, Path]:
    """Map file stem -> path for every audio file below folder."""
    index: dict[str, Path] = {}
    if folder.is_dir():
        for path in folder.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("._"):
                index.setdefault(path.stem, path)
    return index


def load_fakes(root: Path, algorithms: list[str] | None, labels: list[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(root / "fake_songs.csv")
    name_col = "filepath" if "filepath" in frame.columns else "filename"
    frame["stem"] = frame[name_col].astype(str).map(lambda v: Path(v).stem)
    index = index_audio(root / "fake_songs")
    if not index:
        index = index_audio(root)
    frame["source"] = frame["stem"].map(index)
    missing = frame["source"].isna().sum()
    frame = frame[frame["source"].notna()].copy()
    if missing:
        print(f"Fake songs: {missing} rows in fake_songs.csv have no extracted file (zip part not downloaded?) - skipped")
    frame["split"] = column(frame, "split", "unknown").map(normalise_split)
    frame["generator"] = column(frame, "algorithm", "unknown_ai").fillna("unknown_ai").astype(str)
    frame["sonics_label"] = column(frame, "label", "fake").fillna("fake").astype(str)
    if algorithms:
        frame = frame[frame["generator"].str.lower().isin([a.lower() for a in algorithms])]
    if labels:
        frame = frame[frame["sonics_label"].str.lower().isin([l.lower() for l in labels])]
    frame["skip"] = pd.to_numeric(column(frame, "skip_time", 0.0), errors="coerce").fillna(0.0)
    frame["csv_duration"] = pd.to_numeric(column(frame, "duration", float("nan")), errors="coerce")
    frame["key"] = "fake_" + frame["stem"]
    frame["label_int"] = 1
    return frame


def load_reals(root: Path, real_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "real_songs.csv")
    frame = frame[frame["youtube_id"].notna()].copy()
    frame["youtube_id"] = frame["youtube_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("youtube_id")
    index = index_audio(real_dir)
    frame["source"] = frame["youtube_id"].map(index)
    frame = frame[frame["source"].notna()].copy()
    frame["split"] = column(frame, "split", "unknown").map(normalise_split)
    frame["generator"] = "human"
    frame["sonics_label"] = "real"
    frame["csv_duration"] = pd.to_numeric(column(frame, "duration", float("nan")), errors="coerce")
    # Downloaded sections start at 0 inside the file, so the song's skip_time does not apply.
    frame["skip"] = 0.0
    frame["key"] = "real_" + frame["youtube_id"]
    frame["label_int"] = 0
    return frame


def stratified_sample(frame: pd.DataFrame, per_split: dict[str, int], seed: int) -> pd.DataFrame:
    parts = []
    for split, n in per_split.items():
        group = frame[frame["split"] == split]
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts) if parts else frame.head(0)


def limit_frame(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    fractions = frame["split"].value_counts(normalize=True)
    return stratified_sample(frame, {s: max(1, round(limit * f)) for s, f in fractions.items()}, seed)


def encode_clip(source: Path, target: Path, start: float, args: argparse.Namespace) -> str | None:
    if target.is_file() and target.stat().st_size > 0:
        return None
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.2f}", "-t", f"{args.segment_seconds:.2f}", "-i", str(source),
    ]
    if args.lowpass:
        command += ["-af", f"lowpass=f={args.lowpass}"]
    command += ["-ac", str(args.channels), "-ar", str(args.sample_rate)]
    if args.codec == "mp3":
        command += ["-c:a", "libmp3lame", "-b:a", f"{args.bitrate}k"]
    else:
        command += ["-c:a", "pcm_s16le"]
    tmp = target.with_name(target.stem + ".part" + target.suffix)
    result = subprocess.run(command + [str(tmp)], capture_output=True, text=True)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return result.stderr.strip()[:200] or "ffmpeg failed"
    tmp.replace(target)
    return None


def plan_song(row: pd.Series, args: argparse.Namespace) -> list[dict]:
    source = Path(row["source"])
    duration = probe_duration(source)
    if duration is None or duration < 1.0:
        return []
    starts = segment_starts(duration, float(row["skip"]), args.segments_per_song, args.segment_seconds,
                            args.start_fraction, args.end_fraction)
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
            "generator": row["generator"],
            "sonics_label": row["sonics_label"],
        }
        for i, start in enumerate(starts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build uniform SONICS clips and a pipeline manifest.")
    parser.add_argument("--sonics-root", required=True, type=Path, help="Folder with fake_songs.csv, real_songs.csv, fake_songs/")
    parser.add_argument("--real-dir", required=True, type=Path, help="Folder written by download_youtube.py")
    parser.add_argument("--out-dir", required=True, type=Path, help="Where the clips go (outside the repo)")
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest CSV to write, e.g. data/sonics_manifest.csv")
    parser.add_argument("--balance", action="store_true",
                        help="Equal songs AND equal clips per split for both classes; AI side keeps generator x label proportions")
    parser.add_argument("--limit-real", type=int, help="Max real songs (stratified by split)")
    parser.add_argument("--limit-fake", type=int, help="Max fake songs (stratified by split); ignored with --balance")
    parser.add_argument("--algorithms", nargs="+", help="Keep only these generators, e.g. suno_v3.5 udio_130")
    parser.add_argument("--labels", nargs="+", help="Keep only these SONICS labels, e.g. 'full fake' 'mostly fake'")
    parser.add_argument("--segments-per-song", type=int, default=3)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--start-fraction", type=float, default=0.10, help="Ignore the first part of each song")
    parser.add_argument("--end-fraction", type=float, default=0.90, help="Ignore the last part of each song")
    parser.add_argument("--codec", choices=["mp3", "wav"], default="mp3")
    parser.add_argument("--bitrate", type=int, default=128, help="MP3 kbps, identical for both classes")
    parser.add_argument("--sample-rate", type=int, default=32000,
                        help="Common rate for BOTH classes. Resampling is a steep low-pass at rate/2, so 32000 "
                             "caps every clip at 16 kHz. Set it to 2x the lowest bandwidth profile_audio.py reports")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 2], help="1 = mono (the pipeline uses mono anyway)")
    parser.add_argument("--lowpass", type=int, help="Optional extra (gentle) low-pass in Hz for BOTH classes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show counts and a few planned clips, encode nothing")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH (brew install ffmpeg / winget install ffmpeg)")

    reals = limit_frame(load_reals(args.sonics_root, args.real_dir), args.limit_real, args.seed)
    fakes = load_fakes(args.sonics_root, args.algorithms, args.labels)
    if args.balance:
        # Same number of songs per split as the real side, keeping generator x label proportions.
        fakes = stratified_by(fakes, reals["split"].value_counts().to_dict(), ["generator", "sonics_label"], args.seed)
    else:
        fakes = limit_frame(fakes, args.limit_fake, args.seed)
    songs = pd.concat([reals, fakes], ignore_index=True)
    songs = songs[songs["split"].isin(["train", "val", "test"])]
    if songs.empty:
        raise SystemExit("No songs found. Check --sonics-root / --real-dir and that the zips are extracted.")
    print("Songs per split and class (0 = real, 1 = fake):")
    print(pd.crosstab(songs["split"], songs["label_int"]))
    print("Fake generators:", fakes["generator"].value_counts().to_dict())

    if args.dry_run:
        for _, row in songs.sample(n=min(6, len(songs)), random_state=args.seed).iterrows():
            for clip in plan_song(row, args):
                print(f"  {clip['split']:5s} label={clip['label']} start={clip['start']:7.2f}s  {clip['target'].name}")
        return

    (args.out_dir / "fake").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "real").mkdir(parents=True, exist_ok=True)

    print(f"Planning clips for {len(songs)} songs (reading durations)...", flush=True)
    clips: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for planned in pool.map(lambda r: plan_song(r[1], args), songs.iterrows()):
            clips.extend(planned)

    print(f"Encoding {len(clips)} clips with {args.workers} workers...", flush=True)
    failures = 0
    done_clips = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(encode_clip, c["source"], c["target"], c["start"], args): c for c in clips}
        for count, future in enumerate(as_completed(futures), start=1):
            clip = futures[future]
            error = future.result()
            if error:
                failures += 1
                print(f"  failed {clip['target'].name}: {error}")
            else:
                done_clips.append(clip)
            if count % 500 == 0:
                print(f"  {count}/{len(clips)} clips", flush=True)

    rows = [
        {
            "path": clip["target"].resolve().as_posix(),
            "label": clip["label"],
            "track_id": clip["track_id"],
            "split": clip["split"],
            "generator": clip["generator"],
            "sonics_label": clip["sonics_label"],
            "segment_start": clip["start"],
            "segment_seconds": args.segment_seconds,
            "source_file": Path(clip["source"]).name,
        }
        for clip in done_clips
    ]
    manifest = pd.DataFrame(rows)
    if args.balance and not manifest.empty:
        # Short songs give fewer clips, so trim the larger class to the same clip count per split
        # (AI side trimmed proportionally across generator x label).
        balanced = []
        for split, group in manifest.groupby("split"):
            n = group["label"].value_counts().min() if group["label"].nunique() == 2 else 0
            balanced.append(stratified_by(group[group["label"] == 0], {split: n}, [], args.seed))
            balanced.append(stratified_by(group[group["label"] == 1], {split: n}, ["generator", "sonics_label"], args.seed))
        dropped = len(manifest) - sum(len(b) for b in balanced)
        manifest = pd.concat(balanced)
        if dropped:
            print(f"Balanced clip counts per split: dropped {dropped} surplus clips (files stay on disk)")
    manifest = manifest.sort_values(["split", "label", "track_id", "segment_start"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} clips to {args.manifest} ({failures} failures)")
    print(pd.crosstab(manifest["split"], manifest["label"]))


if __name__ == "__main__":
    main()
