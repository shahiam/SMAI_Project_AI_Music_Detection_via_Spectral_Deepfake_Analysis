"""Download the human (real) half of SONICS from YouTube.

SONICS only distributes the AI songs. For the real songs it ships `real_songs.csv`,
which lists one YouTube video per song (`youtube_id`). This script reads that CSV,
optionally samples a subset, and downloads the audio with yt-dlp.

By default it downloads only a short section of each song (40 s starting at 25% of
the song) to save disk space: ~0.6-0.8 MB per song instead of ~3-5 MB. Use --full for
whole songs.

By default the audio is kept in YouTube's own format (usually Opus, sometimes AAC)
instead of being converted to MP3 here: prepare_clips.py later encodes real and AI
clips with identical settings, so converting now would only add an extra lossy step.

The script is resumable: re-running it skips songs that were already downloaded,
and every attempt is logged to <out-dir>/download_log.csv.

Requirements (install into your environment):
    pip install -U yt-dlp pandas
    ffmpeg on PATH  (macOS: brew install ffmpeg | Windows: winget install ffmpeg)

Examples:
    # 3,000 successful downloads, split like SONICS, 40 s sections, 8 parallel jobs
    python -m music_detection.sonics.preprocessing.download_youtube \
        --csv ~/Datasets/SONICS/real_songs.csv \
        --out-dir ~/Datasets/SONICS/real_songs --limit 3000 --jobs 8

    # Preview what would be downloaded without touching the network
    python -m music_detection.sonics.preprocessing.download_youtube --csv real_songs.csv \
        --out-dir real_songs --limit 20 --dry-run

Note: downloaded audio is copyrighted. Use it only for this research project,
keep it outside the git repository, and do not redistribute it.
"""
from __future__ import annotations

import argparse
import csv
import random
import json
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .utils import AUDIO_EXTS, normalise_split

LOG_NAME = "download_log.csv"
LOG_FIELDS = ["youtube_id", "id", "split", "status", "path", "section_start", "section_seconds", "message", "timestamp"]


def load_real_songs(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "youtube_id" not in frame.columns:
        raise SystemExit(f"{csv_path} has no 'youtube_id' column; columns are {list(frame.columns)}")
    frame = frame[frame["youtube_id"].notna()].copy()
    frame["youtube_id"] = frame["youtube_id"].astype(str).str.strip()
    frame = frame[frame["youtube_id"] != ""].drop_duplicates("youtube_id")
    if "split" in frame.columns:
        frame["split"] = frame["split"].map(normalise_split)
    else:
        frame["split"] = "unknown"
    if "id" not in frame.columns:
        frame["id"] = frame["youtube_id"]
    return frame


def existing_file(out_dir: Path, youtube_id: str) -> Path | None:
    for ext in AUDIO_EXTS:
        candidate = out_dir / f"{youtube_id}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def read_log(log_path: Path) -> dict[str, dict[str, str]]:
    if not log_path.is_file():
        return {}
    with log_path.open(newline="", encoding="utf-8") as handle:
        return {row["youtube_id"]: row for row in csv.DictReader(handle)}


def append_log(log_path: Path, row: dict[str, object]) -> None:
    new_file = not log_path.is_file()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LOG_FIELDS})


def section_for(duration: object, section_seconds: float, start_fraction: float) -> tuple[float, float]:
    """Return (start, end) seconds for the section to download."""
    try:
        total = float(duration)
    except (TypeError, ValueError):
        total = float("nan")
    if not total == total or total <= 0:  # NaN or missing: assume a typical 3-minute song
        total = 180.0
    if total <= section_seconds:
        return 0.0, total
    start = min(total * start_fraction, total - section_seconds)
    return round(start, 2), round(start + section_seconds, 2)


def build_ydl_options(out_dir: Path, args: argparse.Namespace, section: tuple[float, float] | None) -> dict:
    import yt_dlp.utils

    options: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": args.codec,
                "preferredquality": str(args.bitrate),
            }
        ],
    }
    if args.codec == "mp3":
        # Force a common sample rate so every real file shares one format.
        options["postprocessor_args"] = {"extractaudio": ["-ar", str(args.sample_rate)]}
    if section is not None:
        options["download_ranges"] = yt_dlp.utils.download_range_func(None, [section])
        options["force_keyframes_at_cuts"] = True
    if args.cookies_from_browser:
        options["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.rate_limit:
        options["ratelimit"] = args.rate_limit
    return options


def download_one(youtube_id: str, out_dir: Path, args: argparse.Namespace, section) -> tuple[str, str, Path | None]:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    try:
        with yt_dlp.YoutubeDL(build_ydl_options(out_dir, args, section)) as ydl:
            ydl.download([url])
    except Exception as error:  # yt-dlp raises DownloadError and friends
        message = str(error).replace("\n", " ")[:300]
        return "failed", message, None
    path = existing_file(out_dir, youtube_id)
    if path is None:
        return "failed", "yt-dlp finished but no audio file was found", None
    return "ok", "", path


BOT_MARKERS = ("confirm you", "not a bot", "429", "too many requests", "rate-limit", "rate limit")


class Progress:
    """Thread-safe bookkeeping for parallel downloads with per-split targets."""

    def __init__(self, quotas: dict[str, int], already: dict[str, int], log_path: Path, pause_minutes: float):
        self.quotas = quotas
        self.ok = dict(already)
        self.inflight = {split: 0 for split in quotas}
        self.failed = 0
        self.attempted = 0
        self.lock = threading.Lock()
        self.log_path = log_path
        self.pause_until = 0.0
        self.pause_seconds = pause_minutes * 60
        self.bot_hits = 0
        self.pauses = 0
        self.max_pauses = 3
        self.stop = False

    def set_candidates(self, rows: list) -> None:
        self.queues = {split: deque() for split in self.quotas}
        for row in rows:
            if row["split"] in self.queues:
                self.queues[row["split"]].append(row)

    def next_task(self):
        """Return the next row to download, "wait" if only in-flight jobs can still change things, or None."""
        with self.lock:
            if self.stop:
                return None
            for split, quota in self.quotas.items():
                if self.ok.get(split, 0) + self.inflight[split] < quota and self.queues[split]:
                    self.inflight[split] += 1
                    return self.queues[split].popleft()
            needy = any(self.ok.get(s, 0) < q and self.queues[s] for s, q in self.quotas.items())
            if needy and any(self.inflight.values()):
                return "wait"  # a running job may still fail and free a slot
            return None

    def finish(self, split: str, status: str, log_row: dict, message: str, row=None) -> None:
        with self.lock:
            self.inflight[split] -= 1
            self.attempted += 1
            if status == "ok":
                self.ok[split] = self.ok.get(split, 0) + 1
                self.bot_hits = 0
                self.pauses = 0
            elif any(marker in message.lower() for marker in BOT_MARKERS):
                # Not the song's fault: put it back and slow down instead of burning candidates.
                log_row["status"] = "blocked"
                if row is not None:
                    self.queues[split].appendleft(row)
                self.bot_hits += 1
                if self.bot_hits >= 3:
                    self.bot_hits = 0
                    self.pauses += 1
                    if self.pauses > self.max_pauses:
                        self.stop = True
                        print("YouTube keeps blocking downloads. Stopping; re-run later or add "
                              "--cookies-from-browser chrome (progress is kept).", flush=True)
                    else:
                        self.pause_until = time.time() + self.pause_seconds
                        print(f"YouTube is rate-limiting/asking for sign-in: pausing all jobs for "
                              f"{self.pause_seconds / 60:.0f} min ({self.pauses}/{self.max_pauses}).", flush=True)
            else:
                self.failed += 1
            append_log(self.log_path, log_row)

    def wait_if_paused(self) -> None:
        while True:
            with self.lock:
                remaining = self.pause_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 30))

    def summary(self) -> str:
        with self.lock:
            per_split = {s: f"{self.ok.get(s, 0)}/{q}" for s, q in self.quotas.items()}
            return f"ok per split {per_split}, failed={self.failed}, attempted={self.attempted}"


def quotas_from_args(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, int]:
    if args.quota_json:
        quotas = {normalise_split(k): int(v) for k, v in json.loads(args.quota_json.read_text()).items()}
        return {s: q for s, q in quotas.items() if q > 0}
    if args.limit is None:
        return frame["split"].value_counts().to_dict()
    fractions = frame["split"].value_counts(normalize=True)
    return {split: max(1, round(args.limit * fraction)) for split, fraction in fractions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SONICS real songs from YouTube with yt-dlp.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to SONICS real_songs.csv")
    parser.add_argument("--out-dir", required=True, type=Path, help="Folder for downloaded audio (outside the repo)")
    parser.add_argument("--limit", type=int, help="Number of SUCCESSFUL downloads wanted, split like SONICS. Default: all")
    parser.add_argument("--quota-json", type=Path, help="JSON {split: successful downloads wanted}; overrides --limit")
    parser.add_argument("--splits", nargs="+", help="Only these splits, e.g. --splits train val")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Download whole songs instead of a section")
    parser.add_argument("--section-seconds", type=float, default=40.0, help="Length of the downloaded section")
    parser.add_argument("--section-start", type=float, default=0.25, help="Section start as a fraction of the song")
    parser.add_argument("--codec", default="best", choices=["best", "mp3", "m4a", "opus", "wav"],
                        help="best = keep YouTube's original audio stream (no re-encode)")
    parser.add_argument("--bitrate", type=int, default=192, help="kbps for the extracted audio (mp3/m4a/opus)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate forced for mp3 output")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds each job waits between songs (be polite)")
    parser.add_argument("--pause-minutes", type=float, default=10.0, help="Pause after repeated rate-limit errors")
    parser.add_argument("--cookies-from-browser", help="e.g. chrome, firefox, safari if YouTube asks you to sign in")
    parser.add_argument("--rate-limit", help="Max download speed per job, e.g. 2M")
    parser.add_argument("--retry-failed", action="store_true", help="Retry songs logged as failed")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading")
    args = parser.parse_args()

    frame = load_real_songs(args.csv)
    if args.splits:
        frame = frame[frame["split"].isin(args.splits)]
    quotas = quotas_from_args(frame, args)
    # Candidate order: shuffled within each split; extra candidates replace failed downloads.
    frame = frame[frame["split"].isin(quotas)].sample(frac=1.0, random_state=args.seed)
    available = frame["split"].value_counts().to_dict()
    print("Target successful downloads per split:", quotas, "| candidates available:", available)

    if args.dry_run:
        for _, row in frame.head(20).iterrows():
            section = None if args.full else section_for(row.get("duration"), args.section_seconds, args.section_start)
            print(f"  {row['youtube_id']}  split={row['split']}  section={section}")
        return

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH. Install it first (brew install ffmpeg / winget install ffmpeg).")
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit("yt-dlp is not installed. Run: pip install -U yt-dlp")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / LOG_NAME
    log = read_log(log_path)

    already: dict[str, int] = {}
    todo = []
    for _, row in frame.iterrows():
        youtube_id = row["youtube_id"]
        if existing_file(args.out_dir, youtube_id) is not None:
            already[row["split"]] = already.get(row["split"], 0) + 1
        elif log.get(youtube_id, {}).get("status") == "failed" and not args.retry_failed:
            continue
        else:
            todo.append(row)
    print("Already downloaded per split:", already)

    progress = Progress(quotas, already, log_path, args.pause_minutes)
    started = time.perf_counter()

    progress.set_candidates(todo)

    def worker() -> None:
        while True:
            task = progress.next_task()
            if task is None:
                return
            if isinstance(task, str):
                time.sleep(0.5)
                continue
            row = task
            progress.wait_if_paused()
            split = row["split"]
            youtube_id = row["youtube_id"]
            section = None if args.full else section_for(row.get("duration"), args.section_seconds, args.section_start)
            status, message, path = download_one(youtube_id, args.out_dir, args, section)
            progress.finish(split, status, {
                "youtube_id": youtube_id,
                "id": row["id"],
                "split": split,
                "status": status,
                "path": path.name if path else "",
                "section_start": "" if section is None else section[0],
                "section_seconds": "" if section is None else round(section[1] - section[0], 2),
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, message, row)
            if progress.attempted % 50 == 0:
                print(f"[{(time.perf_counter() - started) / 60:.1f} min] {progress.summary()}", flush=True)
            time.sleep(args.sleep + random.uniform(0, args.sleep / 2))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for future in [pool.submit(worker) for _ in range(max(1, args.jobs))]:
            future.result()

    print(f"Done. {progress.summary()}. Log: {log_path}")
    short = {s: q - progress.ok.get(s, 0) for s, q in quotas.items() if progress.ok.get(s, 0) < q}
    if short:
        reason = "stopped because YouTube is blocking" if progress.stop else "ran out of candidates"
        print(f"Some splits are short of their target ({reason}): {short}. Re-run to continue, or "
              "prepare_clips.py --balance will match the AI side to what was downloaded.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)
