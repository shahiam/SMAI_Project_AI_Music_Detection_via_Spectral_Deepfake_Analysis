"""Fetch a stratified, class-balanced subset of SONICS onto an external drive.

Subcommands (run them in this order; each one is resumable):

  plan    Download the metadata CSVs, then choose which songs to get:
          * the same number of songs N for AI and human, sized so each class is about
            --gb-per-class GB on disk;
          * the same N per split (train/val/test) for both classes;
          * AI songs stratified by split x generator x SONICS label, so every kind of AI
            song keeps the share it has in the full dataset.
  fakes   Download the AI-song zip parts one after another, extract ONLY the selected
          songs, and delete each zip. All parts are fetched because selected songs are
          spread across them; the disk only ever holds the kept songs plus the zips in flight.
  reals   Download the human songs from YouTube (runs download_youtube.py with the
          per-split targets from the plan). Failed videos are replaced by other songs
          from the same split.
  status  Show progress against the plan.

Example (macOS, external drive):
    caffeinate -i python -m music_detection.sonics.preprocessing.fetch plan  --root /Volumes/BACKUP/SONICS --gb-per-class 15
    caffeinate -i python -m music_detection.sonics.preprocessing.fetch fakes --root /Volumes/BACKUP/SONICS
    caffeinate -i python -m music_detection.sonics.preprocessing.fetch reals --root /Volumes/BACKUP/SONICS --jobs 8
    python -m music_detection.sonics.preprocessing.fetch status --root /Volumes/BACKUP/SONICS

Requirements: pip install -U huggingface_hub pandas yt-dlp ; ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .utils import allocate, normalise_split

REPO_ID = "awsaf49/sonics"
METADATA_FILES = ("fake_songs.csv", "real_songs.csv", "metadata.json")


def use_drive_cache(root: Path) -> None:
    """Keep Hugging Face's cache on the same drive (must run before importing huggingface_hub)."""
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))


def download_metadata(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name in METADATA_FILES:
        if not (root / name).is_file():
            print(f"Downloading {name} ...", flush=True)
            hf_hub_download(REPO_ID, filename=name, repo_type="dataset", local_dir=root)


def list_zip_parts() -> list[tuple[str, int]]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(REPO_ID, path_in_repo="fake_songs", repo_type="dataset")
    parts = [(e.path, int(getattr(e, "size", 0) or 0)) for e in entries if str(e.path).endswith(".zip")]
    return sorted(parts)


def load_fake_table(root: Path) -> pd.DataFrame:
    fakes = pd.read_csv(root / "fake_songs.csv")
    name_col = "filepath" if "filepath" in fakes.columns else "filename"
    fakes["stem"] = fakes[name_col].astype(str).map(lambda v: Path(v).stem)
    fakes["split"] = fakes["split"].map(normalise_split)
    for column, default in (("algorithm", "unknown"), ("label", "fake")):
        values = fakes[column] if column in fakes.columns else pd.Series(default, index=fakes.index)
        fakes[column] = values.fillna(default).astype(str)
    return fakes.drop_duplicates("stem")


def cmd_plan(args: argparse.Namespace) -> None:
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    download_metadata(root)
    fakes = load_fake_table(root)
    reals = pd.read_csv(root / "real_songs.csv")
    reals = reals[reals["youtube_id"].notna()].drop_duplicates("youtube_id")
    reals["split"] = reals["split"].map(normalise_split)

    parts = list_zip_parts()
    zip_bytes = sum(size for _, size in parts)
    fake_mb = zip_bytes / max(len(fakes), 1) / 1e6 if zip_bytes else args.fake_mb_per_song
    budget_mb = args.gb_per_class * 1000
    n_by_fake_budget = int(budget_mb / fake_mb)
    n_by_real_budget = int(budget_mb / args.real_mb_per_song)
    n_total = min(n_by_fake_budget, n_by_real_budget, len(fakes))
    print(f"AI zips on Hugging Face: {len(parts)} parts, {zip_bytes / 1e9:.1f} GB for {len(fakes)} songs "
          f"-> ~{fake_mb:.2f} MB per AI song")
    print(f"Human songs: ~{args.real_mb_per_song:.2f} MB per downloaded {args.section_seconds:.0f} s section")
    print(f"Songs per class: AI budget allows {n_by_fake_budget}, human budget allows {n_by_real_budget} -> N = {n_total}")

    # Same N per split for both classes; split shares follow the AI side of SONICS, capped so that
    # enough human candidates exist (assuming --real-success of YouTube downloads work).
    fake_split = fakes["split"].value_counts().to_dict()
    real_cap = {s: int(reals["split"].eq(s).sum() * args.real_success) for s in fake_split}
    split_quota = allocate(n_total, fake_split, {s: min(fake_split[s], real_cap.get(s, 0)) for s in fake_split})
    split_quota = {s: q for s, q in split_quota.items() if s in ("train", "val", "test") and q > 0}

    chosen = []
    for split, quota in split_quota.items():
        pool = fakes[fakes["split"] == split]
        strata = pool.groupby(["algorithm", "label"]).size().to_dict()
        for (algorithm, label), n in allocate(quota, strata).items():
            group = pool[(pool["algorithm"] == algorithm) & (pool["label"] == label)]
            chosen.append(group.sample(n=n, random_state=args.seed))
    selection = pd.concat(chosen)

    plan_dir = root / "plan"
    plan_dir.mkdir(exist_ok=True)
    keep = [c for c in ["stem", "split", "algorithm", "label", "duration", "filepath", "filename"] if c in selection]
    selection[keep].to_csv(plan_dir / "fake_selection.csv", index=False)
    (plan_dir / "real_quota.json").write_text(json.dumps(split_quota, indent=2))
    (plan_dir / "settings.json").write_text(json.dumps({
        "gb_per_class": args.gb_per_class, "songs_per_class": int(len(selection)),
        "fake_mb_per_song": round(fake_mb, 3), "real_mb_per_song": args.real_mb_per_song,
        "section_seconds": args.section_seconds, "seed": args.seed,
    }, indent=2))

    full = fakes.groupby(["split", "algorithm", "label"]).size().rename("full_dataset")
    picked = selection.groupby(["split", "algorithm", "label"]).size().rename("selected")
    table = pd.concat([full, picked], axis=1).fillna(0).astype(int)
    table["full_share_in_split_%"] = (100 * table["full_dataset"] / table.groupby(level=0)["full_dataset"].transform("sum")).round(1)
    table["selected_share_in_split_%"] = (100 * table["selected"] / table.groupby(level=0)["selected"].transform("sum").clip(lower=1)).round(1)
    pd.set_option("display.width", 160)
    print("\nAI songs: full dataset vs selection (shares within each split should match):")
    print(table.to_string())
    print(f"\nSongs per split, identical for AI and human: {split_quota} (total {sum(split_quota.values())} per class)")
    print(f"Estimated size: AI ~{len(selection) * fake_mb / 1000:.1f} GB, human ~{sum(split_quota.values()) * args.real_mb_per_song / 1000:.1f} GB")
    print(f"Plan written to {plan_dir}. Next: fakes, then reals.")


def cmd_fakes(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    root = args.root
    selection = pd.read_csv(root / "plan" / "fake_selection.csv")
    wanted = set(selection["stem"].astype(str))
    out_dir = root / "fake_songs"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = root / "plan" / "fake_parts_done.txt"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()
    parts = [p for p, _ in list_zip_parts() if p not in done]
    print(f"{len(wanted)} AI songs selected; {len(parts)} zip parts left to process.", flush=True)
    lock = threading.Lock()

    def process(part: str) -> None:
        zip_path = Path(hf_hub_download(REPO_ID, filename=part, repo_type="dataset", local_dir=root / "_zips"))
        kept = 0
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                name = Path(info.filename).name
                if info.is_dir() or name.startswith("._") or Path(name).stem not in wanted:
                    continue
                target = out_dir / name
                if target.is_file() and target.stat().st_size == info.file_size:
                    kept += 1
                    continue
                tmp = target.with_name(target.name + ".part")
                with archive.open(info) as source, open(tmp, "wb") as sink:
                    shutil.copyfileobj(source, sink, 1 << 20)
                tmp.replace(target)
                kept += 1
        zip_path.unlink(missing_ok=True)
        with lock:
            with done_path.open("a") as handle:
                handle.write(part + "\n")
            have = sum(1 for p in out_dir.iterdir() if p.suffix.lower() == ".mp3")
            print(f"{part}: kept {kept} selected songs (total on drive {have}/{len(wanted)})", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.part_jobs)) as pool:
        list(pool.map(process, parts))
    shutil.rmtree(root / "_zips", ignore_errors=True)
    have = {p.stem for p in out_dir.iterdir() if p.suffix.lower() == ".mp3"}
    print(f"Done: {len(have & wanted)}/{len(wanted)} selected AI songs on the drive.")


def cmd_reals(args: argparse.Namespace, passthrough: list[str]) -> None:
    root = args.root
    command = [
        sys.executable, "-m", "music_detection.sonics.preprocessing.download_youtube",
        "--csv", str(root / "real_songs.csv"),
        "--out-dir", str(root / "real_songs"),
        "--quota-json", str(root / "plan" / "real_quota.json"),
        "--jobs", str(args.jobs),
        "--section-seconds", str(args.section_seconds),
        *passthrough,
    ]
    print("Running:", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command))


def folder_stats(folder: Path, exts: tuple[str, ...]) -> tuple[int, float]:
    if not folder.is_dir():
        return 0, 0.0
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return len(files), sum(p.stat().st_size for p in files) / 1e9


def cmd_status(args: argparse.Namespace) -> None:
    root = args.root
    quota = json.loads((root / "plan" / "real_quota.json").read_text())
    selection = pd.read_csv(root / "plan" / "fake_selection.csv")
    n_fake, gb_fake = folder_stats(root / "fake_songs", (".mp3",))
    n_real, gb_real = folder_stats(root / "real_songs", (".mp3", ".m4a", ".opus", ".webm", ".ogg", ".wav"))
    print(f"AI songs:    {n_fake}/{len(selection)} ({gb_fake:.1f} GB)")
    print(f"Human songs: {n_real}/{sum(quota.values())} ({gb_real:.1f} GB)   targets per split: {quota}")
    log = root / "real_songs" / "download_log.csv"
    if log.is_file():
        frame = pd.read_csv(log)
        print("YouTube log:", frame.groupby(["split", "status"]).size().to_dict())
    free = shutil.disk_usage(root).free / 1e9
    print(f"Free space on drive: {free:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a stratified, balanced SONICS subset.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fakes", "reals", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("/Volumes/BACKUP/SONICS"))
        p.add_argument("--section-seconds", type=float, default=40.0)
        if name == "plan":
            p.add_argument("--gb-per-class", type=float, default=15.0)
            p.add_argument("--real-mb-per-song", type=float, default=0.7,
                           help="Expected size of one downloaded human section (40 s Opus ~0.6-0.8 MB)")
            p.add_argument("--fake-mb-per-song", type=float, default=0.7, help="Fallback if zip sizes are unknown")
            p.add_argument("--real-success", type=float, default=0.8,
                           help="Expected fraction of YouTube videos that still download")
            p.add_argument("--seed", type=int, default=42)
        if name == "fakes":
            p.add_argument("--part-jobs", type=int, default=2, help="Zip parts downloaded at the same time")
        if name == "reals":
            p.add_argument("--jobs", type=int, default=8)
    args, passthrough = parser.parse_known_args()
    if args.command != "reals" and passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    use_drive_cache(args.root)
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "fakes":
        cmd_fakes(args)
    elif args.command == "reals":
        cmd_reals(args, passthrough)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
