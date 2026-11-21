from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
import matplotlib.pyplot as plt

from .augmentations import AudioAugmentor
from .data import load_manifest, load_segment, native_duration
from .evaluation import (
    aggregate_clip_probabilities,
    build_window_index,
    cache_key,
    save_predictions,
    summarize,
)
from .features import (
    BANDLIMITED_SAMPLE_RATE,
    SAMPLE_RATE,
    _pad_audio,
    extract_windowed_spectrograms,
    log_mel_spectrogram,
)
from .metrics import save_metrics, tune_threshold


class RandomCropDataset(Dataset):
    """Training dataset: a fresh random window from anywhere in each clip, every epoch.

    Optionally applies audio-domain augmentation (volume, noise, MP3, pitch, stretch) and
    SpecAugment. Nothing is cached, so every epoch sees different crops and augmentations.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        sample_rate: int = BANDLIMITED_SAMPLE_RATE,
        window_duration: float = 5.0,
        augmentor: AudioAugmentor | None = None,
        audio_augment: bool = False,
        seed: int = 0,
    ):
        self.frame = frame.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.augmentor = augmentor
        self.audio_augment = audio_augment
        self.durations = [native_duration(path) for path in self.frame["path"]]
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if self.augmentor is not None:
            self.augmentor.rng = np.random.default_rng(seed + 1)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        latest_start = max(0.0, self.durations[index] - self.window_duration)
        start = float(self.rng.uniform(0.0, latest_start)) if latest_start > 0 else 0.0
        audio = load_segment(row["path"], self.sample_rate, start, self.window_duration)
        audio = _pad_audio(audio, int(self.window_duration * self.sample_rate))
        if self.augmentor is not None and self.audio_augment:
            audio = self.augmentor.apply_audio(audio, sr=self.sample_rate)
        image = log_mel_spectrogram(audio, sr=self.sample_rate)
        if self.augmentor is not None:
            image = self.augmentor.apply_spectrogram(image)
        return _prepare_spectrogram(image), torch.tensor(int(row["label"]), dtype=torch.long)


class WindowDataset(Dataset):
    """Evaluation dataset: every window of every clip, cached as spectrograms on disk."""

    def __init__(
        self,
        windows: pd.DataFrame,
        sample_rate: int = BANDLIMITED_SAMPLE_RATE,
        window_duration: float = 5.0,
        cache_dir: str | Path | None = None,
    ):
        self.windows = windows.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.windows.iloc[index]
        cache_path = None
        if self.cache_dir:
            key = cache_key(row["path"], self.sample_rate, self.window_duration, round(row["start_sec"], 3))
            cache_path = self.cache_dir / f"{key}.npy"
            if cache_path.is_file():
                return _prepare_spectrogram(np.load(cache_path))
        audio = load_segment(row["path"], self.sample_rate, row["start_sec"], self.window_duration)
        audio = _pad_audio(audio, int(self.window_duration * self.sample_rate))
        image = log_mel_spectrogram(audio, sr=self.sample_rate)
        if cache_path is not None:
            np.save(cache_path, image)
        return _prepare_spectrogram(image)


# DataLoader workers are separate processes; without this each one starts a full set of
# BLAS/OpenMP/numba threads and 12 workers on 24 cores thrash (~15x slower augmentation).
_WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def _limit_worker_threads() -> None:
    for name in _WORKER_THREAD_ENV:
        os.environ[name] = "1"


def _seed_worker(worker_id: int) -> None:
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(int(torch.initial_seed() % 2**31))


def track_probabilities(
    model: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    sample_rate: int,
    cache_dir: Path | None,
    batch_size: int = 64,
    workers: int = 0,
    window_duration: float = 5.0,
    windows: pd.DataFrame | None = None,
) -> np.ndarray:
    """Score every window of every clip and return the mean AI probability per clip."""
    if windows is None:
        windows = build_window_index(frame, window_duration, window_duration)
    loader = DataLoader(
        WindowDataset(windows, sample_rate, window_duration, cache_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    outputs = []
    with torch.no_grad():
        for images in loader:
            outputs.append(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
    return aggregate_clip_probabilities(windows, np.concatenate(outputs), len(frame))


def build_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _prepare_spectrogram(image: np.ndarray) -> torch.Tensor:
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    image_tensor = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), size=(224, 224), mode="bilinear"
    ).squeeze(0)
    return ((image_tensor + 80.0) / 80.0).clamp(0.0, 1.0).repeat(3, 1, 1)


def predict_files(
    model_path: str | Path,
    audio_paths: list[str | Path],
    window_duration: float = 5.0,
    hop_duration: float = 5.0,
    aggregation: str = "mean",
    threshold: float | None = None,
    detailed: bool = False,
) -> pd.DataFrame:
    """Predict human/AI labels for uploaded audio files using ResNet-18 with full-track window aggregation."""
    if not audio_paths:
        raise ValueError("At least one audio file is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    sample_rate = int(checkpoint.get("sample_rate", SAMPLE_RATE))
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Determine calibrated threshold
    if threshold is None:
        threshold = float(checkpoint.get("threshold", checkpoint.get("metrics", {}).get("threshold", 0.5)))
    cutoff = float(threshold)

    filenames = []
    aggregated_probabilities = []
    predictions = []
    confidences = []
    flagged_window_counts = []
    total_window_counts = []
    all_segments = []

    with torch.no_grad():
        for path in audio_paths:
            path_obj = Path(path)
            filenames.append(path_obj.name)

            try:
                window_specs = extract_windowed_spectrograms(
                    path_obj,
                    window_duration=window_duration,
                    hop_duration=hop_duration,
                    sr=sample_rate,
                )
                tensors = [_prepare_spectrogram(w["spectrogram"]) for w in window_specs]
                batch = torch.stack(tensors).to(device)
                window_probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            except Exception:
                # Fallback for mock objects or single-chunk
                spec = log_mel_spectrogram(path_obj, sr=sample_rate)
                tensor = _prepare_spectrogram(spec).unsqueeze(0).to(device)
                window_probs = torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
                window_specs = [{"start_sec": 0.0, "end_sec": window_duration, "window_index": 0}]

            if aggregation == "mean":
                agg_prob = float(np.mean(window_probs))
            elif aggregation == "median":
                agg_prob = float(np.median(window_probs))
            elif aggregation == "max":
                agg_prob = float(np.max(window_probs))
            elif aggregation == "min":
                agg_prob = float(np.min(window_probs))
            else:
                agg_prob = float(np.mean(window_probs))

            aggregated_probabilities.append(agg_prob)
            is_ai = bool(agg_prob >= cutoff)
            predictions.append("AI-generated" if is_ai else "Human")

            if agg_prob >= cutoff:
                conf = (agg_prob - cutoff) / max(1.0 - cutoff, 1e-6)
            else:
                conf = (cutoff - agg_prob) / max(cutoff, 1e-6)
            confidences.append(float(np.clip(conf, 0.0, 1.0)))

            flagged_count = int(np.sum(window_probs >= cutoff))
            flagged_window_counts.append(flagged_count)
            total_window_counts.append(len(window_probs))

            segments = [
                {
                    "window_index": i,
                    "start_sec": w["start_sec"],
                    "end_sec": w["end_sec"],
                    "probability": float(window_probs[i]),
                    "flagged": bool(window_probs[i] >= cutoff),
                }
                for i, w in enumerate(window_specs)
            ]
            all_segments.append(segments)

    probabilities_array = np.asarray(aggregated_probabilities)
    if not detailed:
        return pd.DataFrame(
            {
                "filename": filenames,
                "ai_probability": probabilities_array,
                "prediction": predictions,
            }
        )

    return pd.DataFrame(
        {
            "filename": filenames,
            "ai_probability": probabilities_array,
            "prediction": predictions,
            "confidence": np.asarray(confidences),
            "flagged_windows": flagged_window_counts,
            "total_windows": total_window_counts,
            "threshold": [cutoff] * len(filenames),
            "segments": all_segments,
        }
    )


def calibrate_checkpoint_threshold(
    model_path: str | Path,
    manifest_path: str | Path,
    split: str = "val",
    criterion: str = "low_fpr",
    target_fpr: float = 0.02,
    batch_size: int = 64,
    workers: int = 0,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-tune the decision threshold of an existing checkpoint on track-level scores."""
    manifest = load_manifest(manifest_path)
    frame = manifest[manifest["split"] == split]
    if frame.empty:
        frame = manifest[manifest["split"] == "train"]
    frame = frame.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    sample_rate = int(checkpoint.get("sample_rate", SAMPLE_RATE))
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    cache = Path(cache_dir) if cache_dir else Path(model_path).parent / "window_cache" / split
    probabilities = track_probabilities(model, frame, device, sample_rate, cache, batch_size, workers)
    tuning = tune_threshold(frame["label"].to_numpy(), probabilities, criterion=criterion, target_fpr=target_fpr)
    checkpoint["threshold"] = float(tuning["optimal_threshold"])
    checkpoint["threshold_tuning"] = {k: v for k, v in tuning.items() if k != "threshold_curve"}
    torch.save(checkpoint, model_path)
    return tuning


def run(
    manifest_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    pretrained: bool = True,
    workers: int = 4,
    cache_dir: str | Path | None = None,
    resume: str | Path | None = None,
    augment: bool = False,
    audio_augment: bool = False,
    threshold_criterion: str = "low_fpr",
    target_fpr: float = 0.02,
    sample_rate: int = BANDLIMITED_SAMPLE_RATE,
    window_duration: float = 5.0,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Train ResNet-18 on random crops; pick epoch and threshold on track-level validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifest = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=pretrained).to(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir else output / "window_cache"

    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    val_windows = build_window_index(val_frame, window_duration, window_duration)
    test_windows = build_window_index(test_frame, window_duration, window_duration)

    config = {
        "sample_rate": sample_rate,
        "window_duration": window_duration,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "augment": augment,
        "audio_augment": audio_augment,
        "pretrained": pretrained,
        "threshold_criterion": threshold_criterion,
        "target_fpr": target_fpr,
        "seed": seed,
        "manifest": str(manifest_path),
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
        "n_val_windows": len(val_windows),
        "n_test_windows": len(test_windows),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"config: {config}", flush=True)

    if workers > 0:
        _limit_worker_threads()  # inherited by the spawned workers
    augmentor = AudioAugmentor(seed=seed) if (augment or audio_augment) else None
    train_dataset = RandomCropDataset(
        train_frame, sample_rate, window_duration, augmentor=augmentor, audio_augment=audio_augment, seed=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    counts = np.bincount(train_frame["label"].to_numpy(), minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    start_epoch = 0
    best_score = -1.0
    history: list[dict[str, Any]] = []
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("selection_score", -1.0))
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_index % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batch_index}/{len(train_loader)} loss {loss.item():.4f}",
                    flush=True,
                )
        scheduler.step()

        val_probabilities = track_probabilities(
            model, val_frame, device, sample_rate, cache_root / "val", workers=workers,
            window_duration=window_duration, windows=val_windows,
        )
        val_tuning = tune_threshold(
            val_frame["label"].to_numpy(), val_probabilities, criterion=threshold_criterion, target_fpr=target_fpr
        )
        epoch_threshold = float(val_tuning["optimal_threshold"])
        val_metrics = val_tuning["metrics_at_optimal"]
        # Select the epoch on threshold-free AUC (tie-broken by F1) so a lucky threshold can't pick it.
        selection_score = val_metrics["roc_auc"] + 1e-3 * val_metrics["f1"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(1, len(train_loader)),
                "val_auc": val_metrics["roc_auc"],
                "val_f1": val_metrics["f1"],
                "val_fpr": val_metrics["false_positive_rate"],
                "threshold": epoch_threshold,
                "seconds": time.perf_counter() - started,
            }
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": val_metrics,
            "threshold": epoch_threshold,
            "threshold_tuning": {k: v for k, v in val_tuning.items() if k != "threshold_curve"},
            "selection_score": selection_score,
            "sample_rate": sample_rate,
            "window_duration": window_duration,
            "config": config,
            "history": history,
        }
        torch.save(checkpoint, output / "checkpoint_last.pt")
        if selection_score > best_score:
            best_score = selection_score
            torch.save(checkpoint, output / "checkpoint_best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch + 1}/{epochs} loss={history[-1]['train_loss']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_fpr={val_metrics['false_positive_rate']:.4f} threshold={epoch_threshold:.3f} "
            f"elapsed={history[-1]['seconds']:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    best_threshold = float(best_checkpoint["threshold"])

    val_probabilities = track_probabilities(
        model, val_frame, device, sample_rate, cache_root / "val", workers=workers,
        window_duration=window_duration, windows=val_windows,
    )
    test_probabilities = track_probabilities(
        model, test_frame, device, sample_rate, cache_root / "test", workers=workers,
        window_duration=window_duration, windows=test_windows,
    )
    save_predictions(output, "val", val_frame, val_probabilities)
    save_predictions(output, "test", test_frame, test_probabilities)

    metrics_tuned = summarize(test_frame, test_probabilities, best_threshold)
    full_metrics = {
        **metrics_tuned,
        "default_0_5": summarize(test_frame, test_probabilities, 0.5),
        "tuned_threshold": best_threshold,
        "tuning_criterion": threshold_criterion,
        "best_epoch": int(best_checkpoint["epoch"]) + 1,
        "evaluation": "track-level: mean AI probability over all windows of each clip",
        "config": config,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "device": str(device),
            "metrics": {k: v for k, v in metrics_tuned.items() if k != "per_generator"},
            "threshold": best_threshold,
            "threshold_tuning": best_checkpoint["threshold_tuning"],
            "sample_rate": sample_rate,
            "window_duration": window_duration,
            "config": config,
        },
        output / "resnet18.pt",
    )
    labels = test_frame["label"].to_numpy()
    save_metrics(
        output, full_metrics, labels, (test_probabilities >= best_threshold).astype(int), threshold=best_threshold
    )
    print(f"Test metrics (tuned threshold {best_threshold:.3f}): {metrics_tuned}", flush=True)
    return metrics_tuned


def save_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    output_path: str | Path,
    target_layer: nn.Module | None = None,
) -> None:
    """Save a Grad-CAM heatmap for the predicted class of one spectrogram."""
    layer = target_layer or model.layer4[-1]
    activations, gradients = [], []
    forward_hook = layer.register_forward_hook(lambda _, __, value: activations.append(value))
    backward_hook = layer.register_full_backward_hook(lambda _, __, value: gradients.append(value[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image.unsqueeze(0))
        logits[0, logits.argmax(dim=1)].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations[0]).sum(dim=1)).squeeze().detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        plt.imsave(output_path, heatmap, cmap="magma")
    finally:
        forward_hook.remove()
        backward_hook.remove()


def main() -> None:
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
