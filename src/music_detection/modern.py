from __future__ import annotations

import argparse
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
from .data import load_manifest, load_audio, load_full_audio
from .features import (
    SAMPLE_RATE,
    _pad_audio,
    extract_windowed_spectrograms,
    log_mel_spectrogram,
    slice_audio_windows,
)
from .metrics import classification_metrics, save_metrics, tune_threshold


class SpectrogramDataset(Dataset):
    """Dataset producing log-Mel spectrograms, with support for on-the-fly audio & SpecAugment."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_dir: str | Path | None = None,
        augmentor: AudioAugmentor | None = None,
        window_duration: float = 5.0,
        audio_augment: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augmentor = augmentor
        self.window_duration = window_duration
        self.audio_augment = audio_augment
        if self.cache_dir and not (self.augmentor and self.audio_augment):
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        label = int(row["label"])

        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{index:06d}.npy"

        # Augmented training path
        if self.augmentor is not None:
            # If explicit audio augment or cache missing, load raw audio and apply audio-domain transforms
            if self.audio_augment or not (cache_path and cache_path.is_file()):
                path_str = str(row["path"])
                target_samples = int(self.window_duration * SAMPLE_RATE)
                try:
                    full_audio, sr = load_full_audio(path_str, SAMPLE_RATE)
                    total_samples = len(full_audio)
                    if total_samples > target_samples:
                        max_offset = total_samples - target_samples
                        start = np.random.randint(0, max_offset + 1)
                        audio = full_audio[start : start + target_samples]
                    else:
                        audio = _pad_audio(full_audio, target_samples)
                except Exception:
                    audio, sr = load_audio(path_str, SAMPLE_RATE, duration=self.window_duration)
                    audio = _pad_audio(audio, target_samples)

                audio = self.augmentor.apply_audio(audio, sr=SAMPLE_RATE)
                image = log_mel_spectrogram(audio, sr=SAMPLE_RATE)
            else:
                # Fast path: load precomputed spectrogram and apply SpecAugment
                image = np.load(cache_path)

            image = self.augmentor.apply_spectrogram(image)
            image_tensor = _prepare_spectrogram(image)
            return image_tensor, torch.tensor(label, dtype=torch.long)

        # Deterministic path (validation/test/eval)
        if cache_path and cache_path.is_file():
            image = np.load(cache_path)
        else:
            image = log_mel_spectrogram(row["path"])
            if cache_path:
                np.save(cache_path, image)
        image_tensor = _prepare_spectrogram(image)
        return image_tensor, torch.tensor(label, dtype=torch.long)


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
                    sr=SAMPLE_RATE,
                )
                tensors = [_prepare_spectrogram(w["spectrogram"]) for w in window_specs]
                batch = torch.stack(tensors).to(device)
                window_probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            except Exception:
                # Fallback for mock objects or single-chunk
                spec = log_mel_spectrogram(path_obj)
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
    batch_size: int = 16,
    workers: int = 0,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Calibrate decision threshold on validation split for an existing trained checkpoint."""
    manifest = load_manifest(manifest_path)
    val_frame = manifest[manifest["split"] == split]
    if val_frame.empty:
        val_frame = manifest[manifest["split"] == "train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Auto-detect precomputed spectrogram cache if available
    effective_cache = Path(cache_dir) if cache_dir else Path(model_path).parent / "spectrogram_cache" / split
    cache_to_use = effective_cache if effective_cache.is_dir() else None

    val_loader = DataLoader(
        SpectrogramDataset(val_frame, cache_dir=cache_to_use),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    val_probabilities, val_labels = [], []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            val_probabilities.extend(probs)
            val_labels.extend(batch_labels.numpy())

    tuning = tune_threshold(val_labels, val_probabilities, criterion=criterion, target_fpr=target_fpr)
    checkpoint["threshold"] = float(tuning["optimal_threshold"])
    checkpoint["threshold_tuning"] = tuning
    torch.save(checkpoint, model_path)
    return tuning


def run(
    manifest_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    pretrained: bool = True,
    workers: int = 2,
    cache_dir: str | Path | None = None,
    resume: str | Path | None = None,
    augment: bool = False,
    audio_augment: bool = False,
    threshold_criterion: str = "low_fpr",
    target_fpr: float = 0.02,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=pretrained).to(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir else output / "spectrogram_cache"

    train_frame = manifest[manifest["split"] == "train"]
    val_frame = manifest[manifest["split"] == "val"]
    test_frame = manifest[manifest["split"] == "test"]

    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }

    augmentor = AudioAugmentor() if (augment or audio_augment) else None
    train_dataset = SpectrogramDataset(
        train_frame,
        cache_root / "train",
        augmentor=augmentor,
        audio_augment=audio_augment,
    )
    val_dataset = SpectrogramDataset(val_frame, cache_root / "val", augmentor=None)
    test_dataset = SpectrogramDataset(test_frame, cache_root / "test", augmentor=None)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    counts = np.bincount(train_frame["label"].to_numpy(), minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    start_epoch = 0
    best_f1 = -1.0
    best_threshold = 0.5
    best_tuning = {}

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("metrics", {}).get("f1", -1.0))
        best_threshold = float(checkpoint.get("threshold", 0.5))

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batch_index}/{len(train_loader)} loss {loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        val_probabilities, val_labels = [], []
        with torch.no_grad():
            for images, batch_labels in val_loader:
                probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
                val_probabilities.extend(probs)
                val_labels.extend(batch_labels.numpy())

        val_probabilities_array = np.asarray(val_probabilities)
        val_labels_array = np.asarray(val_labels)

        # Calibrate decision threshold on validation set
        val_tuning = tune_threshold(
            val_labels_array,
            val_probabilities_array,
            criterion=threshold_criterion,
            target_fpr=target_fpr,
        )
        epoch_threshold = val_tuning["optimal_threshold"]
        val_metrics = val_tuning["metrics_at_optimal"]

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": val_metrics,
            "threshold": epoch_threshold,
            "threshold_tuning": val_tuning,
        }
        torch.save(checkpoint, output / "checkpoint_last.pt")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_threshold = epoch_threshold
            best_tuning = val_tuning
            torch.save(checkpoint, output / "checkpoint_best.pt")

        print(
            f"epoch {epoch + 1}/{epochs} val_f1={val_metrics['f1']:.4f} "
            f"tuned_threshold={epoch_threshold:.2f} val_fpr={val_metrics['false_positive_rate']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()

    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            probabilities.extend(probs)
            labels.extend(batch_labels.numpy())

    probabilities_array = np.asarray(probabilities)
    labels_array = np.asarray(labels)

    # Evaluate at both baseline (0.5) and calibrated threshold
    metrics_0_5 = classification_metrics(
        labels_array, (probabilities_array >= 0.5).astype(int), probabilities_array, 0.5
    )
    metrics_tuned = classification_metrics(
        labels_array,
        (probabilities_array >= best_threshold).astype(int),
        probabilities_array,
        best_threshold,
    )

    full_metrics = {
        **metrics_tuned,
        "default_0_5": metrics_0_5,
        "tuned": metrics_tuned,
        "tuned_threshold": best_threshold,
        "tuning_criterion": threshold_criterion,
    }

    final_artifact = {
        "model": model.state_dict(),
        "device": str(device),
        "metrics": metrics_tuned,
        "threshold": best_threshold,
        "threshold_tuning": best_tuning,
    }
    torch.save(final_artifact, output / "resnet18.pt")
    save_metrics(
        output,
        full_metrics,
        labels_array,
        (probabilities_array >= best_threshold).astype(int),
        threshold=best_threshold,
    )
    print(f"Test metrics (default threshold 0.50): {metrics_0_5}", flush=True)
    print(f"Test metrics (tuned threshold {best_threshold:.2f}): {metrics_tuned}", flush=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Enable audio and SpecAugment augmentations during training")
    parser.add_argument("--threshold-criterion", default="low_fpr", choices=["low_fpr", "f1", "youden", "cost_sensitive"])
    parser.add_argument("--target-fpr", type=float, default=0.02)
    args = parser.parse_args()
    print(
        run(
            args.manifest,
            args.output,
            args.epochs,
            batch_size=args.batch_size,
            pretrained=not args.no_pretrained,
            workers=args.workers,
            cache_dir=args.cache_dir,
            resume=args.resume,
            augment=args.augment,
            threshold_criterion=args.threshold_criterion,
            target_fpr=args.target_fpr,
        )
    )
