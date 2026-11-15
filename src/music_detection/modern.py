from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
import matplotlib.pyplot as plt

from .data import load_manifest
from .features import log_mel_spectrogram
from .metrics import classification_metrics, save_metrics


class SpectrogramDataset(Dataset):
    def __init__(self, frame, cache_dir: str | Path | None = None):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{index:06d}.npy"
        if cache_path and cache_path.is_file():
            image = np.load(cache_path)
        else:
            image = log_mel_spectrogram(row["path"])
            if cache_path:
                np.save(cache_path, image)
        image = torch.from_numpy(image).unsqueeze(0)
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
        image = ((image + 80.0) / 80.0).clamp(0.0, 1.0).repeat(3, 1, 1)
        return image, torch.tensor(int(row["label"]), dtype=torch.long)


def build_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def run(
    manifest_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    pretrained: bool = True,
    workers: int = 2,
    cache_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> dict[str, float]:
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
    train_loader = DataLoader(
        SpectrogramDataset(train_frame, cache_root / "train"),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        SpectrogramDataset(val_frame, cache_root / "val"),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        SpectrogramDataset(test_frame, cache_root / "test"),
        shuffle=False,
        **loader_options,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    counts = np.bincount(train_frame["label"].to_numpy(), minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    start_epoch = 0
    best_f1 = -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("metrics", {}).get("f1", -1.0))
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print(f"epoch {epoch + 1}/{epochs} batch {batch_index}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        model.eval()
        val_probabilities, val_labels = [], []
        with torch.no_grad():
            for images, batch_labels in val_loader:
                val_probabilities.extend(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
                val_labels.extend(batch_labels.numpy())
        val_probabilities_array = np.asarray(val_probabilities)
        val_predictions = (val_probabilities_array >= 0.5).astype(int)
        val_metrics = classification_metrics(val_labels, val_predictions, val_probabilities_array)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": val_metrics,
        }
        torch.save(checkpoint, output / "checkpoint_last.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(checkpoint, output / "checkpoint_best.pt")
        print(
            f"epoch {epoch + 1}/{epochs} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(output / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            probabilities.extend(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
    probabilities_array = np.asarray(probabilities)
    predictions = (probabilities_array >= 0.5).astype(int)
    metrics = classification_metrics(labels, predictions, probabilities_array)
    torch.save({"model": model.state_dict(), "device": str(device), "metrics": metrics}, output / "resnet18.pt")
    save_metrics(output, metrics, labels, predictions)
    print(f"test metrics: {metrics}", flush=True)
    return metrics


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
        )
    )
