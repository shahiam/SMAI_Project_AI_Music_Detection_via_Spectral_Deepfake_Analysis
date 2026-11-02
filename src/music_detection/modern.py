from __future__ import annotations

import argparse
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
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = log_mel_spectrogram(row["path"])
        image = torch.from_numpy(image).unsqueeze(0)
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
        image = image.repeat(3, 1, 1) / 80.0
        return image, torch.tensor(int(row["label"]), dtype=torch.long)


def build_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def run(
    manifest_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    pretrained: bool = True,
) -> dict[str, float]:
    manifest = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=pretrained).to(device)
    train_loader = DataLoader(SpectrogramDataset(manifest[manifest["split"] == "train"]), batch_size=batch_size, shuffle=True)
    test_frame = manifest[manifest["split"] == "test"]
    test_loader = DataLoader(SpectrogramDataset(test_frame), batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            probabilities.extend(torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
    probabilities_array = np.asarray(probabilities)
    predictions = (probabilities_array >= 0.5).astype(int)
    metrics = classification_metrics(labels, predictions, probabilities_array)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "device": str(device)}, output / "resnet18.pt")
    save_metrics(output, metrics, labels, predictions)
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
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    print(run(args.manifest, args.output, args.epochs, pretrained=not args.no_pretrained))
