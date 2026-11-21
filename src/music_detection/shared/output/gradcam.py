"""Grad-CAM heatmaps: which time-frequency regions drove a ResNet-18 decision."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


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
