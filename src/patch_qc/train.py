from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .model import DinoBinaryClassifier


def train_one_epoch(
    model: DinoBinaryClassifier,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float | int]:
    """Train only the binary head for one epoch and return sample-weighted metrics."""
    model.train()
    model.backbone.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        images = batch["image"]
        labels = batch["label"]
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("loader batches must contain tensor image and label fields")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total_examples += count
        total_loss += float(loss.detach()) * count
        total_correct += int(((logits.detach() > 0) == (labels > 0.5)).sum())
    if total_examples == 0:
        raise ValueError("loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }


@torch.inference_mode()
def evaluate_binary(
    model: DinoBinaryClassifier,
    loader: Iterable[dict[str, object]],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        images = batch["image"]
        labels = batch["label"]
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("loader batches must contain tensor image and label fields")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        total_correct += int(((logits > 0) == (labels > 0.5)).sum())
        total_examples += int(labels.numel())
    if total_examples == 0:
        raise ValueError("loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }


def save_probe_checkpoint(
    model: DinoBinaryClassifier,
    path: str | Path,
    *,
    backbone_id: str,
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    """Save only the lightweight trained head and reproducibility metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not all(math.isfinite(float(value)) for key, value in metrics.items() if key != "examples"):
        raise ValueError("checkpoint metrics must be finite")
    torch.save(
        {
            "format_version": 1,
            "backbone_id": backbone_id,
            "hidden_size": model.hidden_size,
            "num_prefix_tokens": model.num_prefix_tokens,
            "epoch": epoch,
            "metrics": metrics,
            "head_state_dict": model.head.state_dict(),
        },
        destination,
    )
