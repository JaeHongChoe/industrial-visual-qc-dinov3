from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True, slots=True)
class PatchPrediction:
    box: Box
    logit: float
    probability: float


def _axis_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        raise ValueError(f"image axis {length} is smaller than native patch size {patch_size}")
    positions = list(range(0, length - patch_size + 1, stride))
    tail = length - patch_size
    if not positions or positions[-1] != tail:
        positions.append(tail)
    return positions


def tile_boxes(width: int, height: int, patch_size: int = 256, stride: int = 128) -> list[Box]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    xs = _axis_positions(width, patch_size, stride)
    ys = _axis_positions(height, patch_size, stride)
    return [Box(x, y, x + patch_size, y + patch_size) for y in ys for x in xs]


def _normalize_batch(rgb_batch: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_batch)).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[None, :, None, None]
    return (tensor - mean) / std


def predict_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> list[PatchPrediction]:
    """Run native sliding-window binary inference and preserve raw logits."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    boxes = tile_boxes(rgb.shape[1], rgb.shape[0], patch_size, stride)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    predictions: list[PatchPrediction] = []
    with torch.inference_mode():
        for start in range(0, len(boxes), batch_size):
            batch_boxes = boxes[start : start + batch_size]
            patches = np.stack(
                [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in batch_boxes],
                axis=0,
            )
            logits = model(_normalize_batch(patches).to(device)).detach().float().cpu()
            probabilities = torch.sigmoid(logits)
            predictions.extend(
                PatchPrediction(box=box, logit=float(logit), probability=float(probability))
                for box, logit, probability in zip(batch_boxes, logits, probabilities, strict=True)
            )
    if was_training:
        model.train()
    return predictions
