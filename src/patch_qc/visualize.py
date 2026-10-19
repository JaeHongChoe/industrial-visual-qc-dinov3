from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .inference import PatchPrediction


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")


def save_native_side_by_side(
    original: np.ndarray, overlay: np.ndarray, output_path: str | Path
) -> None:
    _validate_rgb(original)
    _validate_rgb(overlay)
    if original.shape != overlay.shape:
        raise ValueError("original and overlay must have the same native shape")
    height, width = original.shape[:2]
    panel = Image.new("RGB", (width * 2, height))
    panel.paste(Image.fromarray(original), (0, 0))
    panel.paste(Image.fromarray(overlay), (width, 0))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination)


def save_prediction_panel(
    original: np.ndarray,
    predictions: Iterable[PatchPrediction],
    *,
    threshold: float,
    output_path: str | Path,
) -> None:
    """Save native original on the left and high-logit patch boxes on the right."""
    _validate_rgb(original)
    overlay_image = Image.fromarray(original.copy())
    draw = ImageDraw.Draw(overlay_image)
    for prediction in sorted(predictions, key=lambda item: item.logit, reverse=True):
        if prediction.logit <= threshold:
            continue
        box = prediction.box
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(255, 20, 20), width=3)
        draw.text((box.x0 + 4, box.y0 + 4), f"{prediction.logit:.3f}", fill=(255, 20, 20))
    save_native_side_by_side(original, np.asarray(overlay_image), output_path)


def save_synthetic_panel(
    original: np.ndarray,
    synthetic: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    *,
    draw_outline: bool = True,
) -> None:
    """Save untouched native patch and a synthetic patch with its support outline."""
    _validate_rgb(original)
    _validate_rgb(synthetic)
    if mask.shape != original.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("mask must be a boolean array matching the images")
    overlay = synthetic.copy()
    if draw_outline:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    save_native_side_by_side(original, overlay, output_path)
