from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .refinement_contract import sha256_file, write_json_atomic
from .refinement_ranking import ReviewCandidate
from .visualize import save_native_side_by_side


def stitch_selected_cams(
    image_shape: tuple[int, int],
    candidates: Sequence[ReviewCandidate],
    cams: Sequence[np.ndarray],
) -> np.ndarray:
    if len(candidates) != len(cams):
        raise ValueError("selected candidates and CAM counts must match")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    output = np.zeros((height, width), dtype=np.float32)
    for candidate, cam in zip(candidates, cams, strict=True):
        values = np.asarray(cam, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("CAMs must be finite two-dimensional arrays")
        box = candidate.box
        x0, y0 = max(0, box.x0), max(0, box.y0)
        x1, y1 = min(width, box.x1), min(height, box.y1)
        if x0 >= x1 or y0 >= y1:
            continue
        target_width = box.x1 - box.x0
        target_height = box.y1 - box.y0
        if values.shape != (target_height, target_width):
            values = cv2.resize(values, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        local = values[y0 - box.y0 : y1 - box.y0, x0 - box.x0 : x1 - box.x0]
        weighted = np.clip(local, 0.0, 1.0) * float(candidate.probability)
        output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], weighted)
    return output


def render_display_heatmap(score_map: np.ndarray, display_threshold: float) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("score_map must be a finite two-dimensional array")
    if not 0.0 <= display_threshold <= 1.0:
        raise ValueError("display_threshold must be between 0 and 1")
    visible = np.where(values >= display_threshold, np.clip(values, 0.0, 1.0), 0.0)
    bgr = cv2.applyColorMap(np.rint(visible * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[visible == 0.0] = 0
    return rgb


def _overlay(rgb: np.ndarray, score_map: np.ndarray, display_threshold: float) -> np.ndarray:
    heatmap = render_display_heatmap(score_map, display_threshold)
    output = rgb.copy()
    visible = score_map >= display_threshold
    blended = np.rint(rgb.astype(np.float32) * 0.55 + heatmap.astype(np.float32) * 0.45).astype(
        np.uint8
    )
    output[visible] = blended[visible]
    return output


def export_review_case(
    rgb: np.ndarray,
    candidates: Sequence[ReviewCandidate],
    cams: Sequence[np.ndarray],
    metadata: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    display_threshold = float(metadata.get("display_threshold", 0.0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "original": output / "original.png",
        "overlay": output / "legrad_overlay.png",
        "comparison": output / "comparison.png",
        "metadata": output / "metadata.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("review output files already exist")
    score_map = stitch_selected_cams(rgb.shape[:2], candidates, cams)
    overlay = _overlay(rgb, score_map, display_threshold)
    Image.fromarray(rgb).save(paths["original"])
    Image.fromarray(overlay).save(paths["overlay"])
    save_native_side_by_side(rgb, overlay, paths["comparison"])
    output_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "metadata"
    }
    payload = {
        **dict(metadata),
        "format_version": 1,
        "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "image_width": rgb.shape[1],
        "image_height": rgb.shape[0],
        "candidates": [asdict(candidate) for candidate in candidates],
        "output_sha256": output_hashes,
    }
    write_json_atomic(paths["metadata"], payload)
    return paths
