from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from .inference import Box, _normalize_batch, tile_boxes


@dataclass(frozen=True, slots=True)
class FullImageMultiTaskPrediction:
    boxes: tuple[Box, ...]
    baseline_probabilities: np.ndarray
    auxiliary_probabilities: np.ndarray
    raw_score_map: np.ndarray
    gated_score_map: np.ndarray

    @property
    def patch_count(self) -> int:
        return len(self.boxes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_multitask_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_backbone_id: str,
    expected_config_digest: str,
) -> dict[str, object]:
    """Verify the complete provenance chain before strict head-only loading."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source}: expected {expected_checkpoint_sha256}, got {actual_sha256}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("unsupported multitask checkpoint format")
    if payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
        raise ValueError("multitask checkpoint base checkpoint SHA-256 mismatch")
    if payload.get("backbone_id") != expected_backbone_id:
        raise ValueError("multitask checkpoint backbone ID mismatch")
    if payload.get("config_digest") != expected_config_digest:
        raise ValueError("multitask checkpoint configuration digest mismatch")
    states = payload.get("multitask_state_dict")
    if not isinstance(states, Mapping):
        raise ValueError("multitask checkpoint is missing head states")
    state_method = getattr(model, "multitask_state_dict", None)
    if not callable(state_method):
        raise TypeError("model must expose multitask_state_dict()")
    expected_names = set(state_method())
    if set(states) != expected_names:
        raise ValueError(
            f"multitask head set mismatch: expected {sorted(expected_names)}, got {sorted(states)}"
        )
    for name in sorted(expected_names):
        module = getattr(model, name, None)
        state = states[name]
        if not isinstance(module, torch.nn.Module) or not isinstance(state, Mapping):
            raise ValueError(f"invalid multitask head state for {name}")
        module.load_state_dict(dict(state), strict=True)
    return {key: value for key, value in payload.items() if key != "multitask_state_dict"}


def predict_multitask_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 64,
    device: torch.device | str | None = None,
    patch_gate_threshold: float = 0.5,
    use_amp: bool = True,
) -> FullImageMultiTaskPrediction:
    """Run native sliding-window localization and uniform overlap averaging."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if batch_size <= 0 or not 0.0 <= patch_gate_threshold <= 1.0:
        raise ValueError("batch size and patch gate threshold are invalid")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    boxes = tuple(tile_boxes(rgb.shape[1], rgb.shape[0], patch_size, stride))
    baseline_probabilities = np.empty(len(boxes), dtype=np.float32)
    auxiliary_probabilities = np.empty(len(boxes), dtype=np.float32)
    raw_sum = np.zeros(rgb.shape[:2], dtype=np.float32)
    raw_count = np.zeros(rgb.shape[:2], dtype=np.uint16)
    gated_sum = np.zeros(rgb.shape[:2], dtype=np.float32)

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(boxes), batch_size):
                batch_boxes = boxes[start : start + batch_size]
                patches = np.stack(
                    [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in batch_boxes], axis=0
                )
                pixel_values = _normalize_batch(patches).to(device, non_blocking=device.type == "cuda")
                amp_context: Any = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with amp_context:
                    output = model(pixel_values)
                baseline = torch.sigmoid(output.baseline_logits).detach().float().cpu().numpy()
                auxiliary = torch.sigmoid(output.auxiliary_logits).detach().float().cpu().numpy()
                masks = torch.sigmoid(output.mask_logits).detach().float().cpu().numpy()[:, 0]
                if baseline.shape != (len(batch_boxes),) or auxiliary.shape != baseline.shape:
                    raise RuntimeError("multitask classifier outputs have invalid shape")
                if masks.shape != (len(batch_boxes), patch_size, patch_size):
                    raise RuntimeError("multitask mask output has invalid shape")
                if not (
                    np.isfinite(baseline).all()
                    and np.isfinite(auxiliary).all()
                    and np.isfinite(masks).all()
                ):
                    raise RuntimeError("multitask model produced non-finite probabilities")
                baseline_probabilities[start : start + len(batch_boxes)] = baseline
                auxiliary_probabilities[start : start + len(batch_boxes)] = auxiliary
                for box, mask, patch_score in zip(batch_boxes, masks, auxiliary, strict=True):
                    region = np.s_[box.y0 : box.y1, box.x0 : box.x1]
                    raw_sum[region] += mask
                    raw_count[region] += 1
                    if float(patch_score) >= patch_gate_threshold:
                        gated_sum[region] += mask
    finally:
        if was_training:
            model.train()

    raw_score_map = np.zeros_like(raw_sum)
    gated_score_map = np.zeros_like(gated_sum)
    np.divide(raw_sum, raw_count, out=raw_score_map, where=raw_count > 0)
    np.divide(gated_sum, raw_count, out=gated_score_map, where=raw_count > 0)
    return FullImageMultiTaskPrediction(
        boxes=boxes,
        baseline_probabilities=baseline_probabilities,
        auxiliary_probabilities=auxiliary_probabilities,
        raw_score_map=raw_score_map,
        gated_score_map=gated_score_map,
    )


def stitch_patch_masks(
    *,
    image_shape: tuple[int, int],
    boxes: Sequence[Box],
    masks: np.ndarray,
    patch_scores: np.ndarray,
    gate_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly average overlapping masks, with and without patch-level gating."""
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    masks = np.asarray(masks, dtype=np.float32)
    patch_scores = np.asarray(patch_scores, dtype=np.float32)
    if masks.ndim != 3 or len(boxes) != masks.shape[0]:
        raise ValueError("masks must have shape [len(boxes), height, width]")
    if patch_scores.shape != (len(boxes),):
        raise ValueError("patch_scores must have shape [len(boxes)]")
    if not np.isfinite(masks).all() or np.any((masks < 0.0) | (masks > 1.0)):
        raise ValueError("masks must contain finite probabilities in [0, 1]")
    if not np.isfinite(patch_scores).all() or not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("patch scores and gate threshold must be finite probabilities")

    raw_sum = np.zeros((height, width), dtype=np.float32)
    raw_count = np.zeros((height, width), dtype=np.uint16)
    gated_sum = np.zeros((height, width), dtype=np.float32)
    for box, mask, patch_score in zip(boxes, masks, patch_scores, strict=True):
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
            raise ValueError("patch box is outside image bounds")
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            raise ValueError("patch box must have positive area")
        expected_shape = (box.y1 - box.y0, box.x1 - box.x0)
        if mask.shape != expected_shape:
            raise ValueError(f"mask shape {mask.shape} does not match box shape {expected_shape}")
        region = np.s_[box.y0 : box.y1, box.x0 : box.x1]
        raw_sum[region] += mask
        raw_count[region] += 1
        if float(patch_score) >= gate_threshold:
            gated_sum[region] += mask

    raw = np.zeros_like(raw_sum)
    gated = np.zeros_like(gated_sum)
    np.divide(raw_sum, raw_count, out=raw, where=raw_count > 0)
    np.divide(gated_sum, raw_count, out=gated, where=raw_count > 0)
    return raw, gated


def load_labelme_mask(
    path: str | Path,
    *,
    image_shape: tuple[int, int],
    expected_sha256: str | None = None,
) -> np.ndarray:
    """Rasterize supported LabelMe shapes without scaling annotation coordinates."""
    height, width = image_shape
    raw = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(f"LabelMe input changed after contract preflight: {path}")
    payload = json.loads(raw.decode("utf-8"))
    annotated_width = int(payload.get("imageWidth", width))
    annotated_height = int(payload.get("imageHeight", height))
    if (annotated_height, annotated_width) != (height, width):
        raise ValueError(
            "LabelMe image size mismatch: "
            f"annotation={(annotated_height, annotated_width)}, image={(height, width)}"
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in payload.get("shapes", []):
        points = np.rint(np.asarray(shape.get("points", []), dtype=np.float64)).astype(np.int32)
        shape_type = str(shape.get("shape_type") or "polygon")
        if shape_type == "polygon":
            if points.shape[0] < 3 or points.shape[1:] != (2,):
                raise ValueError("LabelMe polygon must contain at least three xy points")
            cv2.fillPoly(mask, [points], color=1)
        elif shape_type == "rectangle":
            if points.shape != (2, 2):
                raise ValueError("LabelMe rectangle must contain two xy points")
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), color=1, thickness=-1)
        else:
            raise ValueError(f"unsupported LabelMe shape_type: {shape_type}")
    return mask.astype(bool)


def score_map_metrics(
    score_map: np.ndarray,
    *,
    threshold: float,
    ground_truth: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    score_map = np.asarray(score_map, dtype=np.float32)
    if score_map.ndim != 2 or not np.isfinite(score_map).all():
        raise ValueError("score_map must be a finite two-dimensional array")
    if np.any((score_map < 0.0) | (score_map > 1.0)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("score_map and threshold must be probabilities")
    predicted = score_map >= threshold
    metrics: dict[str, float | int | bool] = {
        "predicted_positive_pixels": int(predicted.sum()),
        "predicted_positive_fraction": float(predicted.mean()),
        "maximum_score": float(score_map.max(initial=0.0)),
        "p99_score": float(np.quantile(score_map, 0.99)),
    }
    if ground_truth is None:
        return metrics
    truth = np.asarray(ground_truth, dtype=bool)
    if truth.shape != score_map.shape:
        raise ValueError("ground_truth shape must match score_map")
    truth_pixels = int(truth.sum())
    if truth_pixels <= 0:
        raise ValueError("ground_truth must contain at least one positive pixel")
    intersection = int(np.logical_and(predicted, truth).sum())
    union = int(np.logical_or(predicted, truth).sum())
    metrics.update(
        {
            "ground_truth_pixels": truth_pixels,
            "ground_truth_hit": intersection > 0,
            "ground_truth_recall": intersection / truth_pixels,
            "ground_truth_iou": intersection / union if union else 0.0,
            "maximum_score_inside_ground_truth": float(score_map[truth].max()),
        }
    )
    return metrics


def render_mask_overlay(
    rgb: np.ndarray,
    score_map: np.ndarray,
    *,
    threshold: float,
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = np.asarray(rgb)
    score_map = np.asarray(score_map, dtype=np.float32)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if score_map.shape != rgb.shape[:2]:
        raise ValueError("score_map shape must match rgb")
    if not 0.0 < alpha <= 1.0 or not 0.0 <= threshold < 1.0:
        raise ValueError("alpha and threshold are outside their supported ranges")
    intensity = np.clip((score_map - threshold) / (1.0 - threshold), 0.0, 1.0)
    active = score_map > threshold
    colors_bgr = cv2.applyColorMap(np.rint(intensity * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    colors = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)
    opacity = (alpha * intensity)[..., None]
    blended = np.rint(rgb.astype(np.float32) * (1.0 - opacity) + colors * opacity).astype(np.uint8)
    output = rgb.copy()
    output[active] = blended[active]
    return output
