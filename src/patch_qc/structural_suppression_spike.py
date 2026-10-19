from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import unicodedata

import numpy as np
import torch

from .inference import Box, _normalize_batch, tile_boxes


_CANDIDATE_ID = "auxiliary_weighted_max_v1"
_CANDIDATE_FORMULA = (
    "max_over_overlaps(mask_probability_times_auxiliary_probability)"
)


def validate_structural_candidate_spec(
    spec: Mapping[str, object],
) -> dict[str, object]:
    if spec.get("candidate_id") != _CANDIDATE_ID:
        raise ValueError("unexpected candidate ID")
    if spec.get("interpretation") != "development_diagnostic_not_sealed_test":
        raise ValueError("candidate must remain a development diagnostic")
    if spec.get("formula") != _CANDIDATE_FORMULA:
        raise ValueError("candidate fusion formula differs from the frozen contract")
    evaluation = dict(spec.get("evaluation", {}))
    if float(evaluation.get("decision_threshold", -1.0)) != 0.5:
        raise ValueError("decision threshold must remain 0.5")
    if evaluation.get("decision_comparator") != "strictly_greater_than":
        raise ValueError("decision comparator must be strictly_greater_than")
    if evaluation.get("visualization_comparator") != "strictly_greater_than":
        raise ValueError("visualization comparator must match decision comparator")
    if evaluation.get("parameter_sweeps_allowed") != 0:
        raise ValueError("parameter sweeps are forbidden")
    if evaluation.get("localized_ng_required_no_loss_vs_raw") is not True:
        raise ValueError("localized NG no-loss gate must be enabled")
    if float(
        evaluation.get("minimum_ok_positive_pixel_reduction_fraction_vs_raw", -1.0)
    ) != 0.1:
        raise ValueError("minimum OK reduction must remain 0.1")

    inputs = dict(spec.get("inputs", {}))
    expected_counts = {
        "source_results_count": 115,
        "ok_count": 27,
        "ng_count": 88,
        "localized_ng_count": 80,
    }
    if any(inputs.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("frozen source cohort counts differ from the contract")

    digest_keys = (
        "source_receipt_sha256",
        "base_checkpoint_sha256",
        "multitask_checkpoint_sha256",
        "training_receipt_sha256",
    )
    digests: dict[str, str] = {}
    for key in digest_keys:
        value = str(inputs.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
        digests[key] = value

    inference = dict(spec.get("inference", {}))
    expected_inference = {
        "patch_size": 256,
        "stride": 128,
        "batch_size": 128,
        "amp": True,
    }
    for key, value in expected_inference.items():
        if inference.get(key) != value:
            raise ValueError(f"inference {key} differs from the frozen contract")

    return {
        "decision_threshold": 0.5,
        "patch_size": 256,
        "stride": 128,
        "batch_size": 128,
        "use_amp": True,
        **digests,
    }


def stitch_auxiliary_weighted_max(
    *,
    image_shape: tuple[int, int],
    boxes: Sequence[Box],
    masks: np.ndarray,
    auxiliary_scores: np.ndarray,
) -> np.ndarray:
    """Fuse overlapping mask predictions without raw-overlap denominator dilution.

    Each patch contributes ``mask_probability * auxiliary_probability`` and the
    strongest supported observation wins at every native-resolution pixel.
    """
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    mask_values = np.asarray(masks, dtype=np.float32)
    scores = np.asarray(auxiliary_scores, dtype=np.float32)
    if mask_values.ndim != 3 or mask_values.shape[0] != len(boxes):
        raise ValueError("masks must have shape [len(boxes), height, width]")
    if scores.shape != (len(boxes),):
        raise ValueError("auxiliary_scores must have shape [len(boxes)]")
    if not np.isfinite(mask_values).all() or np.any(
        (mask_values < 0.0) | (mask_values > 1.0)
    ):
        raise ValueError("masks must contain finite probabilities")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("auxiliary_scores must contain finite probabilities")

    fused = np.zeros((height, width), dtype=np.float32)
    for box, mask, score in zip(boxes, mask_values, scores, strict=True):
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
            raise ValueError("patch box is outside image bounds")
        expected = (box.y1 - box.y0, box.x1 - box.x0)
        if min(expected) <= 0 or mask.shape != expected:
            raise ValueError("mask shape does not match its patch box")
        region = np.s_[box.y0 : box.y1, box.x0 : box.x1]
        np.maximum(fused[region], mask * float(score), out=fused[region])
    return fused


@dataclass(frozen=True, slots=True)
class AuxiliaryMaxPrediction:
    boxes: tuple[Box, ...]
    baseline_probabilities: np.ndarray
    auxiliary_probabilities: np.ndarray
    raw_score_map: np.ndarray
    auxiliary_max_score_map: np.ndarray

    @property
    def patch_count(self) -> int:
        return len(self.boxes)


def predict_auxiliary_max_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 128,
    device: torch.device | str | None = None,
    use_amp: bool = True,
) -> AuxiliaryMaxPrediction:
    """Run one frozen model pass and emit raw-mean plus auxiliary-weighted max maps."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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
    auxiliary_max = np.zeros(rgb.shape[:2], dtype=np.float32)

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(boxes), batch_size):
                batch_boxes = boxes[start : start + batch_size]
                patches = np.stack(
                    [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in batch_boxes],
                    axis=0,
                )
                pixel_values = _normalize_batch(patches).to(
                    device, non_blocking=device.type == "cuda"
                )
                amp_context: Any = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_enabled
                    else nullcontext()
                )
                with amp_context:
                    output = model(pixel_values)
                baseline = (
                    torch.sigmoid(output.baseline_logits)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                auxiliary = (
                    torch.sigmoid(output.auxiliary_logits)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                masks = (
                    torch.sigmoid(output.mask_logits)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()[:, 0]
                )
                expected_batch = len(batch_boxes)
                if baseline.shape != (expected_batch,) or auxiliary.shape != (
                    expected_batch,
                ):
                    raise RuntimeError("multitask classifier outputs have invalid shape")
                if masks.shape != (expected_batch, patch_size, patch_size):
                    raise RuntimeError("multitask mask output has invalid shape")
                if not (
                    np.isfinite(baseline).all()
                    and np.isfinite(auxiliary).all()
                    and np.isfinite(masks).all()
                ):
                    raise RuntimeError("multitask model produced non-finite probabilities")
                baseline_probabilities[start : start + expected_batch] = baseline
                auxiliary_probabilities[start : start + expected_batch] = auxiliary
                for box, mask, score in zip(
                    batch_boxes, masks, auxiliary, strict=True
                ):
                    region = np.s_[box.y0 : box.y1, box.x0 : box.x1]
                    raw_sum[region] += mask
                    raw_count[region] += 1
                    np.maximum(
                        auxiliary_max[region],
                        mask * float(score),
                        out=auxiliary_max[region],
                    )
    finally:
        if was_training:
            model.train()

    raw = np.zeros_like(raw_sum)
    np.divide(raw_sum, raw_count, out=raw, where=raw_count > 0)
    return AuxiliaryMaxPrediction(
        boxes=boxes,
        baseline_probabilities=baseline_probabilities,
        auxiliary_probabilities=auxiliary_probabilities,
        raw_score_map=raw,
        auxiliary_max_score_map=auxiliary_max,
    )


def summarize_auxiliary_max_results(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    localized = [row for row in rows if row.get("label_status") == "localized"]
    ok_rows = [row for row in rows if row.get("cohort") == "ok"]
    white_spot = [
        row
        for row in localized
        if unicodedata.normalize(
            "NFC", str(dict(row.get("metadata", {})).get("defect_type", ""))
        )
        == "접합면 White Spot"
    ]

    def hits(key: str, selected: Sequence[Mapping[str, object]]) -> int:
        return sum(
            bool(dict(row[f"{key}_metrics"]).get("ground_truth_hit", False))
            for row in selected
        )

    ok_pixels = {
        key: sum(
            int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"])
            for row in ok_rows
        )
        for key in ("raw", "candidate")
    }
    raw_ok = ok_pixels["raw"]
    reduction = (
        (raw_ok - ok_pixels["candidate"]) / raw_ok if raw_ok > 0 else 0.0
    )
    localized_hits = {
        key: hits(key, localized) for key in ("raw", "candidate")
    }
    per_ok_delta = [
        int(dict(row["candidate_metrics"])["predicted_positive_pixels"])
        - int(dict(row["raw_metrics"])["predicted_positive_pixels"])
        for row in ok_rows
    ]
    no_hit_loss = localized_hits["candidate"] >= localized_hits["raw"]
    reduction_met = reduction >= 0.10

    def ok_subgroup(identifier: str) -> str:
        try:
            number = int(identifier.removeprefix("ok_"))
        except ValueError:
            return "unclassified"
        if 1 <= number <= 8:
            return "a"
        if 9 <= number <= 16:
            return "b"
        if 17 <= number <= 24:
            return "c"
        if 25 <= number <= 27:
            return "extra"
        return "unclassified"

    subgroup_summary: dict[str, object] = {}
    for subgroup in ("a", "b", "c", "extra", "unclassified"):
        selected = [
            row
            for row in ok_rows
            if ok_subgroup(str(row.get("id", ""))) == subgroup
        ]
        if not selected and subgroup == "unclassified":
            continue
        positive_pixels = {
            key: sum(
                int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"])
                for row in selected
            )
            for key in ("raw", "candidate")
        }
        raw_pixels = positive_pixels["raw"]
        subgroup_summary[subgroup] = {
            "denominator": len(selected),
            "positive_pixels": positive_pixels,
            "images_with_activation": {
                key: sum(
                    int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"])
                    > 0
                    for row in selected
                )
                for key in ("raw", "candidate")
            },
            "reduction_fraction_vs_raw": (
                (raw_pixels - positive_pixels["candidate"]) / raw_pixels
                if raw_pixels > 0
                else 0.0
            ),
        }
    return {
        "localized_denominator": len(localized),
        "localized_hits": localized_hits,
        "ok_denominator": len(ok_rows),
        "ok_positive_pixels": ok_pixels,
        "ok_images_with_activation": {
            key: sum(
                int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"]) > 0
                for row in ok_rows
            )
            for key in ("raw", "candidate")
        },
        "ok_reduction_fraction_vs_raw": reduction,
        "ok_subgroups": subgroup_summary,
        "ok_candidate_minus_raw_pixels": {
            "minimum": min(per_ok_delta, default=0),
            "median": float(np.median(per_ok_delta)) if per_ok_delta else 0.0,
            "maximum": max(per_ok_delta, default=0),
        },
        "white_spot": {
            "denominator": len(white_spot),
            "localized_hits": {
                key: hits(key, white_spot) for key in ("raw", "candidate")
            },
        },
        "screening_gate": {
            "localized_no_loss_vs_raw": no_hit_loss,
            "minimum_ok_positive_pixel_reduction_fraction_vs_raw": 0.10,
            "observed_ok_positive_pixel_reduction_fraction_vs_raw": reduction,
            "minimum_ok_reduction_met": reduction_met,
            "passed": no_hit_loss and reduction_met,
        },
    }
