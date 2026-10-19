from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .inference import Box


@dataclass(frozen=True, slots=True)
class EncodedPatch:
    box: Box
    logit: float
    probability: float
    feature: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    probability_floor: float
    minimum_top_m: int
    maximum_candidates: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_floor <= 1.0:
            raise ValueError("probability_floor must be between 0 and 1")
        if self.minimum_top_m < 0 or self.maximum_candidates <= 0:
            raise ValueError("candidate counts must be non-negative and maximum must be positive")
        if self.minimum_top_m > self.maximum_candidates:
            raise ValueError("minimum_top_m cannot exceed maximum_candidates")


def _normalize_batch_on_device(
    rgb_batch: np.ndarray, device: torch.device
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb_batch))
        .permute(0, 3, 1, 2)
        .to(device, non_blocking=device.type == "cuda")
        .float()
        .div_(255.0)
    )
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype, device=device)[
        None, :, None, None
    ]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype, device=device)[
        None, :, None, None
    ]
    return (tensor - mean) / std


def encode_patch_batch(
    model: torch.nn.Module,
    rgb_batch: np.ndarray,
    device: torch.device | str,
    *,
    precision: str = "fp32",
) -> tuple[np.ndarray, np.ndarray]:
    if rgb_batch.ndim != 4 or rgb_batch.shape[-1] != 3 or rgb_batch.dtype != np.uint8:
        raise ValueError("rgb_batch must be an NxHxWx3 uint8 array")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            normalized = _normalize_batch_on_device(rgb_batch, target_device)
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.float16,
                enabled=target_device.type == "cuda" and precision == "fp16",
            ):
                logits, pooled, _ = model.encode_with_logits(normalized)
            return (
                logits.detach().float().cpu().numpy().copy(),
                pooled.detach().float().cpu().numpy().copy(),
            )
    finally:
        model.train(was_training)


def select_candidate_indices(probabilities: np.ndarray, policy: CandidatePolicy) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be between 0 and 1")
    order = np.argsort(-values, kind="stable")
    keep = set(np.flatnonzero(values >= policy.probability_floor).tolist())
    keep.update(order[: policy.minimum_top_m].tolist())
    selected = [int(index) for index in order if int(index) in keep]
    return np.asarray(selected[: policy.maximum_candidates], dtype=np.int64)


def _validated_cases(
    cases: Sequence[Mapping[str, object]], expected_label: str
) -> list[tuple[np.ndarray, tuple[int, ...]]]:
    validated: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for case in cases:
        source_label = case.get("source_label")
        if source_label != expected_label:
            raise ValueError(f"{source_label or 'missing'} input is not allowed; expected {expected_label}")
        probabilities = np.asarray(case.get("probabilities", ()), dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("each calibration case requires one-dimensional probabilities")
        targets = tuple(int(index) for index in case.get("target_indices", ()))
        if any(index < 0 or index >= probabilities.size for index in targets):
            raise ValueError("target index is outside the probability array")
        validated.append((probabilities, targets))
    return validated


def calibrate_candidate_policy(
    clean_cases: Sequence[Mapping[str, object]],
    synthetic_cases: Sequence[Mapping[str, object]],
    floors: Sequence[float],
    top_m_values: Sequence[int],
    maximum_candidates: int,
    target_local_recall: float,
) -> CandidatePolicy:
    clean = _validated_cases(clean_cases, "OK")
    synthetic = _validated_cases(synthetic_cases, "SYNTHETIC")
    if not 0.0 <= target_local_recall <= 1.0:
        raise ValueError("target_local_recall must be between 0 and 1")
    target_count = sum(len(targets) for _, targets in synthetic)
    if target_count == 0:
        raise ValueError("synthetic calibration requires target indices")

    eligible: list[tuple[tuple[float, int, float, int], CandidatePolicy]] = []
    for floor in floors:
        for top_m in top_m_values:
            policy = CandidatePolicy(float(floor), int(top_m), int(maximum_candidates))
            found = 0
            for probabilities, targets in synthetic:
                selected = set(select_candidate_indices(probabilities, policy).tolist())
                found += sum(index in selected for index in targets)
            recall = found / target_count
            if recall < target_local_recall:
                continue
            clean_counts = [len(select_candidate_indices(probabilities, policy)) for probabilities, _ in clean]
            mean_clean = float(np.mean(clean_counts)) if clean_counts else 0.0
            maximum_clean = max(clean_counts, default=0)
            objective = (mean_clean, maximum_clean, -policy.probability_floor, policy.minimum_top_m)
            eligible.append((objective, policy))

    if not eligible:
        raise ValueError("no candidate policy meets target local recall")
    return min(eligible, key=lambda item: item[0])[1]
