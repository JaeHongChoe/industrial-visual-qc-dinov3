from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .actual_multitask_inference import render_mask_overlay


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    aligned_reference: np.ndarray
    warp_matrix: np.ndarray
    shift_xy: tuple[float, float]
    response: float
    overlap_fraction: float
    accepted: bool
    reason: str
    method: str
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class PriorAggregationResult:
    prior: np.ndarray
    reference_score_ceiling: np.ndarray
    usable: bool
    accepted_count: int
    total_count: int
    registrations: tuple[RegistrationResult, ...]


@dataclass(frozen=True, slots=True)
class PeakPreservedScores:
    baseline: np.ndarray
    candidate: np.ndarray
    compact_preserve_mask: np.ndarray
    native_preserve_mask: np.ndarray
    native_recurrence_prior: np.ndarray


def _probability_map(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return array


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0.8)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    maximum = float(magnitude.max(initial=0.0))
    return magnitude / maximum if maximum > 1e-8 else np.zeros_like(magnitude)


def _sift_similarity_registration(
    reference_map: np.ndarray,
    query_map: np.ndarray,
) -> RegistrationResult | None:
    reference_u8 = np.rint(reference_map * 255.0).astype(np.uint8)
    query_u8 = np.rint(query_map * 255.0).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.01)
    reference_keys, reference_descriptors = sift.detectAndCompute(reference_u8, None)
    query_keys, query_descriptors = sift.detectAndCompute(query_u8, None)
    if reference_descriptors is None or query_descriptors is None:
        return None
    matches = cv2.BFMatcher().knnMatch(reference_descriptors, query_descriptors, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 8:
        return None
    source = np.float32([reference_keys[item.queryIdx].pt for item in good])
    target = np.float32([query_keys[item.trainIdx].pt for item in good])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.sum())
    inlier_ratio = float(inlier_mask.mean())
    scale = float(math.hypot(float(matrix[0, 0]), float(matrix[1, 0])))
    if inliers < 6 or inlier_ratio < 0.25 or not 0.8 <= scale <= 1.2:
        return None
    height, width = query_map.shape
    coverage = cv2.warpAffine(
        np.ones(reference_map.shape, dtype=np.uint8),
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlap = float(coverage.mean())
    if overlap < 0.6:
        return None
    aligned = cv2.warpAffine(
        reference_map,
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return RegistrationResult(
        aligned_reference=np.clip(aligned, 0.0, 1.0).astype(np.float32, copy=False),
        warp_matrix=matrix.astype(np.float32),
        shift_xy=(float(matrix[0, 2]), float(matrix[1, 2])),
        response=inlier_ratio,
        overlap_fraction=overlap,
        accepted=True,
        reason="accepted",
        method="sift_similarity",
        feature_inliers=inliers,
        feature_inlier_ratio=inlier_ratio,
        scale=scale,
    )


def register_reference_thumbnail(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    max_translation_px: float = 8.0,
    min_response: float = 0.15,
    min_overlap_fraction: float = 0.95,
    allow_feature_fallback: bool = False,
) -> RegistrationResult:
    """Align an OK reference to a query with bounded Sobel phase correlation."""
    reference_map = _probability_map(reference, name="reference")
    query_map = _probability_map(query, name="query")
    if reference_map.shape != query_map.shape:
        raise ValueError("reference and query must have the same shape")
    if max_translation_px < 0 or not math.isfinite(max_translation_px):
        raise ValueError("max_translation_px must be finite and non-negative")
    if not 0.0 <= min_response <= 1.0 or not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError("registration thresholds must be in [0, 1]")

    height, width = query_map.shape
    reference_edges = _sobel_magnitude(reference_map)
    query_edges = _sobel_magnitude(query_map)
    if float(reference_edges.max(initial=0.0)) == 0.0 or float(
        query_edges.max(initial=0.0)
    ) == 0.0:
        return RegistrationResult(
            reference_map.copy(),
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (0.0, 0.0),
            0.0,
            1.0,
            False,
            "response_below_minimum",
            "none",
        )
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(reference_edges, query_edges, window)
    response_value = float(response)
    overlap = max(0.0, (width - abs(dx)) / width) * max(0.0, (height - abs(dy)) / height)
    phase_reason = (
        "response_below_minimum"
        if not math.isfinite(response_value) or response_value < min_response
        else "shift_out_of_bounds"
        if abs(dx) > max_translation_px or abs(dy) > max_translation_px
        else "overlap_below_minimum"
        if overlap < min_overlap_fraction
        else None
    )
    if phase_reason is not None:
        if allow_feature_fallback:
            feature_result = _sift_similarity_registration(reference_map, query_map)
            if feature_result is not None:
                return feature_result
        return RegistrationResult(
            reference_map.copy(),
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (float(dx), float(dy)),
            response_value,
            overlap,
            False,
            phase_reason,
            "none",
        )
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        reference_map,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return RegistrationResult(
        np.clip(aligned, 0.0, 1.0).astype(np.float32, copy=False),
        matrix,
        (float(dx), float(dy)),
        response_value,
        overlap,
        True,
        "accepted",
        "phase_translation",
    )


def aggregate_binary_recurrence_prior(
    query_thumbnail: np.ndarray,
    reference_thumbnails: Sequence[np.ndarray],
    reference_score_maps: Sequence[np.ndarray],
    *,
    score_threshold: float = 0.5,
    dilation_radius_compact_px: int = 1,
    required_fraction: float = 2.0 / 3.0,
) -> PriorAggregationResult:
    """Return a 2-of-3 aligned binary recurrence prior, failing open on any bad registration."""
    query = _probability_map(query_thumbnail, name="query_thumbnail")
    if len(reference_thumbnails) != len(reference_score_maps) or not reference_thumbnails:
        raise ValueError("reference thumbnails and score maps must be non-empty and equal length")
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 < required_fraction <= 1.0:
        raise ValueError("score_threshold and required_fraction must be probabilities")
    if dilation_radius_compact_px < 0:
        raise ValueError("dilation radius must be non-negative")

    height, width = query.shape
    registrations: list[RegistrationResult] = []
    aligned_binary: list[np.ndarray] = []
    aligned_continuous: list[np.ndarray] = []
    for thumbnail, score_map in zip(reference_thumbnails, reference_score_maps, strict=True):
        score = _probability_map(score_map, name="reference_score_map")
        if score.shape != query.shape:
            raise ValueError("reference score maps must match the query shape")
        registration = register_reference_thumbnail(
            thumbnail, query, allow_feature_fallback=True
        )
        registrations.append(registration)
        if not registration.accepted:
            continue
        aligned = cv2.warpAffine(
            score,
            registration.warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_continuous.append(aligned)
        binary = (aligned >= score_threshold).astype(np.uint8)
        if dilation_radius_compact_px:
            radius = int(dilation_radius_compact_px)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            binary = cv2.dilate(binary, kernel)
        aligned_binary.append(binary)

    accepted_count = len(aligned_binary)
    usable = accepted_count == len(reference_thumbnails)
    if not usable:
        prior = np.zeros(query.shape, dtype=np.float32)
        reference_score_ceiling = np.zeros(query.shape, dtype=np.float32)
    else:
        required = int(math.ceil(required_fraction * len(aligned_binary) - 1e-9))
        prior = (np.sum(np.stack(aligned_binary), axis=0) >= required).astype(np.float32)
        reference_score_ceiling = np.max(
            np.stack(aligned_continuous), axis=0
        ).astype(np.float32, copy=False)
    return PriorAggregationResult(
        prior=prior,
        reference_score_ceiling=reference_score_ceiling,
        usable=usable,
        accepted_count=accepted_count,
        total_count=len(reference_thumbnails),
        registrations=tuple(registrations),
    )


def reference_relative_local_peak_mask(
    query_score_map: np.ndarray,
    reference_score_ceiling: np.ndarray,
    *,
    radius_px: int,
) -> np.ndarray:
    """Protect query regional maxima that strictly exceed nearby normal references."""
    query = _probability_map(query_score_map, name="query_score_map")
    ceiling = _probability_map(
        reference_score_ceiling, name="reference_score_ceiling"
    )
    if query.shape != ceiling.shape:
        raise ValueError("query and reference ceiling maps must have the same shape")
    if (
        not isinstance(radius_px, (int, np.integer))
        or isinstance(radius_px, (bool, np.bool_))
        or int(radius_px) < 0
    ):
        raise ValueError("radius_px must be a non-negative integer")
    radius = int(radius_px)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    query_local_max = cv2.dilate(query, kernel)
    reference_local_ceiling = cv2.dilate(ceiling, kernel)
    return np.logical_and(query == query_local_max, query > reference_local_ceiling)


def apply_recurrence_suppression(
    raw_score_map: np.ndarray,
    recurrence_prior: np.ndarray,
    *,
    suppression_strength: float = 0.5,
    preserve_mask: np.ndarray | None = None,
) -> np.ndarray:
    raw = _probability_map(raw_score_map, name="raw_score_map")
    prior = _probability_map(recurrence_prior, name="recurrence_prior")
    if raw.shape != prior.shape:
        raise ValueError("raw and recurrence maps must have the same shape")
    if not 0.0 <= suppression_strength <= 0.5:
        raise ValueError("suppression_strength must be in [0, 0.5]")
    effective_prior = prior
    if preserve_mask is not None:
        preserve = np.asarray(preserve_mask)
        if preserve.shape != raw.shape or preserve.dtype != np.bool_:
            raise ValueError("preserve_mask must be a matching boolean array")
        effective_prior = prior * np.logical_not(preserve)
    return (raw * (1.0 - suppression_strength * effective_prior)).astype(
        np.float32, copy=False
    )


def build_reference_relative_peak_candidate(
    raw_score_map: np.ndarray,
    compact_recurrence_prior: np.ndarray,
    compact_reference_score_ceiling: np.ndarray,
    *,
    radius_compact_px: int = 1,
    suppression_strength: float = 0.5,
) -> PeakPreservedScores:
    """Build native B/P maps from a fixed compact normal-reference gate."""
    raw = _probability_map(raw_score_map, name="raw_score_map")
    compact_prior = _probability_map(
        compact_recurrence_prior, name="compact_recurrence_prior"
    )
    compact_ceiling = _probability_map(
        compact_reference_score_ceiling,
        name="compact_reference_score_ceiling",
    )
    if compact_prior.shape != compact_ceiling.shape:
        raise ValueError("compact prior and reference ceiling must have the same shape")

    compact_height, compact_width = compact_prior.shape
    compact_query = cv2.resize(
        raw,
        (compact_width, compact_height),
        interpolation=cv2.INTER_AREA,
    )
    compact_preserve = reference_relative_local_peak_mask(
        compact_query,
        compact_ceiling,
        radius_px=radius_compact_px,
    )
    native_size = (raw.shape[1], raw.shape[0])
    native_prior = cv2.resize(
        compact_prior,
        native_size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32, copy=False)
    native_preserve = cv2.resize(
        compact_preserve.astype(np.uint8),
        native_size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.bool_, copy=False)
    baseline = apply_recurrence_suppression(
        raw,
        native_prior,
        suppression_strength=suppression_strength,
    )
    candidate = apply_recurrence_suppression(
        raw,
        native_prior,
        suppression_strength=suppression_strength,
        preserve_mask=native_preserve,
    )
    return PeakPreservedScores(
        baseline=baseline,
        candidate=candidate,
        compact_preserve_mask=compact_preserve,
        native_preserve_mask=native_preserve,
        native_recurrence_prior=native_prior,
    )


def apply_case_support_boost(
    raw_score_map: np.ndarray,
    suppressed_score_map: np.ndarray,
    *,
    support: float,
    restore_fraction: float = 0.25,
) -> np.ndarray:
    raw = _probability_map(raw_score_map, name="raw_score_map")
    suppressed = _probability_map(suppressed_score_map, name="suppressed_score_map")
    if raw.shape != suppressed.shape or np.any(suppressed > raw + 1e-6):
        raise ValueError("suppressed map must match and not exceed the raw map")
    if not 0.0 <= support <= 1.0 or not 0.0 <= restore_fraction <= 1.0:
        raise ValueError("support and restore_fraction must be in [0, 1]")
    return (suppressed + restore_fraction * support * (raw - suppressed)).astype(
        np.float32, copy=False
    )


def render_original_overlay_comparison(
    rgb: np.ndarray,
    score_map: np.ndarray,
    *,
    threshold: float = 0.5,
    alpha: float = 0.55,
) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    score = _probability_map(score_map, name="score_map")
    if score.shape != rgb.shape[:2]:
        raise ValueError("score_map shape must match rgb")
    overlay = render_mask_overlay(rgb, score, threshold=threshold, alpha=alpha)
    return np.concatenate((rgb, overlay), axis=1)
