from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .capture import parse_standard_capture
from .inference import Box
from .refinement_contract import (
    AnchorRow,
    CandidateRow,
    PatchKey,
    SignalRow,
    SourceRow,
    canonical_digest,
    sha256_file,
)
from .refinement_memory import MemoryQuery
from .roi import estimate_placement_mask
from .synthetic import SyntheticDefectGenerator


@dataclass(frozen=True, slots=True)
class TTAStability:
    probability_stability: float
    feature_stability: float


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    patch_id: str
    probability_stability: float
    feature_stability: float
    neighbor_mean_feature_key: str
    context_cosine_contrast: float
    foreground_fraction: float
    roi_relation: str
    roi_fraction: float
    spatial_support: int
    capture_family: str = "unknown"
    view_index: int | None = None
    metadata_missing: bool = True


@dataclass(frozen=True, slots=True)
class SignalSchema:
    names: tuple[str, ...]
    feature_dimension: int
    digest: str

    @classmethod
    def default(cls, pooled_dimension: int) -> "SignalSchema":
        if pooled_dimension <= 0:
            raise ValueError("pooled_dimension must be positive")
        names = tuple(
            [f"local_{index:04d}" for index in range(pooled_dimension)]
            + [f"neighbor_mean_{index:04d}" for index in range(pooled_dimension)]
            + [f"local_minus_neighbor_{index:04d}" for index in range(pooled_dimension)]
            + [
                "epoch3_probability",
                "normal_memory_distance",
                "tta_probability_stability",
                "tta_feature_stability",
                "context_cosine_contrast",
                "spatial_support",
                "foreground_fraction",
                "roi_inside",
                "roi_edge",
                "roi_outside",
                "roi_unavailable",
                "capture_A",
                "capture_B",
                "capture_C",
                "capture_extra",
                "capture_unknown",
                "normalized_view_index",
                "metadata_missing",
            ]
        )
        digest = canonical_digest({"names": names, "feature_dimension": len(names)})
        return cls(names=names, feature_dimension=len(names), digest=digest)


@dataclass(frozen=True, slots=True)
class SignalBundle:
    row: SignalRow
    values: np.ndarray


def _validate_patch(rgb_patch: np.ndarray) -> None:
    if rgb_patch.shape != (256, 256, 3) or rgb_patch.dtype != np.uint8:
        raise ValueError("rgb_patch must be a 256 x 256 RGB uint8 array")


def deterministic_tta_views(rgb_patch: np.ndarray) -> tuple[np.ndarray, ...]:
    _validate_patch(rgb_patch)
    identity = rgb_patch.copy()
    darker = np.rint(rgb_patch.astype(np.float32) * 0.95).clip(0, 255).astype(np.uint8)
    brighter = np.rint(rgb_patch.astype(np.float32) * 1.05).clip(0, 255).astype(np.uint8)
    padded = cv2.copyMakeBorder(rgb_patch, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)
    translated_x = padded[2:258, 0:256].copy()
    translated_y = padded[0:256, 2:258].copy()
    return identity, darker, brighter, translated_x, translated_y


def tta_stability(probabilities: np.ndarray, features: np.ndarray) -> TTAStability:
    values = np.asarray(probabilities, dtype=np.float32)
    feature_rows = np.asarray(features, dtype=np.float32)
    if values.ndim != 1 or feature_rows.ndim != 2 or len(values) != len(feature_rows) or not len(values):
        raise ValueError("TTA probabilities and features must have matching non-empty rows")
    if not np.isfinite(values).all() or not np.isfinite(feature_rows).all():
        raise ValueError("TTA probabilities and features must be finite")
    probability_score = float(np.clip(1.0 - (float(values.max()) - float(values.min())), 0.0, 1.0))
    norms = np.linalg.norm(feature_rows, axis=1, keepdims=True)
    normalized = feature_rows / np.maximum(norms, 1e-12)
    cosine = normalized @ normalized[0]
    feature_score = float(np.clip(((cosine + 1.0) * 0.5).mean(), 0.0, 1.0))
    return TTAStability(probability_score, feature_score)


def context_boxes(
    center: Box,
    image_shape: tuple[int, int],
    patch_size: int = 256,
    stride: int = 128,
) -> tuple[Box, ...]:
    height, width = image_shape
    if height < patch_size or width < patch_size:
        raise ValueError("image is smaller than the native patch size")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    unique: dict[tuple[int, int], Box] = {}
    for delta_y in (-stride, 0, stride):
        for delta_x in (-stride, 0, stride):
            x0 = int(np.clip(center.x0 + delta_x, 0, width - patch_size))
            y0 = int(np.clip(center.y0 + delta_y, 0, height - patch_size))
            unique[(x0, y0)] = Box(x0, y0, x0 + patch_size, y0 + patch_size)
    return tuple(unique[key] for key in sorted(unique, key=lambda item: (item[1], item[0])))


def roi_relation(box: Box, roi_mask: np.ndarray | None) -> tuple[str, float]:
    if roi_mask is None:
        return "unavailable", 0.0
    if roi_mask.ndim != 2 or roi_mask.dtype != np.bool_:
        raise ValueError("roi_mask must be a two-dimensional boolean array")
    if box.x0 < 0 or box.y0 < 0 or box.x1 > roi_mask.shape[1] or box.y1 > roi_mask.shape[0]:
        raise ValueError("box lies outside the ROI mask")
    fraction = float(roi_mask[box.y0 : box.y1, box.x0 : box.x1].mean())
    if fraction >= 0.95:
        relation = "inside"
    elif fraction <= 0.05:
        relation = "outside"
    else:
        relation = "edge"
    return relation, fraction


def _encode(encoder: Any, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(encoder, "encode"):
        raise TypeError("candidate evidence encoder must expose encode(rgb_batch)")
    logits, features = encoder.encode(rgb_batch)
    logits = np.asarray(logits, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if logits.shape != (len(rgb_batch),) or features.ndim != 2 or features.shape[0] != len(rgb_batch):
        raise ValueError("encoder output rows do not match input patches")
    if not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise ValueError("encoder outputs must be finite")
    probabilities = (1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))).astype(np.float32)
    return probabilities, features


def _feature_key(feature: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(feature, dtype=np.float32).tobytes()).hexdigest()


def _spatial_support(rows: Sequence[CandidateRow], center: CandidateRow, stride: int = 128) -> int:
    return sum(
        row.source_sha256 == center.source_sha256
        and abs(row.box[0] - center.box[0]) <= stride
        and abs(row.box[1] - center.box[1]) <= stride
        for row in rows
    )


def augment_candidate_evidence(
    rows: Sequence[CandidateRow],
    image_loader: Callable[[str], np.ndarray],
    encoder: Any,
    roi_provider: Callable[[str, np.ndarray], np.ndarray | None],
    *,
    batch_size: int = 128,
) -> tuple[list[CandidateEvidence], np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    evidence_rows: list[CandidateEvidence] = []
    feature_blocks: list[np.ndarray] = []
    by_source: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source_sha256].append(row)

    for source_sha256 in sorted(by_source):
        source_rows = by_source[source_sha256]
        source_path = source_rows[0].source_path
        if sha256_file(source_path) != source_sha256:
            raise RuntimeError(f"source digest drift for {source_path}")
        rgb = np.asarray(image_loader(source_path))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("image_loader must return an HxWx3 uint8 array")
        roi_mask = roi_provider(source_path, rgb)
        source_plans: list[tuple[CandidateRow, Box, int, int]] = []
        source_patches: list[np.ndarray] = []
        for row in sorted(source_rows, key=lambda item: (item.box[1], item.box[0], item.patch_id)):
            box = Box(*row.box)
            patch = rgb[box.y0 : box.y1, box.x0 : box.x1]
            _validate_patch(patch)
            neighborhood = context_boxes(box, rgb.shape[:2])
            neighbor_boxes = [candidate for candidate in neighborhood if candidate != box]
            patch_block = [*deterministic_tta_views(patch)]
            patch_block.extend(
                rgb[item.y0 : item.y1, item.x0 : item.x1] for item in neighbor_boxes
            )
            source_plans.append((row, box, len(source_patches), len(neighbor_boxes)))
            source_patches.extend(patch_block)

        probabilities: list[np.ndarray] = []
        features: list[np.ndarray] = []
        for start in range(0, len(source_patches), batch_size):
            batch = np.stack(source_patches[start : start + batch_size])
            batch_probabilities, batch_features = _encode(encoder, batch)
            probabilities.append(batch_probabilities)
            features.append(batch_features)
        source_probabilities = np.concatenate(probabilities, axis=0)
        source_features = np.concatenate(features, axis=0)

        for row, box, start, neighbor_count in source_plans:
            tta_probabilities = source_probabilities[start : start + 5]
            tta_features = source_features[start : start + 5]
            stability = tta_stability(tta_probabilities, tta_features)
            if neighbor_count:
                neighbor_features = source_features[
                    start + 5 : start + 5 + neighbor_count
                ]
                neighbor_mean = neighbor_features.mean(axis=0, dtype=np.float32)
            else:
                neighbor_mean = tta_features[0].copy()
            local_feature = tta_features[0]
            local_norm = max(float(np.linalg.norm(local_feature)), 1e-12)
            neighbor_norm = max(float(np.linalg.norm(neighbor_mean)), 1e-12)
            context_contrast = float(
                1.0 - np.clip(np.dot(local_feature, neighbor_mean) / (local_norm * neighbor_norm), -1.0, 1.0)
            )
            relation, roi_fraction = roi_relation(box, roi_mask)
            condition = parse_standard_capture(Path(source_path).name)
            evidence_rows.append(
                CandidateEvidence(
                    patch_id=row.patch_id,
                    probability_stability=stability.probability_stability,
                    feature_stability=stability.feature_stability,
                    neighbor_mean_feature_key=_feature_key(neighbor_mean),
                    context_cosine_contrast=context_contrast,
                    foreground_fraction=row.foreground_fraction,
                    roi_relation=relation,
                    roi_fraction=roi_fraction,
                    spatial_support=_spatial_support(rows, row),
                    capture_family=condition.family if condition else "unknown",
                    view_index=condition.view_index if condition else None,
                    metadata_missing=condition is None,
                )
            )
            feature_blocks.append(
                np.stack((local_feature, tta_features.mean(axis=0), neighbor_mean)).astype(np.float32)
            )

    if not feature_blocks:
        return [], np.empty((0, 3, 0), dtype=np.float32)
    dimensions = {block.shape for block in feature_blocks}
    if len(dimensions) != 1:
        raise ValueError("candidate evidence feature dimensions do not match")
    output = np.stack(feature_blocks)
    if not np.isfinite(output).all():
        raise ValueError("candidate evidence features must be finite")
    return evidence_rows, output


def _sample_supported_patch(
    rgb: np.ndarray,
    support: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Box]:
    height, width = rgb.shape[:2]
    if height < 256 or width < 256:
        raise ValueError("synthetic source is smaller than 256 x 256")
    active = np.argwhere(support)
    if not len(active):
        raise ValueError("OK source has no ceramic support for synthesis")
    for _ in range(64):
        y, x = active[int(rng.integers(0, len(active)))]
        x0 = int(np.clip(int(x) - int(rng.integers(64, 193)), 0, width - 256))
        y0 = int(np.clip(int(y) - int(rng.integers(64, 193)), 0, height - 256))
        box = Box(x0, y0, x0 + 256, y0 + 256)
        patch_support = support[y0 : y0 + 256, x0 : x0 + 256]
        if float(patch_support.mean()) >= 0.25:
            return rgb[y0 : y0 + 256, x0 : x0 + 256].copy(), patch_support.copy(), box
    raise RuntimeError("could not sample a supported synthetic center patch")


def build_synthetic_anchor_rows(
    ok_sources: Sequence[SourceRow],
    encoder: Any,
    generator: SyntheticDefectGenerator,
    seed: int,
    count: int,
    *,
    model_digest: str = "",
    config_digest: str = "",
) -> tuple[list[AnchorRow], np.ndarray]:
    if count <= 0:
        raise ValueError("synthetic anchor count must be positive")
    if not ok_sources or any(source.source_label != "OK" for source in ok_sources):
        raise ValueError("synthetic anchors require explicit OK sources")
    sources = sorted(ok_sources, key=lambda row: (row.product, row.group_id, row.path))
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    combinations = [
        (family, severity)
        for severity in generator.severities
        for family in generator.families
    ]
    anchor_rows: list[AnchorRow] = []
    feature_rows: list[np.ndarray] = []

    for index in range(count):
        source = sources[index % len(sources)]
        if source.path not in source_cache:
            if sha256_file(source.path) != source.source_sha256:
                raise RuntimeError(f"source digest drift for {source.path}")
            with Image.open(source.path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            support = estimate_placement_mask(rgb, margin=8)
            if not support.any():
                support = rgb.mean(axis=2) > 8
            source_cache[source.path] = rgb, support.astype(bool)
        rgb, support = source_cache[source.path]
        family, severity = combinations[index % len(combinations)]
        sample_seed = int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(sample_seed)
        original_patch, patch_support, box = _sample_supported_patch(rgb, support, rng)
        result = generator.apply(
            original_patch,
            patch_support,
            family=family,
            severity=severity,
            rng=rng,
        )
        tta_probabilities, tta_features = _encode(
            encoder, np.stack(deterministic_tta_views(result.image))
        )
        stability = tta_stability(tta_probabilities, tta_features)
        local_feature = tta_features[0].astype(np.float32, copy=True)
        neighbor_boxes = [
            candidate
            for candidate in context_boxes(box, rgb.shape[:2])
            if candidate != box
        ]
        if neighbor_boxes:
            original_neighbors = np.stack(
                [rgb[item.y0 : item.y1, item.x0 : item.x1] for item in neighbor_boxes]
            )
            _, original_neighbor_features = _encode(encoder, original_neighbors)
            neighbor_mean = original_neighbor_features.mean(axis=0, dtype=np.float32)
        else:
            _, original_features = _encode(encoder, original_patch[None, ...])
            neighbor_mean = original_features[0]
        local_norm = max(float(np.linalg.norm(local_feature)), 1e-12)
        neighbor_norm = max(float(np.linalg.norm(neighbor_mean)), 1e-12)
        context_contrast = float(
            1.0
            - np.clip(
                np.dot(local_feature, neighbor_mean) / (local_norm * neighbor_norm),
                -1.0,
                1.0,
            )
        )

        provenance = {
            "config_digest": config_digest,
            "context_cosine_contrast": context_contrast,
            "generator_parameters": result.parameters,
            "model_digest": model_digest,
            "neighbor_mean_feature_key": _feature_key(neighbor_mean),
            "sample_seed": sample_seed,
            "seed": seed,
            "source_label": "OK",
            "source_sha256": source.source_sha256,
        }
        provenance_json = json.dumps(
            provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        synthetic_digest = hashlib.sha256(
            f"{source.source_sha256}:{provenance_json}".encode("utf-8")
        ).hexdigest()
        box_values = (box.x0, box.y0, box.x1, box.y1)
        anchor_rows.append(
            AnchorRow(
                patch_id=PatchKey(synthetic_digest, box_values).patch_id,
                source_sha256=source.source_sha256,
                product=source.product,
                group_id=source.group_id,
                condition_id=source.condition_id,
                box=box_values,
                probability=float(tta_probabilities[0]),
                tta_stability=min(
                    stability.probability_stability, stability.feature_stability
                ),
                foreground_fraction=float(patch_support.mean()),
                feature_key=_feature_key(local_feature),
                source_label="SYNTHETIC",
                source_kind="synthetic_anchor",
                synthetic_family=family,
                synthetic_severity=severity,
                support_fraction=float(result.mask.mean()),
                seed=seed,
                provenance_json=provenance_json,
                model_digest=model_digest,
                config_digest=config_digest,
            )
        )
        feature_rows.append(local_feature)

    features = np.stack(feature_rows).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("synthetic anchor features must be finite")
    return anchor_rows, features


def build_signal_bundle(
    candidate: CandidateRow,
    local_feature: np.ndarray,
    neighbor_features: Sequence[np.ndarray],
    memory_query: MemoryQuery,
    evidence: CandidateEvidence,
) -> SignalBundle:
    local = np.asarray(local_feature, dtype=np.float32)
    if local.ndim != 1 or not local.size or not np.isfinite(local).all():
        raise ValueError("local feature must be a finite one-dimensional array")
    neighbors = [np.asarray(feature, dtype=np.float32) for feature in neighbor_features]
    if any(feature.shape != local.shape or not np.isfinite(feature).all() for feature in neighbors):
        raise ValueError("neighbor feature dimensions must match the local feature")
    neighbor_mean = (
        np.stack(neighbors).mean(axis=0, dtype=np.float32) if neighbors else local.copy()
    )
    if evidence.patch_id != candidate.patch_id:
        raise ValueError("candidate evidence patch ID does not match")
    if not np.isfinite(memory_query.distance):
        raise ValueError("normal-memory distance must be finite")
    roi_categories = ("inside", "edge", "outside", "unavailable")
    if evidence.roi_relation not in roi_categories:
        raise ValueError(f"unsupported ROI relation: {evidence.roi_relation}")
    family_categories = ("A", "B", "C", "extra", "unknown")
    family = evidence.capture_family if evidence.capture_family in family_categories else "unknown"
    roi_one_hot = [float(evidence.roi_relation == category) for category in roi_categories]
    family_one_hot = [float(family == category) for category in family_categories]
    normalized_view = (
        0.0
        if evidence.view_index is None
        else float(np.clip(evidence.view_index, 0, 99) / 99.0)
    )
    scalar_values = np.asarray(
        [
            candidate.probability,
            memory_query.distance,
            evidence.probability_stability,
            evidence.feature_stability,
            evidence.context_cosine_contrast,
            float(evidence.spatial_support),
            evidence.foreground_fraction,
            *roi_one_hot,
            *family_one_hot,
            normalized_view,
            float(evidence.metadata_missing),
        ],
        dtype=np.float32,
    )
    values = np.concatenate((local, neighbor_mean, local - neighbor_mean, scalar_values)).astype(
        np.float32
    )
    if not np.isfinite(values).all():
        raise ValueError("signal vector must be finite")
    schema = SignalSchema.default(local.size)
    if values.size != schema.feature_dimension:
        raise RuntimeError("signal vector does not match the declared schema")
    signal_key = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    row = SignalRow(
        patch_id=candidate.patch_id,
        source_sha256=candidate.source_sha256,
        product=candidate.product,
        group_id=candidate.group_id,
        condition_id=candidate.condition_id,
        box=candidate.box,
        probability=candidate.probability,
        roi_relation=evidence.roi_relation,
        spatial_support=evidence.spatial_support,
        signal_key=signal_key,
        feature_dimension=schema.feature_dimension,
        schema_digest=schema.digest,
        source_label=candidate.source_label,
        source_kind=(
            "synthetic_anchor"
            if candidate.source_label == "SYNTHETIC"
            else "unlabeled_candidate"
        ),
    )
    return SignalBundle(row=row, values=values)
