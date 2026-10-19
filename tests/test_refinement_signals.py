from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from patch_qc.inference import Box
from patch_qc.refinement_contract import CandidateRow, PatchKey, SourceRow, sha256_file
from patch_qc.refinement_signals import (
    CandidateEvidence,
    SignalSchema,
    augment_candidate_evidence,
    build_signal_bundle,
    build_synthetic_anchor_rows,
    context_boxes,
    deterministic_tta_views,
    roi_relation,
    tta_stability,
)
from patch_qc.refinement_memory import MemoryQuery
from patch_qc.synthetic import SyntheticDefectGenerator


class _MeanEncoder:
    def encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = rgb_batch.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32) / 255.0
        features = np.stack((means, means**2, 1.0 - means, np.ones_like(means)), axis=1)
        return means, features.astype(np.float32)


class _CountingMeanEncoder(_MeanEncoder):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.batch_sizes.append(len(rgb_batch))
        return super().encode(rgb_batch)


def _write_source(path: Path) -> Path:
    y, x = np.mgrid[:384, :384]
    rgb = np.stack(((x % 256), (y % 256), ((x + y) % 256)), axis=2).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return path


def _candidate(path: Path) -> CandidateRow:
    digest = sha256_file(path)
    box = (0, 0, 256, 256)
    return CandidateRow(
        patch_id=PatchKey(digest, box).patch_id,
        source_sha256=digest,
        source_path=str(path),
        product="P1",
        group_id="G1",
        condition_id="C_L1-00",
        split="train",
        box=box,
        logit=0.5,
        probability=0.62,
        foreground_fraction=0.8,
        feature_key="candidate:0",
    )


def test_tta_views_are_repeatable_and_native_sized() -> None:
    patch = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)

    first = deterministic_tta_views(patch)
    second = deterministic_tta_views(patch)

    assert len(first) == 5
    assert all(view.shape == patch.shape for view in first)
    assert all(view.dtype == np.uint8 for view in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert np.array_equal(first[0], patch)


def test_context_boxes_clip_to_image_without_duplicates() -> None:
    boxes = context_boxes(Box(0, 0, 256, 256), (5464, 8192), patch_size=256, stride=128)

    assert Box(0, 0, 256, 256) in boxes
    assert len(boxes) == len(set(boxes))
    assert len(boxes) == 4
    assert all(box.x0 >= 0 and box.y0 >= 0 for box in boxes)


def test_roi_relation_is_soft_evidence_not_a_filter(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.png")
    mask = np.zeros((512, 512), dtype=bool)
    mask[128:384, 128:384] = True

    relation, fraction = roi_relation(Box(0, 0, 256, 256), mask)

    assert relation == "edge"
    assert np.isclose(fraction, 0.25)
    assert _candidate(source).patch_id
    assert roi_relation(Box(0, 0, 256, 256), None) == ("unavailable", 0.0)


def test_tta_stability_is_finite_and_rejects_nan() -> None:
    stability = tta_stability(
        np.array([0.5, 0.51, 0.49], dtype=np.float32),
        np.array([[1.0, 0.0], [0.99, 0.01], [1.01, -0.01]], dtype=np.float32),
    )

    assert 0.0 <= stability.probability_stability <= 1.0
    assert 0.0 <= stability.feature_stability <= 1.0
    with pytest.raises(ValueError, match="finite"):
        tta_stability(np.array([np.nan]), np.ones((1, 2), dtype=np.float32))


def test_candidate_augmentation_reopens_only_selected_native_regions(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.png")
    candidate = _candidate(source)
    full_mask = np.ones((384, 384), dtype=bool)

    evidence, feature_blocks = augment_candidate_evidence(
        [candidate],
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
        encoder=_MeanEncoder(),
        roi_provider=lambda _path, _rgb: full_mask,
    )

    assert len(evidence) == 1
    assert evidence[0].patch_id == candidate.patch_id
    assert evidence[0].roi_relation == "inside"
    assert evidence[0].spatial_support == 1
    assert feature_blocks.shape == (1, 3, 4)
    assert np.isfinite(feature_blocks).all()


def test_candidate_augmentation_batches_tta_and_context_per_source(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.png")
    digest = sha256_file(source)
    candidates = []
    for index, box in enumerate(((0, 0, 256, 256), (128, 128, 384, 384))):
        candidates.append(
            CandidateRow(
                patch_id=PatchKey(digest, box).patch_id,
                source_sha256=digest,
                source_path=str(source),
                product="P1",
                group_id="G1",
                condition_id="C_L1-00",
                split="train",
                box=box,
                logit=0.1,
                probability=0.52,
                foreground_fraction=0.8,
                feature_key=f"candidate:{index}",
            )
        )
    encoder = _CountingMeanEncoder()

    evidence, feature_blocks = augment_candidate_evidence(
        candidates,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
        encoder=encoder,
        roi_provider=lambda _path, _rgb: np.ones((384, 384), dtype=bool),
        batch_size=64,
    )

    assert len(evidence) == 2
    assert feature_blocks.shape == (2, 3, 4)
    assert len(encoder.batch_sizes) == 1
    assert encoder.batch_sizes[0] <= 64


def test_synthetic_anchors_use_ok_pixels_and_balanced_known_support(tmp_path: Path) -> None:
    source_path = tmp_path / "ok_source.png"
    Image.fromarray(np.full((384, 384, 3), 128, dtype=np.uint8)).save(source_path)
    source = SourceRow(
        source_sha256=sha256_file(source_path),
        path=str(source_path),
        product="P1",
        group_id="G1",
        condition_id="C_L1-00",
        split="train",
        width=384,
        height=384,
    )

    rows, features = build_synthetic_anchor_rows(
        [source],
        encoder=_MeanEncoder(),
        generator=SyntheticDefectGenerator(),
        seed=17,
        count=18,
        model_digest="1" * 64,
        config_digest="2" * 64,
    )

    assert {row.synthetic_family for row in rows} == set(SyntheticDefectGenerator.families)
    assert {row.synthetic_severity for row in rows} == set(SyntheticDefectGenerator.severities)
    assert all(row.source_label == "SYNTHETIC" for row in rows)
    assert all(row.source_kind == "synthetic_anchor" for row in rows)
    assert all(row.support_fraction > 0 for row in rows)
    assert all(row.seed == 17 for row in rows)
    assert all(row.model_digest == "1" * 64 and row.config_digest == "2" * 64 for row in rows)
    assert all('"source_label":"NG"' not in row.provenance_json for row in rows)
    assert all("labelme" not in row.provenance_json.casefold() for row in rows)
    assert set(Counter(row.synthetic_family for row in rows).values()) == {3}
    assert set(Counter(row.synthetic_severity for row in rows).values()) == {6}
    provenance = json.loads(rows[0].provenance_json)
    assert provenance["context_cosine_contrast"] >= 0.0
    assert len(provenance["neighbor_mean_feature_key"]) == 64
    assert features.shape == (len(rows), 4)
    assert np.isfinite(features).all()


def test_synthetic_anchor_builder_rejects_non_ok_sources(tmp_path: Path) -> None:
    source_path = tmp_path / "forbidden.png"
    Image.fromarray(np.full((384, 384, 3), 128, dtype=np.uint8)).save(source_path)
    source = SourceRow(
        source_sha256=sha256_file(source_path),
        path=str(source_path),
        product="P1",
        group_id="G1",
        condition_id="C",
        split="train",
        width=384,
        height=384,
        source_label="NG",
    )

    with pytest.raises(ValueError, match="OK sources"):
        build_synthetic_anchor_rows(
            [source], _MeanEncoder(), SyntheticDefectGenerator(), seed=17, count=18
        )


def test_signal_bundle_has_fixed_finite_schema(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source.png")
    candidate = _candidate(source)
    local = np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32)
    neighbor = np.array([0.8, 0.4, -0.1, 0.6], dtype=np.float32)
    memory = MemoryQuery(0.2, ("normal-1",), ("G2",), "product_condition")
    evidence = CandidateEvidence(
        patch_id=candidate.patch_id,
        probability_stability=0.98,
        feature_stability=0.97,
        neighbor_mean_feature_key="f" * 64,
        context_cosine_contrast=0.1,
        foreground_fraction=0.8,
        roi_relation="inside",
        roi_fraction=0.8,
        spatial_support=1,
        capture_family="C",
        view_index=3,
        metadata_missing=False,
    )

    bundle = build_signal_bundle(candidate, local, [neighbor], memory, evidence)

    assert bundle.row.patch_id == candidate.patch_id
    assert bundle.values.shape == (bundle.row.feature_dimension,)
    assert np.isfinite(bundle.values).all()
    assert bundle.row.schema_digest == SignalSchema.default(local.size).digest
    assert bundle.values.size == local.size * 3 + 18
