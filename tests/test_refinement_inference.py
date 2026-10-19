from __future__ import annotations

import numpy as np
from PIL import Image

from patch_qc.refinement_contract import AnchorRow, sha256_file
from patch_qc.refinement_encoding import CandidatePolicy
from patch_qc.refinement_inference import (
    InferenceCase,
    rank_image_case,
    review_candidates_to_json,
    select_native_candidates,
)
from patch_qc.refinement_memory import NormalMemoryIndex
from patch_qc.refinement_ranking import EmpiricalTailCalibrator, ReviewCandidate
from patch_qc.inference import Box


class _SequenceEncoder:
    def __init__(self) -> None:
        self.offset = 0

    def encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = np.asarray((0.1, 0.2, 0.8, 0.9), dtype=np.float32)[
            self.offset : self.offset + len(rgb_batch)
        ]
        self.offset += len(rgb_batch)
        logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
        features = np.repeat(probabilities[:, None], 4, axis=1)
        return logits, features


class _MeanEncoder:
    def encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = rgb_batch.mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32) / 255.0
        logits = (means - 0.5) * 8.0
        features = np.stack((means, means**2, 1.0 - means, np.ones_like(means)), axis=1)
        return logits.astype(np.float32), features.astype(np.float32)


def test_native_candidate_selection_preserves_stride_tail_and_evaluation_boundary() -> None:
    rgb = np.full((384, 384, 3), 128, dtype=np.uint8)
    case = InferenceCase(
        case_id="case-1",
        image_path="/tmp/case-1.png",
        source_sha256="a" * 64,
        product="P1",
        group_id="G1",
        condition_id="unknown",
    )

    rows, total_patches = select_native_candidates(
        rgb,
        case,
        _SequenceEncoder(),
        CandidatePolicy(probability_floor=0.8, minimum_top_m=1, maximum_candidates=2),
        batch_size=2,
        patch_size=256,
        stride=128,
    )

    assert total_patches == 4
    assert [row.box for row in rows] == [(128, 128, 384, 384), (0, 128, 256, 384)]
    assert all(row.source_label == "EVALUATION" for row in rows)
    assert all(row.source_sha256 == "a" * 64 for row in rows)


def test_review_candidate_json_is_native_and_review_only() -> None:
    rows = [
        ReviewCandidate(
            component_id="component-1",
            patch_ids=("patch-1",),
            box=Box(128, 256, 384, 512),
            score=0.9,
            probability=0.8,
            support=1,
            roi_relation="inside",
        )
    ]

    payload = review_candidates_to_json(rows)

    assert payload[0]["box"] == {"x0": 128, "y0": 256, "x1": 384, "y1": 512}
    assert payload[0]["review_state"] == "REVIEW"


def test_rank_image_case_produces_refined_and_epoch3_topk_without_labels(tmp_path) -> None:
    image_path = tmp_path / "case.png"
    rgb = np.full((384, 384, 3), 128, dtype=np.uint8)
    rgb[:, 192:] = 196
    Image.fromarray(rgb).save(image_path)
    digest = sha256_file(image_path)
    case = InferenceCase(
        case_id="case-1",
        image_path=str(image_path),
        source_sha256=digest,
        product="P1",
        group_id="sealed-case-1",
        condition_id="unknown",
    )
    anchors = [
        AnchorRow(
            patch_id=f"normal-{index}",
            source_sha256=f"{index + 1:064x}",
            product="P1",
            group_id=f"G{index}",
            condition_id="unknown",
            box=(0, 0, 256, 256),
            probability=0.01,
            tta_stability=1.0,
            foreground_fraction=1.0,
            feature_key=f"normal-{index}",
            source_label="OK",
            source_kind="clean_anchor",
            roi_relation="inside",
        )
        for index in range(5)
    ]
    normal_features = np.stack(
        [np.asarray((0.5, 0.25, 0.5, 1.0), dtype=np.float32) for _ in anchors]
    )
    memory = NormalMemoryIndex(normal_features, anchors)
    calibrator = EmpiricalTailCalibrator(
        distributions={
            name: {"global": (0.0, 0.1, 0.2, 0.3, 0.4)}
            for name in ("probability", "memory_distance", "context_contrast")
        }
    )

    result = rank_image_case(
        rgb,
        case,
        _MeanEncoder(),
        CandidatePolicy(0.7, 2, 4),
        memory,
        calibrator,
        batch_size=16,
        query_k=5,
        top_k=3,
        stride=128,
    )

    assert result.total_patches == 4
    assert result.candidate_count >= 2
    assert result.refined
    assert result.baseline
    assert all(row.review_state == "REVIEW" for row in (*result.refined, *result.baseline))
