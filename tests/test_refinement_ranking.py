from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from patch_qc.inference import Box
from patch_qc.refinement_contract import CandidateRow, SignalRow
from patch_qc.refinement_ranking import (
    EmpiricalTailCalibrator,
    RankedPatch,
    deterministic_relevance,
    fit_weak_ranker,
    merge_and_rank_candidates,
    rank_with_fallback,
    select_refiner,
)
from patch_qc.refinement_signals import SignalBundle, SignalSchema


def _signal(
    index: int,
    *,
    probability: float,
    memory_distance: float,
    context_contrast: float = 0.1,
    tta_feature_stability: float = 0.95,
    spatial_support: int = 1,
    source_label: str = "OK",
    source_kind: str = "clean_anchor",
) -> SignalBundle:
    schema = SignalSchema.default(2)
    values = np.zeros(schema.feature_dimension, dtype=np.float32)
    values[:2] = (-1.0, -0.5) if source_label == "OK" else (1.0, 0.5)
    values[-18] = probability
    values[-17] = memory_distance
    values[-16] = 0.98
    values[-15] = tta_feature_stability
    values[-14] = context_contrast
    values[-13] = spatial_support
    values[-12] = 0.9
    values[-11] = 1.0
    row = SignalRow(
        patch_id=f"patch-{index}",
        source_sha256=f"{index + 1:064x}",
        product="P1",
        group_id=f"G{index % 3}",
        condition_id="C1",
        box=(index * 512, 0, index * 512 + 256, 256),
        probability=probability,
        roi_relation="inside",
        spatial_support=spatial_support,
        signal_key=f"signal-{index}",
        feature_dimension=schema.feature_dimension,
        schema_digest=schema.digest,
        source_label=source_label,
        source_kind=source_kind,
    )
    return SignalBundle(row, values)


def _tail_calibrator() -> EmpiricalTailCalibrator:
    clean = [
        _signal(index, probability=0.05 + index * 0.03, memory_distance=0.05 + index * 0.02)
        for index in range(6)
    ]
    return EmpiricalTailCalibrator.fit(clean)


def test_fusion_rewards_anomaly_evidence_without_requiring_clusters() -> None:
    calibrator = _tail_calibrator()
    singleton = _signal(20, probability=0.9, memory_distance=0.9, spatial_support=1)
    ordinary = _signal(21, probability=0.2, memory_distance=0.1, spatial_support=4)

    assert deterministic_relevance(singleton, calibrator) > deterministic_relevance(ordinary, calibrator)


def test_fusion_outside_roi_deducts_score_without_deleting_candidate() -> None:
    calibrator = _tail_calibrator()
    inside = _signal(30, probability=0.8, memory_distance=0.8)
    outside = SignalBundle(replace(inside.row, roi_relation="outside"), inside.values.copy())

    assert np.isclose(
        deterministic_relevance(inside, calibrator) - deterministic_relevance(outside, calibrator),
        0.05,
    )


def _clean_rows() -> list[SignalBundle]:
    return [
        _signal(index, probability=0.01 + index * 0.005, memory_distance=0.02 + index * 0.005)
        for index in range(6)
    ]


def _synthetic_rows() -> list[SignalBundle]:
    return [
        _signal(
            100 + index,
            probability=0.7 + index * 0.02,
            memory_distance=0.6 + index * 0.02,
            context_contrast=0.5,
            source_label="SYNTHETIC",
            source_kind="synthetic_anchor",
        )
        for index in range(6)
    ]


def test_weak_ranker_repeats_same_seed_and_never_trains_on_unlabeled_ok() -> None:
    calibration = [_clean_rows()[0], _synthetic_rows()[0]]

    first = fit_weak_ranker(_clean_rows(), _synthetic_rows(), calibration, seed=17)
    second = fit_weak_ranker(_clean_rows(), _synthetic_rows(), calibration, seed=17)

    assert first.state_digest == second.state_digest
    unlabeled = SignalBundle(
        replace(_clean_rows()[0].row, source_kind="unlabeled_candidate"),
        _clean_rows()[0].values,
    )
    with pytest.raises(ValueError, match="unlabeled"):
        fit_weak_ranker(_clean_rows() + [unlabeled], _synthetic_rows(), calibration, seed=17)


def test_refiner_gate_requires_all_three_weak_ranker_improvements() -> None:
    fusion = {
        "synthetic_local_recall_at_5": 0.90,
        "clean_mean_candidates_per_image": 2.0,
        "tta_top5_agreement": 0.95,
    }
    weak = {
        "synthetic_local_recall_at_5": 0.92,
        "clean_mean_candidates_per_image": 2.04,
        "tta_top5_agreement": 0.95,
    }

    assert select_refiner(fusion, weak) == "weak_ranker"
    assert select_refiner(fusion, {**weak, "tta_top5_agreement": 0.94}) == "deterministic_fusion"


def _ranked_patch(
    patch_id: str,
    box: Box,
    score: float,
    feature: tuple[float, float],
) -> RankedPatch:
    return RankedPatch(
        patch_id=patch_id,
        box=box,
        relevance=score,
        probability=score,
        feature=np.asarray(feature, dtype=np.float32),
        roi_relation="inside",
    )


def test_top_k_merges_overlap_preserves_singleton_and_diversifies() -> None:
    rows = [
        _ranked_patch("a", Box(0, 0, 256, 256), 0.9, (1.0, 0.0)),
        _ranked_patch("b", Box(128, 0, 384, 256), 0.8, (1.0, 0.0)),
        _ranked_patch("single", Box(512, 0, 768, 256), 0.75, (0.0, 1.0)),
        _ranked_patch("c", Box(1024, 0, 1280, 256), 0.7, (0.7, 0.7)),
        _ranked_patch("d", Box(1536, 0, 1792, 256), 0.6, (-1.0, 0.0)),
    ]

    ranked = merge_and_rank_candidates(rows, top_k=3, stride=128, diversity_cosine=0.98)

    assert len(ranked) == 3
    assert any(candidate.support == 1 for candidate in ranked)
    assert any(set(candidate.patch_ids) == {"a", "b"} for candidate in ranked)
    assert len({candidate.component_id for candidate in ranked}) == 3
    assert all(candidate.review_state == "REVIEW" for candidate in ranked)


def _candidate_rows() -> list[CandidateRow]:
    probabilities = [0.4, 0.9, 0.7]
    return [
        CandidateRow(
            patch_id=f"candidate-{index}",
            source_sha256="f" * 64,
            source_path="/unused/source.png",
            product="P1",
            group_id="G1",
            condition_id="C1",
            split="holdout",
            box=(index * 512, 0, index * 512 + 256, 256),
            logit=float(probability),
            probability=probability,
            foreground_fraction=0.8,
            feature_key=f"feature-{index}",
        )
        for index, probability in enumerate(probabilities)
    ]


def test_ranking_falls_back_to_epoch3_probability_when_signals_are_unavailable() -> None:
    receipt = rank_with_fallback(_candidate_rows(), signal_bundles=None, refiner=None, top_k=3)

    assert receipt.fallback_level == "epoch3_probability"
    assert [item.patch_ids for item in receipt.candidates] == [
        ("candidate-1",),
        ("candidate-2",),
        ("candidate-0",),
    ]
    assert all(item.review_state == "REVIEW" for item in receipt.candidates)


def test_ranking_does_not_hide_nonfinite_signal_values_behind_fallback() -> None:
    candidates = _candidate_rows()
    bundles = [
        SignalBundle(
            replace(
                _signal(index, probability=row.probability, memory_distance=0.1).row,
                patch_id=row.patch_id,
            ),
            _signal(index, probability=row.probability, memory_distance=0.1).values,
        )
        for index, row in enumerate(candidates)
    ]
    bundles[0].values[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        rank_with_fallback(candidates, bundles, _tail_calibrator(), top_k=3)
