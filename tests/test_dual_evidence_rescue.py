from __future__ import annotations

import numpy as np

from patch_qc.dual_evidence_rescue import (
    fuse_primary_weighted_consensus,
    summarize_dual_evidence_results,
    validate_dual_evidence_spec,
)


def test_fusion_uses_primary_weighted_geometric_consensus() -> None:
    p_map = np.asarray([[0.9, 0.9], [0.4, 0.8]], dtype=np.float32)
    auxiliary_map = np.asarray([[0.4, 0.9], [0.9, 0.1]], dtype=np.float32)
    expected = np.cbrt(np.square(p_map) * auxiliary_map)

    actual = fuse_primary_weighted_consensus(
        p_map=p_map,
        auxiliary_map=auxiliary_map,
    )

    np.testing.assert_allclose(actual, expected)


def test_summary_requires_all_subgroups_and_all_localized_defects() -> None:
    rows: list[dict[str, object]] = []
    for number in range(1, 28):
        identifier = f"ok_{number:04d}"
        rows.append(
            {
                "id": identifier,
                "cohort": "ok",
                "label_status": "ok_unlabeled",
                "metadata": {},
                "raw_metrics": {"predicted_positive_pixels": 100},
                "candidate_metrics": {"predicted_positive_pixels": 70},
            }
        )
    for number in range(1, 81):
        rows.append(
            {
                "id": f"ng_{number:04d}",
                "cohort": "ng",
                "label_status": "localized",
                "metadata": {
                    "defect_type": "접합면 White Spot" if number <= 12 else "Scratch"
                },
                "raw_metrics": {"ground_truth_hit": True},
                "p_metrics": {
                    "ground_truth_hit": True,
                    "ground_truth_pixels": 100,
                    "ground_truth_recall": 1.0,
                },
                "candidate_metrics": {
                    "ground_truth_hit": True,
                    "ground_truth_pixels": 100,
                    "ground_truth_recall": 1.0,
                },
                "consensus_metrics": {
                    "p_ground_truth_positive_pixels": 100,
                    "p_candidate_ground_truth_overlap_pixels": 96,
                },
            }
        )

    summary = summarize_dual_evidence_results(rows)

    assert summary["localized_hits"] == {"raw": 80, "candidate": 80}
    assert summary["white_spot"]["localized_hits"] == {
        "raw": 12,
        "candidate": 12,
    }
    assert all(
        group["reduction_fraction_vs_raw"] == 0.3
        for group in summary["ok_subgroups"].values()
    )
    assert summary["p_ground_truth_positive_pixel_preservation_fraction"] == 0.96
    assert summary["screening_gate"]["passed"] is True

    extra = {
        **rows[-1],
        "id": "ng_9999",
        "metadata": {"defect_type": "Scratch"},
    }
    changed = summarize_dual_evidence_results([*rows, extra])
    assert changed["localized_denominator"] == 81
    assert changed["screening_gate"]["passed"] is False


def test_spec_freezes_dual_evidence_formula_and_thresholds() -> None:
    spec = {
        "candidate_id": "primary_weighted_consensus_v1",
        "interpretation": "development_set_selected_diagnostic_not_sealed_test",
        "formula": "cuberoot(square(p_map)_times_auxiliary_map)",
        "inputs": {
            "source_receipt_sha256": "a" * 64,
            "normal_peak_receipt_sha256": "b" * 64,
            "auxiliary_max_receipt_sha256": "c" * 64,
            "source_results_count": 115,
            "ok_count": 27,
            "ng_count": 88,
            "localized_ng_count": 80,
            "white_spot_count": 12,
        },
        "evaluation": {
            "decision_threshold": 0.5,
            "decision_comparator": "strictly_greater_than",
            "parameter_sweeps_allowed": 0,
            "localized_ng_required_hits": 80,
            "white_spot_required_hits": 12,
            "minimum_p_ground_truth_positive_pixel_preservation_fraction": 0.95,
            "minimum_ok_positive_pixel_reduction_fraction_vs_raw": 0.25,
            "all_ok_subgroups_must_reduce": True,
        },
    }

    settings = validate_dual_evidence_spec(spec)

    assert settings["decision_threshold"] == 0.5
    assert settings["localized_ng_required_hits"] == 80
    changed = {
        **spec,
        "formula": "sqrt(p_map_times_auxiliary_map)",
    }
    try:
        validate_dual_evidence_spec(changed)
    except ValueError as error:
        assert "formula" in str(error)
    else:
        raise AssertionError("formula change must be rejected")
