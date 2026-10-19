from __future__ import annotations

from typing import Mapping, Sequence
import unicodedata

import numpy as np


_CANDIDATE_ID = "primary_weighted_consensus_v1"
_FORMULA = "cuberoot(square(p_map)_times_auxiliary_map)"


def _probability_map(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional map")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities")
    return array


def fuse_primary_weighted_consensus(
    *,
    p_map: np.ndarray,
    auxiliary_map: np.ndarray,
) -> np.ndarray:
    """Use P as primary evidence and M as a lower-weight corroborating signal."""
    p = _probability_map(p_map, name="p_map")
    auxiliary = _probability_map(auxiliary_map, name="auxiliary_map")
    if p.shape != auxiliary.shape:
        raise ValueError("all evidence maps must have the same shape")
    weighted_product = np.square(p, dtype=np.float32) * auxiliary
    return np.cbrt(weighted_product, dtype=np.float32)


def validate_dual_evidence_spec(spec: Mapping[str, object]) -> dict[str, object]:
    if spec.get("candidate_id") != _CANDIDATE_ID:
        raise ValueError("unexpected candidate ID")
    if (
        spec.get("interpretation")
        != "development_set_selected_diagnostic_not_sealed_test"
    ):
        raise ValueError("candidate interpretation must disclose development selection")
    if spec.get("formula") != _FORMULA:
        raise ValueError("candidate formula differs from the frozen contract")

    inputs = dict(spec.get("inputs", {}))
    expected_counts = {
        "source_results_count": 115,
        "ok_count": 27,
        "ng_count": 88,
        "localized_ng_count": 80,
        "white_spot_count": 12,
    }
    if any(inputs.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("frozen cohort counts differ from the contract")
    digest_keys = (
        "source_receipt_sha256",
        "normal_peak_receipt_sha256",
        "auxiliary_max_receipt_sha256",
    )
    digests: dict[str, str] = {}
    for key in digest_keys:
        value = str(inputs.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
        digests[key] = value

    evaluation = dict(spec.get("evaluation", {}))
    expected_evaluation = {
        "decision_threshold": 0.5,
        "decision_comparator": "strictly_greater_than",
        "parameter_sweeps_allowed": 0,
        "localized_ng_required_hits": 80,
        "white_spot_required_hits": 12,
        "minimum_p_ground_truth_positive_pixel_preservation_fraction": 0.95,
        "minimum_ok_positive_pixel_reduction_fraction_vs_raw": 0.25,
        "all_ok_subgroups_must_reduce": True,
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise ValueError(f"evaluation {key} differs from the frozen contract")
    return {**expected_evaluation, **digests}


def _ok_subgroup(identifier: str) -> str:
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


def summarize_dual_evidence_results(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("cohort") == "ok"]
    localized = [row for row in rows if row.get("label_status") == "localized"]
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
    overall_reduction = (
        (raw_ok - ok_pixels["candidate"]) / raw_ok if raw_ok > 0 else 0.0
    )
    subgroups: dict[str, object] = {}
    subgroup_reductions: list[float] = []
    expected_subgroup_denominators = {"a": 8, "b": 8, "c": 8, "extra": 3}
    for name in expected_subgroup_denominators:
        selected = [
            row for row in ok_rows if _ok_subgroup(str(row.get("id", ""))) == name
        ]
        pixels = {
            key: sum(
                int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"])
                for row in selected
            )
            for key in ("raw", "candidate")
        }
        raw_pixels = pixels["raw"]
        reduction = (
            (raw_pixels - pixels["candidate"]) / raw_pixels
            if raw_pixels > 0
            else 0.0
        )
        subgroup_reductions.append(reduction)
        subgroups[name] = {
            "denominator": len(selected),
            "positive_pixels": pixels,
            "images_with_activation": {
                key: sum(
                    int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"])
                    > 0
                    for row in selected
                )
                for key in ("raw", "candidate")
            },
            "reduction_fraction_vs_raw": reduction,
        }

    localized_hits = {
        key: hits(key, localized) for key in ("raw", "candidate")
    }
    white_spot_hits = {
        key: hits(key, white_spot) for key in ("raw", "candidate")
    }

    def consensus_pixels(key: str) -> int:
        return sum(
            int(dict(row["consensus_metrics"])[key])
            for row in localized
        )

    p_true_positive_pixels = consensus_pixels("p_ground_truth_positive_pixels")
    candidate_true_positive_pixels = consensus_pixels(
        "p_candidate_ground_truth_overlap_pixels"
    )
    p_preservation = (
        candidate_true_positive_pixels / p_true_positive_pixels
        if p_true_positive_pixels > 0
        else 0.0
    )
    screening_gate = {
        "localized_required_denominator": 80,
        "localized_observed_denominator": len(localized),
        "localized_required_hits": 80,
        "localized_observed_hits": localized_hits["candidate"],
        "white_spot_required_denominator": 12,
        "white_spot_observed_denominator": len(white_spot),
        "white_spot_required_hits": 12,
        "white_spot_observed_hits": white_spot_hits["candidate"],
        "minimum_p_ground_truth_positive_pixel_preservation_fraction": 0.95,
        "observed_p_ground_truth_positive_pixel_preservation_fraction": p_preservation,
        "minimum_ok_positive_pixel_reduction_fraction_vs_raw": 0.25,
        "observed_ok_positive_pixel_reduction_fraction_vs_raw": overall_reduction,
        "all_ok_subgroups_must_reduce": True,
        "all_ok_subgroups_reduced": all(value > 0.0 for value in subgroup_reductions),
        "ok_required_denominator": 27,
        "ok_observed_denominator": len(ok_rows),
        "ok_required_subgroup_denominators": expected_subgroup_denominators,
        "ok_observed_subgroup_denominators": {
            name: int(dict(value)["denominator"])
            for name, value in subgroups.items()
        },
        "ok_unclassified_count": sum(
            _ok_subgroup(str(row.get("id", ""))) == "unclassified"
            for row in ok_rows
        ),
    }
    subgroup_denominators_match = screening_gate[
        "ok_observed_subgroup_denominators"
    ] == expected_subgroup_denominators
    screening_gate["passed"] = bool(
        len(localized) == 80
        and localized_hits["candidate"] == 80
        and len(white_spot) == 12
        and white_spot_hits["candidate"] == 12
        and p_preservation >= 0.95
        and overall_reduction >= 0.25
        and screening_gate["all_ok_subgroups_reduced"]
        and len(ok_rows) == 27
        and subgroup_denominators_match
        and screening_gate["ok_unclassified_count"] == 0
    )
    return {
        "localized_denominator": len(localized),
        "localized_hits": localized_hits,
        "white_spot": {
            "denominator": len(white_spot),
            "localized_hits": white_spot_hits,
        },
        "p_ground_truth_positive_pixels": p_true_positive_pixels,
        "candidate_ground_truth_positive_pixels": candidate_true_positive_pixels,
        "p_ground_truth_positive_pixel_preservation_fraction": p_preservation,
        "ok_denominator": len(ok_rows),
        "ok_positive_pixels": ok_pixels,
        "ok_reduction_fraction_vs_raw": overall_reduction,
        "ok_images_with_activation": {
            key: sum(
                int(dict(row[f"{key}_metrics"])["predicted_positive_pixels"]) > 0
                for row in ok_rows
            )
            for key in ("raw", "candidate")
        },
        "ok_subgroups": subgroups,
        "screening_gate": screening_gate,
    }
