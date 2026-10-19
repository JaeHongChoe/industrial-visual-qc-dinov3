from __future__ import annotations

import numpy as np
import torch
from torch import nn

from patch_qc.inference import Box
from patch_qc.model import MultiTaskOutput
from patch_qc.structural_suppression_spike import (
    predict_auxiliary_max_image,
    stitch_auxiliary_weighted_max,
    summarize_auxiliary_max_results,
    validate_structural_candidate_spec,
)


class _BrightnessMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> MultiTaskOutput:
        logits = pixel_values.mean(dim=(1, 2, 3)) + self.anchor
        masks = torch.zeros(
            (pixel_values.shape[0], 1, pixel_values.shape[2], pixel_values.shape[3]),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return MultiTaskOutput(logits, logits, masks)


def test_auxiliary_weighted_max_keeps_one_supported_overlap_without_denominator_dilution() -> None:
    """A valid defect patch must not be divided by unrelated overlapping patches."""
    boxes = (
        Box(0, 0, 2, 2),
        Box(1, 0, 3, 2),
    )
    masks = np.asarray(
        [
            [[1.0, 0.5], [0.25, 0.0]],
            [[0.5, 1.0], [1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    auxiliary = np.asarray([0.4, 0.8], dtype=np.float32)

    actual = stitch_auxiliary_weighted_max(
        image_shape=(2, 3),
        boxes=boxes,
        masks=masks,
        auxiliary_scores=auxiliary,
    )

    expected = np.asarray(
        [[0.4, 0.4, 0.8], [0.1, 0.8, 0.4]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_auxiliary_weighted_max_rejects_non_probability_scores() -> None:
    boxes = (Box(0, 0, 2, 2),)
    masks = np.ones((1, 2, 2), dtype=np.float32)

    try:
        stitch_auxiliary_weighted_max(
            image_shape=(2, 2),
            boxes=boxes,
            masks=masks,
            auxiliary_scores=np.asarray([1.1], dtype=np.float32),
        )
    except ValueError as error:
        assert "probabilities" in str(error)
    else:
        raise AssertionError("out-of-range auxiliary scores must be rejected")


def test_predict_auxiliary_max_is_batch_invariant_and_uses_max_support() -> None:
    """Changing runtime batch size must not change the candidate heatmap."""
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, 2] = 255
    model = _BrightnessMaskModel()

    serial = predict_auxiliary_max_image(
        model,
        rgb,
        patch_size=2,
        stride=1,
        batch_size=1,
        device="cpu",
        use_amp=False,
    )
    batched = predict_auxiliary_max_image(
        model,
        rgb,
        patch_size=2,
        stride=1,
        batch_size=2,
        device="cpu",
        use_amp=False,
    )

    expected = np.empty((2, 3), dtype=np.float32)
    expected[:, 0] = 0.5 * serial.auxiliary_probabilities[0]
    expected[:, 1] = 0.5 * max(serial.auxiliary_probabilities)
    expected[:, 2] = 0.5 * serial.auxiliary_probabilities[1]
    np.testing.assert_allclose(serial.auxiliary_max_score_map, expected)
    np.testing.assert_allclose(
        serial.auxiliary_max_score_map, batched.auxiliary_max_score_map
    )
    np.testing.assert_allclose(serial.raw_score_map, 0.5)
    assert serial.patch_count == 2


def test_summary_requires_no_localized_hit_loss_and_reports_ok_reduction() -> None:
    def row(
        identifier: str,
        cohort: str,
        raw_pixels: int,
        candidate_pixels: int,
        raw_hit: bool | None,
        candidate_hit: bool | None,
    ) -> dict[str, object]:
        raw: dict[str, object] = {"predicted_positive_pixels": raw_pixels}
        candidate: dict[str, object] = {
            "predicted_positive_pixels": candidate_pixels
        }
        status = "ok_unlabeled"
        if raw_hit is not None and candidate_hit is not None:
            raw["ground_truth_hit"] = raw_hit
            candidate["ground_truth_hit"] = candidate_hit
            status = "localized"
        return {
            "id": identifier,
            "cohort": cohort,
            "label_status": status,
            "metadata": {},
            "raw_metrics": raw,
            "candidate_metrics": candidate,
        }

    summary = summarize_auxiliary_max_results(
        [
            row("ok_0001", "ok", 100, 70, None, None),
            row("ok_0002", "ok", 100, 80, None, None),
            row("ng_0001", "ng", 10, 5, True, True),
            row("ng_0002", "ng", 10, 0, True, False),
        ]
    )

    assert summary["localized_hits"] == {"raw": 2, "candidate": 1}
    assert summary["ok_positive_pixels"] == {"raw": 200, "candidate": 150}
    assert summary["ok_reduction_fraction_vs_raw"] == 0.25
    assert summary["ok_subgroups"]["a"]["denominator"] == 2
    assert summary["ok_subgroups"]["a"]["positive_pixels"] == {
        "raw": 200,
        "candidate": 150,
    }
    assert summary["screening_gate"]["passed"] is False


def test_structural_candidate_spec_freezes_one_formula_and_no_parameter_sweep() -> None:
    spec = {
        "candidate_id": "auxiliary_weighted_max_v1",
        "interpretation": "development_diagnostic_not_sealed_test",
        "formula": "max_over_overlaps(mask_probability_times_auxiliary_probability)",
        "inputs": {
            "source_receipt_sha256": "a" * 64,
            "source_results_count": 115,
            "ok_count": 27,
            "ng_count": 88,
            "localized_ng_count": 80,
            "base_checkpoint_sha256": "b" * 64,
            "multitask_checkpoint_sha256": "c" * 64,
            "training_receipt_sha256": "d" * 64,
        },
        "inference": {
            "patch_size": 256,
            "stride": 128,
            "batch_size": 128,
            "amp": True,
        },
        "evaluation": {
            "decision_threshold": 0.5,
            "decision_comparator": "strictly_greater_than",
            "visualization_comparator": "strictly_greater_than",
            "parameter_sweeps_allowed": 0,
            "localized_ng_required_no_loss_vs_raw": True,
            "minimum_ok_positive_pixel_reduction_fraction_vs_raw": 0.1,
        },
    }

    actual = validate_structural_candidate_spec(spec)

    assert actual == {
        "decision_threshold": 0.5,
        "patch_size": 256,
        "stride": 128,
        "batch_size": 128,
        "use_amp": True,
        "source_receipt_sha256": "a" * 64,
        "base_checkpoint_sha256": "b" * 64,
        "multitask_checkpoint_sha256": "c" * 64,
        "training_receipt_sha256": "d" * 64,
    }
    changed = {**spec, "formula": "mean"}
    try:
        validate_structural_candidate_spec(changed)
    except ValueError as error:
        assert "formula" in str(error)
    else:
        raise AssertionError("a different fusion formula must be rejected")

    changed = {**spec, "inference": {**spec["inference"], "stride": 192}}
    try:
        validate_structural_candidate_spec(changed)
    except ValueError as error:
        assert "stride" in str(error)
    else:
        raise AssertionError("a different stride must be rejected")

    changed = {**spec, "inputs": {**spec["inputs"], "ok_count": 26}}
    try:
        validate_structural_candidate_spec(changed)
    except ValueError as error:
        assert "cohort" in str(error)
    else:
        raise AssertionError("a different frozen cohort must be rejected")
