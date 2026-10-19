from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_summary_keeps_a_baseline_and_peak_candidate_separate() -> None:
    from patch_qc.normal_peak_spike import summarize_peak_results

    rows = [
        {
            "cohort": "ok",
            "prior_usable": True,
            "preserve_pixels_native": 4,
            "metrics": {
                "a": {"predicted_positive_pixels": 10},
                "b": {"predicted_positive_pixels": 4},
                "p": {"predicted_positive_pixels": 6},
            },
        },
        {
            "cohort": "ng",
            "prior_usable": True,
            "preserve_pixels_native": 8,
            "metrics": {
                "a": {
                    "predicted_positive_pixels": 9,
                    "ground_truth_hit": True,
                    "ground_truth_recall": 0.5,
                    "ground_truth_iou": 0.2,
                },
                "b": {
                    "predicted_positive_pixels": 3,
                    "ground_truth_hit": False,
                    "ground_truth_recall": 0.0,
                    "ground_truth_iou": 0.0,
                },
                "p": {
                    "predicted_positive_pixels": 5,
                    "ground_truth_hit": True,
                    "ground_truth_recall": 0.4,
                    "ground_truth_iou": 0.25,
                },
            },
        },
        {
            "cohort": "ng",
            "prior_usable": False,
            "preserve_pixels_native": 0,
            "metrics": {
                "a": {"predicted_positive_pixels": 0},
                "b": {"predicted_positive_pixels": 0},
                "p": {"predicted_positive_pixels": 0},
            },
        },
    ]

    summary = summarize_peak_results(rows)

    assert summary["localized_denominator"] == 1
    assert summary["localized_hits"] == {"a": 1, "b": 0, "p": 1}
    assert summary["ok_images_with_activation"] == {"a": 1, "b": 1, "p": 1}
    assert summary["ok_predicted_positive_pixels"] == {"a": 10, "b": 4, "p": 6}
    assert summary["prior_usable_queries"] == 2
    assert summary["preserve_pixels_native"] == 12


def test_strict_metrics_and_delivered_overlay_agree_at_exact_threshold() -> None:
    from patch_qc.normal_peak_spike import score_map_metrics_strict

    score = np.asarray([[0.50, 0.51]], dtype=np.float32)
    truth_at_exact_threshold = np.asarray([[True, False]])

    metrics = score_map_metrics_strict(
        score, threshold=0.5, ground_truth=truth_at_exact_threshold
    )

    assert metrics["predicted_positive_pixels"] == 1
    assert metrics["ground_truth_hit"] is False


def test_delivered_quantization_is_the_single_source_for_saved_map_and_metrics() -> None:
    from patch_qc.normal_peak_spike import quantize_delivered_score_map

    encoded, delivered = quantize_delivered_score_map(
        np.asarray([[0.50, 0.25]], dtype=np.float32)
    )

    assert encoded.dtype == np.uint8
    np.testing.assert_array_equal(encoded, [[128, 64]])
    np.testing.assert_allclose(delivered, encoded.astype(np.float32) / 255.0)


def test_inference_contract_rejects_reference_stride_mismatch() -> None:
    from patch_qc.normal_peak_spike import validate_inference_compatibility

    source = {
        "fixed_settings": {
            "aggregation": "uniform_mean",
            "batch_size": 128,
            "patch_size": 256,
            "stride": 128,
        },
        "digests": {
            "base_checkpoint_sha256": "base",
            "multitask_checkpoint_sha256": "multi",
            "training_receipt_sha256": "training",
        },
        "checkpoint_metadata": {
            "backbone_id": "backbone",
            "config_digest": "config",
        },
    }
    reference = {
        "stage": "normal_reference_raw_inference",
        "settings": {
            "batch_size": 128,
            "patch_size": 256,
            "stride": 256,
        },
        "digests": source["digests"],
        "checkpoint_metadata": source["checkpoint_metadata"],
    }

    try:
        validate_inference_compatibility(source, [reference])
    except ValueError as error:
        assert "stride" in str(error)
    else:
        raise AssertionError("stride mismatch was accepted")


def test_tiny_sealed_export_verifies_contract_and_fails_open(tmp_path: Path) -> None:
    from patch_qc.normal_peak_spike import (
        _runtime_code_digests,
        export_peak_comparisons,
    )

    original = tmp_path / "original.png"
    Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8)).save(original)
    original_sha = _sha256(original)

    source = tmp_path / "source"
    raw_path = source / "raw_heatmaps" / "ok" / "ok_0001.png"
    raw_path.parent.mkdir(parents=True)
    raw = np.zeros((8, 8), dtype=np.uint8)
    raw[3:5, 3:5] = 255
    Image.fromarray(raw).save(raw_path)
    predictions = source / "predictions.jsonl"
    prediction = {
        "id": "ok_0001",
        "cohort": "ok",
        "image_path": str(original),
        "image_sha256": original_sha,
        "label_path": None,
        "label_sha256": None,
        "metadata": {"product": "fixture"},
        "outputs": {"raw_heatmap": "raw_heatmaps/ok/ok_0001.png"},
    }
    predictions.write_text(json.dumps(prediction, sort_keys=True) + "\n")
    model_digests = {
        "base_checkpoint_sha256": "base",
        "multitask_checkpoint_sha256": "multi",
        "training_receipt_sha256": "training",
    }
    checkpoint = {"backbone_id": "backbone", "config_digest": "config"}
    source_receipt = {
        "status": "completed",
        "fixed_settings": {
            "aggregation": "uniform_mean",
            "batch_size": 128,
            "patch_size": 256,
            "stride": 128,
        },
        "digests": model_digests,
        "checkpoint_metadata": checkpoint,
        "artifacts": {
            "predictions.jsonl": _sha256(predictions),
            "raw_heatmaps/ok/ok_0001.png": _sha256(raw_path),
        },
    }
    source_receipt_path = source / "run_receipt.json"
    _write_json(source_receipt_path, source_receipt)

    reference_ids = ["ref_1", "ref_2", "ref_3"]
    plan = {
        "queries": [
            {
                "id": "ok_0001",
                "image_path": str(original),
                "image_sha256": original_sha,
                "status": "ready_visual_nearest_3",
                "selected_reference_ids": reference_ids,
            }
        ],
        "references": [
            {
                "reference_id": identifier,
                "image_path": str(original),
                "image_sha256": original_sha,
            }
            for identifier in reference_ids
        ],
    }
    plan_path = tmp_path / "selected_plan.json"
    _write_json(plan_path, plan)

    shard = tmp_path / "reference_shard"
    artifacts: dict[str, str] = {}
    records = []
    for identifier in reference_ids:
        preview = shard / "previews" / f"{identifier}.jpg"
        score = shard / "raw_maps" / f"{identifier}.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        score.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8)).save(preview)
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(score)
        preview_relative = preview.relative_to(shard).as_posix()
        score_relative = score.relative_to(shard).as_posix()
        artifacts[preview_relative] = _sha256(preview)
        artifacts[score_relative] = _sha256(score)
        records.append(
            {
                "id": identifier,
                "image_path": str(original),
                "image_sha256": original_sha,
                "map_shape": [8, 8],
                "outputs": {"preview": preview_relative, "raw_map": score_relative},
            }
        )
    records_path = shard / "references.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    artifacts["references.jsonl"] = _sha256(records_path)
    reference_receipt = {
        "status": "completed",
        "stage": "normal_reference_raw_inference",
        "settings": {"batch_size": 128, "patch_size": 256, "stride": 128},
        "digests": model_digests,
        "checkpoint_metadata": checkpoint,
        "artifacts": artifacts,
    }
    reference_receipt_path = shard / "run_receipt.json"
    _write_json(reference_receipt_path, reference_receipt)

    inference_contract = {
        "aggregation": "uniform_mean",
        "batch_size": 128,
        "patch_size": 256,
        "stride": 128,
        **model_digests,
        **checkpoint,
    }
    spec = {
        "schema_version": 2,
        "candidate_id": "normal_reference_relative_peak_v2",
        "interpretation": "development_diagnostic_not_sealed_test",
        "runtime_code_sha256": _runtime_code_digests(),
        "inputs": {
            "source_a_receipt_sha256": _sha256(source_receipt_path),
            "selected_plan_sha256": _sha256(plan_path),
            "reference_shard_receipt_sha256": [_sha256(reference_receipt_path)],
        },
        "inference_contract": inference_contract,
        "registration": {"compact_width": 512, "all_three_required": True},
        "recurrence": {
            "reference_score_threshold": 0.5,
            "required_fraction": 2.0 / 3.0,
            "dilation_radius_compact_px": 1,
            "suppression_strength": 0.5,
        },
        "peak_preservation": {"regional_maximum_radius_compact_px": 1},
        "evaluation": {
            "decision_threshold": 0.5,
            "decision_comparator": "strictly_greater_than",
            "visualization_comparator": "strictly_greater_than",
            "parameter_sweeps_allowed": 0,
            "localized_ng_required_no_loss_vs_a": True,
            "minimum_ok_positive_pixel_reduction_fraction_vs_a": 0.1,
            "white_spot_reported_separately": True,
            "unlabeled_ng_reported_separately": True,
        },
    }
    spec_path = tmp_path / "candidate_spec.json"
    _write_json(spec_path, spec)
    destination = tmp_path / "export"

    receipt = export_peak_comparisons(
        candidate_spec_path=spec_path,
        expected_candidate_spec_sha256=_sha256(spec_path),
        selected_plan_path=plan_path,
        source_run=source,
        reference_shard_dirs=[shard],
        output_dir=destination,
    )

    assert receipt["status"] == "completed"
    assert receipt["candidate_status"] == "adoption_gate_rejected"
    assert receipt["counts"]["total"] == 1
    assert receipt["summary"]["prior_usable_queries"] == 0
    assert receipt["summary"]["preserve_pixels_native"] == 0
    assert (destination / "candidate_comparisons/ok/ok_0001.jpg").exists()
    assert not list(tmp_path.glob(".export.attempt-*"))
