from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
import unicodedata

import cv2
import numpy as np
from PIL import Image

from .actual_multitask_eval import _write_image_atomic
from .actual_multitask_inference import (
    _sha256_file,
    load_labelme_mask,
)
from .normal_prior_postprocess import (
    PeakPreservedScores,
    aggregate_binary_recurrence_prior,
    build_reference_relative_peak_candidate,
    render_original_overlay_comparison,
)
from .normal_prior_spike import (
    _load_compact_gray,
    _load_json,
    _load_jsonl,
    _utc_now,
    _write_json,
    _write_jsonl,
)


_EXPECTED_CANDIDATE_ID = "normal_reference_relative_peak_v2"


def quantize_delivered_score_map(
    score_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score_map, dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("score_map must be a finite two-dimensional array")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("score_map must contain values in [0, 1]")
    encoded = np.rint(score * 255.0).astype(np.uint8)
    delivered = encoded.astype(np.float32) / 255.0
    return encoded, delivered


def score_map_metrics_strict(
    score_map: np.ndarray,
    *,
    threshold: float,
    ground_truth: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Metrics whose positive comparator matches the delivered overlay: score > threshold."""
    score = np.asarray(score_map, dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("score_map must be a finite two-dimensional array")
    if np.any((score < 0.0) | (score > 1.0)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("score_map and threshold must be probabilities")
    predicted = score > threshold
    metrics: dict[str, float | int | bool] = {
        "predicted_positive_pixels": int(predicted.sum()),
        "predicted_positive_fraction": float(predicted.mean()),
        "maximum_score": float(score.max(initial=0.0)),
        "p99_score": float(np.quantile(score, 0.99)),
    }
    if ground_truth is None:
        return metrics
    truth = np.asarray(ground_truth, dtype=bool)
    if truth.shape != score.shape:
        raise ValueError("ground_truth shape must match score_map")
    truth_pixels = int(truth.sum())
    if truth_pixels <= 0:
        raise ValueError("ground_truth must contain at least one positive pixel")
    intersection = int(np.logical_and(predicted, truth).sum())
    union = int(np.logical_or(predicted, truth).sum())
    metrics.update(
        {
            "ground_truth_pixels": truth_pixels,
            "ground_truth_hit": intersection > 0,
            "ground_truth_recall": intersection / truth_pixels,
            "ground_truth_iou": intersection / union if union else 0.0,
            "maximum_score_inside_ground_truth": float(score[truth].max()),
        }
    )
    return metrics


def validate_inference_compatibility(
    source_receipt: Mapping[str, Any],
    reference_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    source_settings = dict(source_receipt["fixed_settings"])
    if source_settings.get("aggregation") != "uniform_mean":
        raise ValueError("source aggregation must be uniform_mean")
    expected_settings = {
        key: source_settings[key] for key in ("batch_size", "patch_size", "stride")
    }
    source_digests = dict(source_receipt["digests"])
    expected_digests = {
        key: source_digests[key]
        for key in (
            "base_checkpoint_sha256",
            "multitask_checkpoint_sha256",
            "training_receipt_sha256",
        )
    }
    source_checkpoint = dict(source_receipt["checkpoint_metadata"])
    expected_checkpoint = {
        key: source_checkpoint[key] for key in ("backbone_id", "config_digest")
    }
    for index, receipt in enumerate(reference_receipts):
        if receipt.get("stage") != "normal_reference_raw_inference":
            raise ValueError(f"reference {index} has an unexpected stage")
        settings = dict(receipt["settings"])
        for key, expected in expected_settings.items():
            if settings.get(key) != expected:
                raise ValueError(
                    f"reference {index} {key} mismatch: "
                    f"expected {expected}, got {settings.get(key)}"
                )
        digests = dict(receipt["digests"])
        for key, expected in expected_digests.items():
            if digests.get(key) != expected:
                raise ValueError(f"reference {index} {key} mismatch")
        checkpoint = dict(receipt["checkpoint_metadata"])
        for key, expected in expected_checkpoint.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"reference {index} checkpoint {key} mismatch")
    return {
        "aggregation": "uniform_mean",
        **expected_settings,
        **expected_digests,
        **expected_checkpoint,
        "reference_receipts": len(reference_receipts),
    }


def summarize_peak_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    localized = [
        row
        for row in rows
        if "ground_truth_hit" in dict(dict(row["metrics"])["a"])
    ]
    ok_rows = [row for row in rows if row["cohort"] == "ok"]
    unlabeled_ng = [
        row
        for row in rows
        if row["cohort"] == "ng"
        and "ground_truth_hit" not in dict(dict(row["metrics"])["a"])
    ]
    white_spot = [
        row
        for row in rows
        if row["cohort"] == "ng"
        and unicodedata.normalize(
            "NFC", str(dict(row.get("metadata", {})).get("defect_type", ""))
        )
        == "접합면 White Spot"
    ]
    keys = ("a", "b", "p")
    localized_hits = {
        key: sum(
            bool(dict(dict(row["metrics"])[key])["ground_truth_hit"])
            for row in localized
        )
        for key in keys
    }
    ok_positive_pixels = {
        key: sum(
            int(dict(dict(row["metrics"])[key])["predicted_positive_pixels"])
            for row in ok_rows
        )
        for key in keys
    }
    a_ok_pixels = ok_positive_pixels["a"]
    ok_reduction = (
        (a_ok_pixels - ok_positive_pixels["p"]) / a_ok_pixels
        if a_ok_pixels
        else 0.0
    )
    localized_no_loss = localized_hits["p"] >= localized_hits["a"]
    minimum_ok_reduction_met = ok_reduction >= 0.10
    return {
        "localized_denominator": len(localized),
        "localized_hits": localized_hits,
        "localized_mean_recall": {
            key: float(
                np.mean(
                    [
                        float(dict(dict(row["metrics"])[key])["ground_truth_recall"])
                        for row in localized
                    ]
                )
            )
            if localized
            else 0.0
            for key in keys
        },
        "localized_mean_iou": {
            key: float(
                np.mean(
                    [
                        float(dict(dict(row["metrics"])[key])["ground_truth_iou"])
                        for row in localized
                    ]
                )
            )
            if localized
            else 0.0
            for key in keys
        },
        "ok_images_with_activation": {
            key: sum(
                int(dict(dict(row["metrics"])[key])["predicted_positive_pixels"]) > 0
                for row in ok_rows
            )
            for key in keys
        },
        "ok_predicted_positive_pixels": ok_positive_pixels,
        "ok_candidate_positive_pixel_reduction_fraction_vs_a": ok_reduction,
        "white_spot": {
            "denominator": len(white_spot),
            "localized_hits": {
                key: sum(
                    bool(dict(dict(row["metrics"])[key]).get("ground_truth_hit", False))
                    for row in white_spot
                )
                for key in keys
            },
            "images_with_activation": {
                key: sum(
                    int(dict(dict(row["metrics"])[key])["predicted_positive_pixels"]) > 0
                    for row in white_spot
                )
                for key in keys
            },
        },
        "unlabeled_ng": {
            "denominator": len(unlabeled_ng),
            "images_with_activation": {
                key: sum(
                    int(dict(dict(row["metrics"])[key])["predicted_positive_pixels"]) > 0
                    for row in unlabeled_ng
                )
                for key in keys
            },
        },
        "prior_usable_queries": sum(bool(row["prior_usable"]) for row in rows),
        "preserve_pixels_native": sum(
            int(row["preserve_pixels_native"]) for row in rows
        ),
        "adoption_gate": {
            "localized_no_loss_vs_a": localized_no_loss,
            "minimum_ok_positive_pixel_reduction_fraction_vs_a": 0.10,
            "observed_ok_positive_pixel_reduction_fraction_vs_a": ok_reduction,
            "minimum_ok_reduction_met": minimum_ok_reduction_met,
            "passed": localized_no_loss and minimum_ok_reduction_met,
        },
    }


def _validate_candidate_spec(spec: Mapping[str, Any]) -> dict[str, float | int]:
    if spec.get("candidate_id") != _EXPECTED_CANDIDATE_ID:
        raise ValueError("unexpected candidate ID")
    if spec.get("interpretation") != "development_diagnostic_not_sealed_test":
        raise ValueError("candidate must remain a development diagnostic")
    registration = dict(spec["registration"])
    recurrence = dict(spec["recurrence"])
    preservation = dict(spec["peak_preservation"])
    evaluation = dict(spec["evaluation"])
    expected = {
        "compact_width": 512,
        "reference_score_threshold": 0.5,
        "required_fraction": 2.0 / 3.0,
        "dilation_radius_compact_px": 1,
        "suppression_strength": 0.5,
        "regional_maximum_radius_compact_px": 1,
        "decision_threshold": 0.5,
    }
    actual = {
        "compact_width": int(registration["compact_width"]),
        "reference_score_threshold": float(recurrence["reference_score_threshold"]),
        "required_fraction": float(recurrence["required_fraction"]),
        "dilation_radius_compact_px": int(
            recurrence["dilation_radius_compact_px"]
        ),
        "suppression_strength": float(recurrence["suppression_strength"]),
        "regional_maximum_radius_compact_px": int(
            preservation["regional_maximum_radius_compact_px"]
        ),
        "decision_threshold": float(evaluation["decision_threshold"]),
    }
    if actual != expected:
        raise ValueError(f"candidate settings differ from frozen contract: {actual}")
    if registration.get("all_three_required") is not True:
        raise ValueError("all three normal references must be required")
    if evaluation.get("decision_comparator") != "strictly_greater_than":
        raise ValueError("decision comparator must be strictly_greater_than")
    if evaluation.get("visualization_comparator") != "strictly_greater_than":
        raise ValueError("visualization comparator must match decision comparator")
    if evaluation.get("parameter_sweeps_allowed") != 0:
        raise ValueError("parameter sweeps are forbidden")
    if evaluation.get("localized_ng_required_no_loss_vs_a") is not True:
        raise ValueError("localized NG no-loss gate must be enabled")
    if float(evaluation.get("minimum_ok_positive_pixel_reduction_fraction_vs_a")) != 0.1:
        raise ValueError("OK reduction gate must remain 0.1")
    if evaluation.get("white_spot_reported_separately") is not True:
        raise ValueError("White Spot subgroup reporting must be enabled")
    if evaluation.get("unlabeled_ng_reported_separately") is not True:
        raise ValueError("unlabeled NG reporting must be enabled")
    return actual


def _reference_records(
    *,
    plan: Mapping[str, Any],
    reference_shard_dirs: Sequence[str | Path],
    expected_receipt_sha256: Sequence[str],
) -> tuple[
    dict[str, tuple[dict[str, Any], Path]],
    tuple[int, int],
    list[dict[str, Any]],
]:
    if len(reference_shard_dirs) != len(expected_receipt_sha256):
        raise ValueError("reference shard count differs from frozen candidate")
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    receipts: list[dict[str, Any]] = []
    original_shape: tuple[int, int] | None = None
    for shard_value, expected_digest in zip(
        reference_shard_dirs, expected_receipt_sha256, strict=True
    ):
        shard = Path(shard_value)
        receipt_path = shard / "run_receipt.json"
        if _sha256_file(receipt_path) != expected_digest:
            raise ValueError(f"reference receipt SHA-256 mismatch: {shard}")
        receipt = _load_json(receipt_path)
        receipts.append(receipt)
        artifacts = receipt.get("artifacts")
        if receipt.get("status") != "completed" or not isinstance(artifacts, Mapping):
            raise ValueError(f"reference shard is incomplete: {shard}")
        records_path = shard / "references.jsonl"
        if _sha256_file(records_path) != artifacts.get("references.jsonl"):
            raise ValueError("reference records SHA-256 mismatch")
        for record in _load_jsonl(records_path):
            identifier = str(record["id"])
            if identifier in records:
                raise ValueError("duplicate reference inference record")
            shape = tuple(int(value) for value in record["map_shape"])
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError("invalid reference map shape")
            if original_shape is None:
                original_shape = shape
            elif shape != original_shape:
                raise ValueError("reference map shapes differ")
            for relative in dict(record["outputs"]).values():
                relative_string = str(relative)
                if _sha256_file(shard / relative_string) != artifacts.get(relative_string):
                    raise ValueError(
                        f"reference artifact SHA-256 mismatch: {relative_string}"
                    )
            records[identifier] = (record, shard)
    expected_by_id = {
        str(row["reference_id"]): row for row in plan["references"]
    }
    if set(records) != set(expected_by_id):
        raise ValueError("reference inference coverage differs from selected plan")
    for identifier, (record, _) in records.items():
        planned = expected_by_id[identifier]
        for key in ("image_path", "image_sha256"):
            if record.get(key) != planned.get(key):
                raise ValueError(f"reference {identifier} {key} differs from plan")
    if original_shape is None:
        raise ValueError("no reference inference records")
    return records, original_shape, receipts


def _fail_open_scores(raw: np.ndarray, shape: tuple[int, int]) -> PeakPreservedScores:
    return PeakPreservedScores(
        baseline=raw.copy(),
        candidate=raw.copy(),
        compact_preserve_mask=np.zeros(shape, dtype=np.bool_),
        native_preserve_mask=np.zeros(raw.shape, dtype=np.bool_),
        native_recurrence_prior=np.zeros(raw.shape, dtype=np.float32),
    )


def _runtime_code_digests() -> dict[str, str]:
    package = Path(__file__).parent
    names = (
        "normal_peak_spike.py",
        "normal_prior_postprocess.py",
        "normal_prior_spike.py",
        "actual_multitask_eval.py",
        "actual_multitask_inference.py",
    )
    return {name: _sha256_file(package / name) for name in names}


def export_peak_comparisons(
    *,
    candidate_spec_path: str | Path,
    expected_candidate_spec_sha256: str,
    selected_plan_path: str | Path,
    source_run: str | Path,
    reference_shard_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    candidate_spec_path = Path(candidate_spec_path)
    if _sha256_file(candidate_spec_path) != expected_candidate_spec_sha256:
        raise ValueError("candidate spec SHA-256 mismatch")
    spec = _load_json(candidate_spec_path)
    settings = _validate_candidate_spec(spec)
    inputs = dict(spec["inputs"])
    initial_code_digests = _runtime_code_digests()
    frozen_code_digests = {
        str(key): str(value)
        for key, value in dict(spec["runtime_code_sha256"]).items()
    }
    if initial_code_digests != frozen_code_digests:
        raise ValueError("runtime code differs from frozen candidate")

    selected_plan_path = Path(selected_plan_path)
    expected_plan_digest = str(inputs["selected_plan_sha256"])
    if _sha256_file(selected_plan_path) != expected_plan_digest:
        raise ValueError("selected plan SHA-256 mismatch")
    plan = _load_json(selected_plan_path)

    source = Path(source_run)
    source_receipt_path = source / "run_receipt.json"
    expected_source_digest = str(inputs["source_a_receipt_sha256"])
    if _sha256_file(source_receipt_path) != expected_source_digest:
        raise ValueError("source receipt SHA-256 mismatch")
    source_receipt = _load_json(source_receipt_path)
    source_artifacts = source_receipt.get("artifacts")
    if source_receipt.get("status") != "completed" or not isinstance(
        source_artifacts, Mapping
    ):
        raise ValueError("source run is not sealed and completed")
    predictions_path = source / "predictions.jsonl"
    if _sha256_file(predictions_path) != source_artifacts.get("predictions.jsonl"):
        raise ValueError("source predictions SHA-256 mismatch")
    source_rows = _load_jsonl(predictions_path)
    query_by_id = {str(row["id"]): row for row in plan["queries"]}
    if set(query_by_id) != {str(row["id"]) for row in source_rows}:
        raise ValueError("selected plan and source query IDs differ")
    for row in source_rows:
        identifier = str(row["id"])
        planned = query_by_id[identifier]
        for key in ("image_path", "image_sha256"):
            if row.get(key) != planned.get(key):
                raise ValueError(f"query {identifier} {key} differs from plan")

    expected_reference_receipts = [
        str(value) for value in inputs["reference_shard_receipt_sha256"]
    ]
    reference_records, original_reference_shape, reference_receipts = _reference_records(
        plan=plan,
        reference_shard_dirs=reference_shard_dirs,
        expected_receipt_sha256=expected_reference_receipts,
    )
    inference_contract = validate_inference_compatibility(
        source_receipt, reference_receipts
    )
    frozen_inference_contract = dict(spec["inference_contract"])
    for key in (
        "aggregation",
        "batch_size",
        "patch_size",
        "stride",
        "base_checkpoint_sha256",
        "multitask_checkpoint_sha256",
        "training_receipt_sha256",
        "backbone_id",
        "config_digest",
    ):
        if frozen_inference_contract.get(key) != inference_contract.get(key):
            raise ValueError(f"frozen inference contract mismatch: {key}")
    compact_width = int(settings["compact_width"])
    compact_height = int(
        round(original_reference_shape[0] * compact_width / original_reference_shape[1])
    )
    compact_shape = (compact_height, compact_width)
    threshold = float(settings["decision_threshold"])

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    attempt.mkdir(parents=True)
    started_at = _utc_now()
    started = time.perf_counter()
    artifacts: dict[str, str] = {}
    result_rows: list[dict[str, object]] = []
    try:
        frozen_spec_path = attempt / "candidate_spec.json"
        shutil.copyfile(candidate_spec_path, frozen_spec_path)
        artifacts["candidate_spec.json"] = _sha256_file(frozen_spec_path)

        for index, row in enumerate(source_rows, start=1):
            identifier = str(row["id"])
            cohort = str(row["cohort"])
            query_plan = query_by_id[identifier]
            raw_relative = str(row["outputs"]["raw_heatmap"])
            raw_path = source / raw_relative
            if _sha256_file(raw_path) != source_artifacts.get(raw_relative):
                raise ValueError(f"source raw map SHA-256 mismatch: {identifier}")
            with Image.open(raw_path) as image:
                raw = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

            image_path = Path(str(row["image_path"]))
            image_bytes = image_path.read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
                raise ValueError(f"query source image SHA-256 mismatch: {identifier}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != raw.shape:
                raise ValueError(f"query image/map shape mismatch: {identifier}")
            query_thumbnail = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0,
                (compact_width, compact_height),
                interpolation=cv2.INTER_AREA,
            )

            selected_ids = [
                str(value) for value in query_plan.get("selected_reference_ids", [])
            ]
            aggregation = None
            if len(selected_ids) == 3:
                reference_thumbnails: list[np.ndarray] = []
                reference_maps: list[np.ndarray] = []
                for reference_id in selected_ids:
                    record, shard = reference_records[reference_id]
                    outputs = dict(record["outputs"])
                    reference_thumbnails.append(
                        _load_compact_gray(
                            shard / str(outputs["preview"]), compact_shape
                        )
                    )
                    reference_maps.append(
                        _load_compact_gray(
                            shard / str(outputs["raw_map"]), compact_shape
                        )
                    )
                aggregation = aggregate_binary_recurrence_prior(
                    query_thumbnail,
                    reference_thumbnails,
                    reference_maps,
                    score_threshold=float(settings["reference_score_threshold"]),
                    dilation_radius_compact_px=int(
                        settings["dilation_radius_compact_px"]
                    ),
                    required_fraction=float(settings["required_fraction"]),
                )
            prior_usable = bool(aggregation is not None and aggregation.usable)
            if prior_usable:
                assert aggregation is not None
                scores = build_reference_relative_peak_candidate(
                    raw,
                    aggregation.prior,
                    aggregation.reference_score_ceiling,
                    radius_compact_px=int(
                        settings["regional_maximum_radius_compact_px"]
                    ),
                    suppression_strength=float(settings["suppression_strength"]),
                )
            else:
                scores = _fail_open_scores(raw, compact_shape)

            baseline_u8, baseline_delivered = quantize_delivered_score_map(
                scores.baseline
            )
            candidate_u8, candidate_delivered = quantize_delivered_score_map(
                scores.candidate
            )
            outputs: dict[str, str] = {}
            for label, relative, image_value, image_format in (
                (
                    "b_map",
                    f"baseline_maps/{cohort}/{identifier}.png",
                    baseline_u8,
                    "PNG",
                ),
                (
                    "p_map",
                    f"candidate_maps/{cohort}/{identifier}.png",
                    candidate_u8,
                    "PNG",
                ),
                (
                    "p_comparison",
                    f"candidate_comparisons/{cohort}/{identifier}.jpg",
                    render_original_overlay_comparison(
                        rgb, candidate_delivered, threshold=threshold, alpha=0.55
                    ),
                    "JPEG",
                ),
            ):
                path = attempt / relative
                _write_image_atomic(path, image_value, format_name=image_format)
                outputs[label] = relative
                artifacts[relative] = _sha256_file(path)

            label_path = row.get("label_path")
            truth = (
                load_labelme_mask(
                    str(label_path),
                    image_shape=raw.shape,
                    expected_sha256=str(row["label_sha256"]),
                )
                if label_path
                else None
            )
            registrations = (
                [
                    {
                        "accepted": item.accepted,
                        "reason": item.reason,
                        "method": item.method,
                        "response": item.response,
                        "shift_xy": list(item.shift_xy),
                        "overlap_fraction": item.overlap_fraction,
                        "feature_inliers": item.feature_inliers,
                        "feature_inlier_ratio": item.feature_inlier_ratio,
                        "scale": item.scale,
                    }
                    for item in aggregation.registrations
                ]
                if aggregation is not None
                else []
            )
            result_rows.append(
                {
                    "id": identifier,
                    "cohort": cohort,
                    "source_image_sha256": row["image_sha256"],
                    "metadata": row.get("metadata", {}),
                    "query_plan_status": query_plan["status"],
                    "selected_reference_ids": selected_ids,
                    "prior_usable": prior_usable,
                    "registration": registrations,
                    "peak_pixels_compact": int(
                        scores.compact_preserve_mask.sum()
                    ),
                    "peak_pixels_native": int(scores.native_preserve_mask.sum()),
                    "preserve_pixels_native": int(
                        np.count_nonzero(candidate_u8 > baseline_u8)
                    ),
                    "metrics": {
                        "a": score_map_metrics_strict(
                            raw, threshold=threshold, ground_truth=truth
                        ),
                        "b": score_map_metrics_strict(
                            baseline_delivered,
                            threshold=threshold,
                            ground_truth=truth,
                        ),
                        "p": score_map_metrics_strict(
                            candidate_delivered,
                            threshold=threshold,
                            ground_truth=truth,
                        ),
                    },
                    "outputs": outputs,
                }
            )
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(source_rows),
                        "id": identifier,
                        "prior_usable": prior_usable,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        results_path = attempt / "results.jsonl"
        _write_jsonl(results_path, result_rows)
        artifacts["results.jsonl"] = _sha256_file(results_path)
        summary = summarize_peak_results(result_rows)
        final_code_digests = _runtime_code_digests()
        if final_code_digests != initial_code_digests:
            raise RuntimeError("runtime code changed during export")
        receipt = {
            "status": "completed",
            "stage": "normal_reference_relative_peak_v2_spike",
            "interpretation": "development_diagnostic_not_sealed_test",
            "candidate_status": (
                "adoption_gate_passed"
                if bool(dict(summary["adoption_gate"])["passed"])
                else "adoption_gate_rejected"
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "counts": {
                "total": len(result_rows),
                "ok": sum(row["cohort"] == "ok" for row in result_rows),
                "ng": sum(row["cohort"] == "ng" for row in result_rows),
                "baseline_maps": len(result_rows),
                "candidate_maps": len(result_rows),
                "candidate_comparisons": len(result_rows),
                "artifacts": len(artifacts),
            },
            "settings": {
                **settings,
                "compact_shape": list(compact_shape),
                "a": "sealed_raw_score_map_8bit",
                "b": "A*(1-0.5*binary_2_of_3_normal_recurrence)",
                "p": "B_except_query_regional_maxima_strictly_above_local_reference_ceiling",
                "registration": "sobel_phase_then_sift_similarity_ransac",
                "max_translation_compact_px": 8,
                "min_phase_response": 0.15,
                "min_overlap_fraction": 0.95,
                "sift_max_features": 800,
                "sift_contrast_threshold": 0.01,
                "feature_ratio_test": 0.72,
                "min_feature_good_matches": 8,
                "min_feature_inliers": 6,
                "min_feature_inlier_ratio": 0.25,
                "feature_scale_range": [0.8, 1.2],
                "min_feature_overlap_fraction": 0.6,
                "ransac_reprojection_threshold": 3.0,
                "ransac_max_iters": 5000,
                "ransac_confidence": 0.999,
                "overlay_alpha": 0.55,
                "decision_comparator": "strictly_greater_than",
                "visualization_comparator": "strictly_greater_than",
            },
            "inference_contract": inference_contract,
            "digests": {
                "candidate_spec_sha256": expected_candidate_spec_sha256,
                "selected_plan_sha256": expected_plan_digest,
                "source_receipt_sha256": expected_source_digest,
                "reference_shard_receipt_sha256": expected_reference_receipts,
                "code": initial_code_digests,
            },
            "summary": summary,
            "artifacts": artifacts,
            "execution_failures": 0,
            "failures": 0,
        }
        _write_json(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen reference-relative local-peak preservation spike"
    )
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--expected-candidate-spec-sha256", required=True)
    parser.add_argument("--selected-plan", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--reference-shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = export_peak_comparisons(
        candidate_spec_path=args.candidate_spec,
        expected_candidate_spec_sha256=args.expected_candidate_spec_sha256,
        selected_plan_path=args.selected_plan,
        source_run=args.source_run,
        reference_shard_dirs=args.reference_shard_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
