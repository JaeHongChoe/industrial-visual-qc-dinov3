from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from .actual_multitask_eval import (
    PINNED_BASE_CHECKPOINT_SHA256,
    PINNED_MULTITASK_CHECKPOINT_SHA256,
    PINNED_TRAINING_RECEIPT_SHA256,
    _write_image_atomic,
    validate_frozen_settings,
    verify_training_receipt,
)
from .actual_multitask_inference import (
    _sha256_file,
    load_labelme_mask,
    load_multitask_checkpoint,
    score_map_metrics,
)
from .capture import parse_standard_capture
from .model import DinoMaskMultiTaskProbe, load_timm_probe
from .normal_prior_plan import build_reference_plan, write_reference_plan
from .normal_prior_postprocess import (
    aggregate_binary_recurrence_prior,
    apply_case_support_boost,
    apply_recurrence_suppression,
    render_original_overlay_comparison,
)
from .normal_prior_reference import ReferenceImage, evaluate_reference_rows
from .normal_prior_select import select_visual_references
from .refinement_contract import canonical_digest
from .train_multitask import _load_trusted_epoch3_head, _verify_sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _load_frozen_model(
    *,
    config_path: str | Path,
    base_checkpoint_path: str | Path,
    multitask_checkpoint_path: str | Path,
    training_receipt_path: str | Path,
) -> tuple[torch.nn.Module, dict[str, int | float], dict[str, object]]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a mapping")
    model_config = dict(config["model"])
    inputs_config = dict(config["inputs"])
    backbone_id = str(model_config["backbone_id"])
    expected_base = str(inputs_config["expected_base_checkpoint_sha256"])
    if expected_base != PINNED_BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint is not the pinned epoch-3 model")
    _verify_sha256(base_checkpoint_path, expected_base)
    verify_training_receipt(
        training_receipt_path,
        expected_receipt_sha256=PINNED_TRAINING_RECEIPT_SHA256,
        expected_multitask_checkpoint_sha256=PINNED_MULTITASK_CHECKPOINT_SHA256,
        expected_base_checkpoint_sha256=expected_base,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("reference inference requires exactly one visible CUDA GPU")
    torch.set_float32_matmul_precision("high")
    base = load_timm_probe(
        backbone_id,
        local_files_only=bool(model_config.get("local_files_only", True)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    _load_trusted_epoch3_head(
        base,
        base_checkpoint_path,
        expected_base,
        expected_backbone_id=backbone_id,
    )
    model = DinoMaskMultiTaskProbe(
        base, decoder_channels=int(model_config.get("decoder_channels", 32))
    )
    metadata = load_multitask_checkpoint(
        model,
        multitask_checkpoint_path,
        expected_checkpoint_sha256=PINNED_MULTITASK_CHECKPOINT_SHA256,
        expected_base_checkpoint_sha256=expected_base,
        expected_backbone_id=backbone_id,
        expected_config_digest=canonical_digest(config),
    )
    settings = validate_frozen_settings(config, metadata)
    model.to(torch.device("cuda:0")).eval()
    return model, settings, metadata


def build_and_select_plan(
    *,
    ok_manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    max_candidates: int = 12,
    thumbnail_width: int = 1024,
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    attempt.mkdir(parents=True)
    try:
        plan = build_reference_plan(ok_manifest_path, predictions_path, max_candidates=max_candidates)
        write_reference_plan(plan, attempt / "candidate_plan.json")
        selected = select_visual_references(plan, required_references=3, thumbnail_width=thumbnail_width)
        selected["digests"] = {
            "ok_manifest_sha256": _sha256_file(Path(ok_manifest_path)),
            "predictions_sha256": _sha256_file(Path(predictions_path)),
            "candidate_plan_sha256": _sha256_file(attempt / "candidate_plan.json"),
        }
        _write_json(attempt / "selected_plan.json", selected)
        receipt = {
            "status": "completed",
            "stage": "strict_group_safe_reference_plan",
            "created_at": _utc_now(),
            "candidate_plan_sha256": _sha256_file(attempt / "candidate_plan.json"),
            "selected_plan_sha256": _sha256_file(attempt / "selected_plan.json"),
            "candidate_audit": plan["audit"],
            "selected_audit": selected["audit"],
            "failures": 0,
        }
        _write_json(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def reselect_existing_plan(
    *,
    candidate_plan_path: str | Path,
    expected_candidate_plan_sha256: str,
    output_dir: str | Path,
    thumbnail_width: int = 512,
) -> dict[str, object]:
    source = Path(candidate_plan_path)
    if _sha256_file(source) != expected_candidate_plan_sha256:
        raise ValueError("candidate plan SHA-256 mismatch")
    plan = _load_json(source)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    attempt.mkdir(parents=True)
    try:
        selected = select_visual_references(
            plan, required_references=3, thumbnail_width=thumbnail_width
        )
        selected["digests"] = {
            "candidate_plan_sha256": expected_candidate_plan_sha256,
        }
        _write_json(attempt / "selected_plan.json", selected)
        receipt = {
            "status": "completed",
            "stage": "strict_group_safe_feature_reference_selection",
            "created_at": _utc_now(),
            "candidate_plan_sha256": expected_candidate_plan_sha256,
            "selected_plan_sha256": _sha256_file(attempt / "selected_plan.json"),
            "thumbnail_width": thumbnail_width,
            "selected_audit": selected["audit"],
            "failures": 0,
        }
        _write_json(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def infer_reference_shard(
    *,
    selected_plan_path: str | Path,
    expected_selected_plan_sha256: str,
    output_dir: str | Path,
    shard_index: int,
    shard_count: int,
    config_path: str | Path,
    base_checkpoint_path: str | Path,
    multitask_checkpoint_path: str | Path,
    training_receipt_path: str | Path,
    stride: int,
    batch_size: int,
    map_width: int = 1024,
) -> dict[str, object]:
    source_plan = Path(selected_plan_path)
    if _sha256_file(source_plan) != expected_selected_plan_sha256:
        raise ValueError("selected plan SHA-256 mismatch")
    plan = _load_json(source_plan)
    references = plan.get("references")
    if not isinstance(references, list) or not 0 <= shard_index < shard_count:
        raise ValueError("invalid reference plan or shard")
    shard = [row for index, row in enumerate(references) if index % shard_count == shard_index]
    if not shard:
        raise ValueError("reference shard is empty")
    rows = [
        ReferenceImage(
            identifier=str(row["reference_id"]),
            image_path=Path(str(row["image_path"])),
            image_sha256=str(row["image_sha256"]),
            product=str(row["product"]),
            condition_id=str(row["condition_id"]),
            capture_id=str(row["capture_id"]),
            group_id=str(row["group_id"]),
        )
        for row in shard
    ]
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    started = time.perf_counter()
    model, settings, metadata = _load_frozen_model(
        config_path=config_path,
        base_checkpoint_path=base_checkpoint_path,
        multitask_checkpoint_path=multitask_checkpoint_path,
        training_receipt_path=training_receipt_path,
    )
    records, artifacts = evaluate_reference_rows(
        rows,
        model,
        output_dir=attempt,
        patch_size=int(settings["patch_size"]),
        stride=stride,
        batch_size=batch_size,
        device="cuda:0",
        map_width=map_width,
        use_amp=True,
    )
    records_path = attempt / "references.jsonl"
    _write_jsonl(records_path, records)
    artifacts["references.jsonl"] = _sha256_file(records_path)
    receipt = {
        "status": "completed",
        "stage": "normal_reference_raw_inference",
        "started_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {"references": len(records), "artifacts": len(artifacts)},
        "shard": {"index": shard_index, "count": shard_count},
        "settings": {
            "patch_size": int(settings["patch_size"]),
            "stride": stride,
            "batch_size": batch_size,
            "map_width": map_width,
            "mask_threshold": 0.5,
        },
        "digests": {
            "selected_plan_sha256": expected_selected_plan_sha256,
            "base_checkpoint_sha256": PINNED_BASE_CHECKPOINT_SHA256,
            "multitask_checkpoint_sha256": PINNED_MULTITASK_CHECKPOINT_SHA256,
            "training_receipt_sha256": PINNED_TRAINING_RECEIPT_SHA256,
        },
        "checkpoint_metadata": metadata,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "artifacts": artifacts,
        "failures": 0,
    }
    _write_json(attempt / "run_receipt.json", receipt)
    attempt.replace(destination)
    return receipt


def _tail_mean(score_map: np.ndarray, fraction: float = 0.0001) -> float:
    values = np.asarray(score_map, dtype=np.float32).ravel()
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def _physical_view(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    condition = parse_standard_capture(Path(str(row["image_path"])).name)
    if condition is None or condition.view_index is None:
        return None
    capture = metadata.get("capture_id")
    if not isinstance(capture, str):
        relative = metadata.get("relative_source_path")
        if not isinstance(relative, str):
            return None
        parts = Path(relative).parts
        if len(parts) < 2:
            return None
        lot = re.split(r"[\s(]", parts[1], maxsplit=1)[0]
        capture = f"{parts[0]}/{lot}"
    return capture, condition.family, condition.view_index


def _load_compact_gray(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return gray


def export_comparisons(
    *,
    selected_plan_path: str | Path,
    expected_selected_plan_sha256: str,
    source_run: str | Path,
    expected_source_receipt_sha256: str,
    reference_shard_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    selected_plan_path = Path(selected_plan_path)
    if _sha256_file(selected_plan_path) != expected_selected_plan_sha256:
        raise ValueError("selected plan SHA-256 mismatch")
    plan = _load_json(selected_plan_path)
    source = Path(source_run)
    source_receipt_path = source / "run_receipt.json"
    if _sha256_file(source_receipt_path) != expected_source_receipt_sha256:
        raise ValueError("source receipt SHA-256 mismatch")
    source_receipt = _load_json(source_receipt_path)
    source_artifacts = source_receipt.get("artifacts")
    if source_receipt.get("status") != "completed" or not isinstance(source_artifacts, Mapping):
        raise ValueError("source run is not sealed and completed")
    predictions_path = source / "predictions.jsonl"
    if _sha256_file(predictions_path) != source_artifacts.get("predictions.jsonl"):
        raise ValueError("source predictions SHA-256 mismatch")
    source_rows = _load_jsonl(predictions_path)
    query_by_id = {str(row["id"]): row for row in plan["queries"]}
    if set(query_by_id) != {str(row["id"]) for row in source_rows}:
        raise ValueError("selected plan and source predictions query IDs differ")

    reference_records: dict[str, tuple[dict[str, Any], Path]] = {}
    reference_tail_scores: list[float] = []
    for shard_value in reference_shard_dirs:
        shard = Path(shard_value)
        receipt = _load_json(shard / "run_receipt.json")
        if receipt.get("status") != "completed":
            raise ValueError(f"reference shard is incomplete: {shard}")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("reference shard artifacts missing")
        records_path = shard / "references.jsonl"
        if _sha256_file(records_path) != artifacts.get("references.jsonl"):
            raise ValueError("reference records SHA-256 mismatch")
        for record in _load_jsonl(records_path):
            reference_id = str(record["id"])
            if reference_id in reference_records:
                raise ValueError("duplicate reference inference record")
            outputs = record["outputs"]
            for relative in outputs.values():
                if _sha256_file(shard / str(relative)) != artifacts.get(str(relative)):
                    raise ValueError(f"reference artifact SHA-256 mismatch: {relative}")
            reference_records[reference_id] = (record, shard)
            with Image.open(shard / str(outputs["raw_map"])) as image:
                compact = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            reference_tail_scores.append(_tail_mean(compact))
    expected_reference_ids = {str(row["reference_id"]) for row in plan["references"]}
    if set(reference_records) != expected_reference_ids:
        raise ValueError("reference inference coverage differs from selected plan")
    support_threshold = float(np.quantile(reference_tail_scores, 0.99))

    raw_tail_by_id: dict[str, float] = {}
    physical_by_id: dict[str, tuple[str, str, int] | None] = {}
    for row in source_rows:
        raw_relative = str(row["outputs"]["raw_heatmap"])
        raw_path = source / raw_relative
        if _sha256_file(raw_path) != source_artifacts.get(raw_relative):
            raise ValueError(f"source raw map SHA-256 mismatch: {row['id']}")
        with Image.open(raw_path) as image:
            raw_tail_by_id[str(row["id"])] = _tail_mean(
                np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            )
        physical_by_id[str(row["id"])] = _physical_view(row)
    support_by_id: dict[str, float] = {}
    for row in source_rows:
        identifier = str(row["id"])
        physical = physical_by_id[identifier]
        support = 0.0
        if physical is not None:
            capture, family, view_index = physical
            for neighbor_id, neighbor in physical_by_id.items():
                if neighbor_id == identifier or neighbor is None:
                    continue
                if (
                    neighbor[0] == capture
                    and neighbor[1] == family
                    and abs(neighbor[2] - view_index) == 1
                    and raw_tail_by_id[neighbor_id] > support_threshold
                ):
                    support = 1.0
                    break
        support_by_id[identifier] = support

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    attempt.mkdir(parents=True)
    started = time.perf_counter()
    artifacts: dict[str, str] = {}
    result_rows: list[dict[str, object]] = []
    try:
        for index, row in enumerate(source_rows, start=1):
            identifier = str(row["id"])
            cohort = str(row["cohort"])
            query_plan = query_by_id[identifier]
            raw_relative = str(row["outputs"]["raw_heatmap"])
            with Image.open(source / raw_relative) as image:
                raw = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            image_bytes = Path(str(row["image_path"])).read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
                raise ValueError(f"query source image SHA-256 mismatch: {identifier}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            compact_shape = next(iter(reference_records.values()))[0]["map_shape"]
            shape = (int(compact_shape[0]), int(compact_shape[1]))
            query_thumbnail = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            selected_ids = list(query_plan.get("selected_reference_ids", []))
            reference_thumbnails: list[np.ndarray] = []
            reference_maps: list[np.ndarray] = []
            for reference_id in selected_ids:
                record, shard = reference_records[str(reference_id)]
                reference_thumbnails.append(
                    _load_compact_gray(shard / str(record["outputs"]["preview"]), shape)
                )
                reference_maps.append(
                    _load_compact_gray(shard / str(record["outputs"]["raw_map"]), shape)
                )
            if len(selected_ids) == 3:
                aggregation = aggregate_binary_recurrence_prior(
                    query_thumbnail,
                    reference_thumbnails,
                    reference_maps,
                    score_threshold=0.5,
                    dilation_radius_compact_px=1,
                )
            else:
                aggregation = None
            compact_prior = (
                aggregation.prior
                if aggregation is not None and aggregation.usable
                else np.zeros(shape, dtype=np.float32)
            )
            full_prior = cv2.resize(
                compact_prior, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            b_score = apply_recurrence_suppression(raw, full_prior)
            c_score = apply_case_support_boost(
                raw, b_score, support=support_by_id[identifier]
            )
            outputs: dict[str, str] = {}
            for name, score in (("b", b_score), ("c", c_score)):
                map_path = attempt / f"{name}_maps" / cohort / f"{identifier}.png"
                comparison_path = attempt / f"{name}_comparisons" / cohort / f"{identifier}.jpg"
                _write_image_atomic(
                    map_path,
                    np.rint(np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8),
                    format_name="PNG",
                )
                _write_image_atomic(
                    comparison_path,
                    render_original_overlay_comparison(rgb, score, threshold=0.5),
                    format_name="JPEG",
                )
                for label, path in (
                    (f"{name}_map", map_path),
                    (f"{name}_comparison", comparison_path),
                ):
                    relative = path.relative_to(attempt).as_posix()
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
            result_rows.append(
                {
                    "id": identifier,
                    "cohort": cohort,
                    "source_image_sha256": row["image_sha256"],
                    "query_plan_status": query_plan["status"],
                    "selected_reference_ids": selected_ids,
                    "prior_usable": bool(aggregation is not None and aggregation.usable),
                    "registration": [
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
                    else [],
                    "support": support_by_id[identifier],
                    "metrics": {
                        "a": score_map_metrics(raw, threshold=0.5, ground_truth=truth),
                        "b": score_map_metrics(b_score, threshold=0.5, ground_truth=truth),
                        "c": score_map_metrics(c_score, threshold=0.5, ground_truth=truth),
                    },
                    "outputs": outputs,
                }
            )
            print(
                json.dumps(
                    {"completed": index, "total": len(source_rows), "id": identifier},
                    sort_keys=True,
                ),
                flush=True,
            )
        results_path = attempt / "results.jsonl"
        _write_jsonl(results_path, result_rows)
        artifacts["results.jsonl"] = _sha256_file(results_path)
        localized = [row for row in result_rows if "ground_truth_hit" in row["metrics"]["a"]]
        ok_rows = [row for row in result_rows if row["cohort"] == "ok"]
        summary = {
            "localized_hits": {
                key: sum(bool(row["metrics"][key]["ground_truth_hit"]) for row in localized)
                for key in ("a", "b", "c")
            },
            "ok_images_with_activation": {
                key: sum(int(row["metrics"][key]["predicted_positive_pixels"]) > 0 for row in ok_rows)
                for key in ("a", "b", "c")
            },
            "ok_predicted_positive_pixels": {
                key: sum(int(row["metrics"][key]["predicted_positive_pixels"]) for row in ok_rows)
                for key in ("a", "b", "c")
            },
            "prior_usable_queries": sum(bool(row["prior_usable"]) for row in result_rows),
            "case_support_queries": sum(float(row["support"]) > 0 for row in result_rows),
        }
        receipt = {
            "status": "completed",
            "stage": "normal_recurrence_abc_spike",
            "started_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "counts": {
                "total": len(result_rows),
                "ok": sum(row["cohort"] == "ok" for row in result_rows),
                "ng": sum(row["cohort"] == "ng" for row in result_rows),
                "comparisons": 2 * len(result_rows),
                "artifacts": len(artifacts),
            },
            "settings": {
                "a": "sealed_raw_score_map_8bit",
                "b": "A*(1-0.5*binary_2_of_3_normal_recurrence)",
                "c": "B+0.25*case_support*(A-B)",
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
                "display_threshold": 0.5,
                "overlay_alpha": 0.55,
                "support_tail_fraction": 0.0001,
                "support_reference_p99": support_threshold,
            },
            "digests": {
                "selected_plan_sha256": expected_selected_plan_sha256,
                "source_receipt_sha256": expected_source_receipt_sha256,
            },
            "summary": summary,
            "artifacts": artifacts,
            "failures": 0,
        }
        _write_json(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe normal recurrence A/B/C spike")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--ok-manifest", required=True)
    plan.add_argument("--predictions", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--max-candidates", type=int, default=12)
    plan.add_argument("--thumbnail-width", type=int, default=1024)
    reselect = sub.add_parser("reselect")
    reselect.add_argument("--candidate-plan", required=True)
    reselect.add_argument("--expected-candidate-plan-sha256", required=True)
    reselect.add_argument("--output-dir", required=True)
    reselect.add_argument("--thumbnail-width", type=int, default=512)
    infer = sub.add_parser("infer")
    infer.add_argument("--selected-plan", required=True)
    infer.add_argument("--expected-selected-plan-sha256", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--shard-index", type=int, required=True)
    infer.add_argument("--shard-count", type=int, required=True)
    infer.add_argument("--config", required=True)
    infer.add_argument("--base-checkpoint", required=True)
    infer.add_argument("--multitask-checkpoint", required=True)
    infer.add_argument("--training-receipt", required=True)
    infer.add_argument("--stride", type=int, default=128)
    infer.add_argument("--batch-size", type=int, default=128)
    infer.add_argument("--map-width", type=int, default=1024)
    export = sub.add_parser("export")
    export.add_argument("--selected-plan", required=True)
    export.add_argument("--expected-selected-plan-sha256", required=True)
    export.add_argument("--source-run", required=True)
    export.add_argument("--expected-source-receipt-sha256", required=True)
    export.add_argument("--reference-shard-dir", action="append", required=True)
    export.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        receipt = build_and_select_plan(
            ok_manifest_path=args.ok_manifest,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            max_candidates=args.max_candidates,
            thumbnail_width=args.thumbnail_width,
        )
    elif args.command == "reselect":
        receipt = reselect_existing_plan(
            candidate_plan_path=args.candidate_plan,
            expected_candidate_plan_sha256=args.expected_candidate_plan_sha256,
            output_dir=args.output_dir,
            thumbnail_width=args.thumbnail_width,
        )
    elif args.command == "infer":
        receipt = infer_reference_shard(
            selected_plan_path=args.selected_plan,
            expected_selected_plan_sha256=args.expected_selected_plan_sha256,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            config_path=args.config,
            base_checkpoint_path=args.base_checkpoint,
            multitask_checkpoint_path=args.multitask_checkpoint,
            training_receipt_path=args.training_receipt,
            stride=args.stride,
            batch_size=args.batch_size,
            map_width=args.map_width,
        )
    else:
        receipt = export_comparisons(
            selected_plan_path=args.selected_plan,
            expected_selected_plan_sha256=args.expected_selected_plan_sha256,
            source_run=args.source_run,
            expected_source_receipt_sha256=args.expected_source_receipt_sha256,
            reference_shard_dirs=args.reference_shard_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
