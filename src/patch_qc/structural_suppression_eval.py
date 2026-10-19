from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import yaml

from .actual_multitask_eval import (
    PINNED_BASE_CHECKPOINT_SHA256,
    PINNED_MULTITASK_CHECKPOINT_SHA256,
    PINNED_TRAINING_RECEIPT_SHA256,
    _utc_now,
    _write_image_atomic,
    _write_json_atomic,
    _write_jsonl_atomic,
    validate_frozen_settings,
    verify_training_receipt,
)
from .actual_multitask_inference import (
    _sha256_file,
    load_labelme_mask,
    load_multitask_checkpoint,
    render_mask_overlay,
)
from .model import DinoMaskMultiTaskProbe, load_timm_probe
from .normal_peak_spike import (
    quantize_delivered_score_map,
    score_map_metrics_strict,
)
from .refinement_contract import canonical_digest
from .structural_suppression_spike import (
    predict_auxiliary_max_image,
    summarize_auxiliary_max_results,
    validate_structural_candidate_spec,
)
from .train_multitask import _load_trusted_epoch3_head, _verify_sha256


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required on line {line_number}: {path}")
            rows.append(value)
    return rows


def _runtime_code_digests() -> dict[str, str]:
    package = Path(__file__).parent
    names = (
        "structural_suppression_eval.py",
        "structural_suppression_spike.py",
        "actual_multitask_eval.py",
        "actual_multitask_inference.py",
        "inference.py",
        "model.py",
        "normal_peak_spike.py",
        "refinement_contract.py",
        "train_multitask.py",
    )
    return {name: _sha256_file(package / name) for name in names}


def run(
    *,
    candidate_spec_path: str | Path,
    expected_candidate_spec_sha256: str,
    config_path: str | Path,
    base_checkpoint_path: str | Path,
    multitask_checkpoint_path: str | Path,
    training_receipt_path: str | Path,
    source_run: str | Path,
    expected_source_receipt_sha256: str,
    output_dir: str | Path,
    batch_size: int,
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    if attempt.exists():
        raise FileExistsError(attempt)

    spec_path = Path(candidate_spec_path)
    if _sha256_file(spec_path) != expected_candidate_spec_sha256:
        raise ValueError("candidate spec SHA-256 mismatch")
    spec = _load_json(spec_path)
    candidate_settings = validate_structural_candidate_spec(spec)
    if (
        expected_source_receipt_sha256
        != candidate_settings["source_receipt_sha256"]
    ):
        raise ValueError("source receipt CLI value differs from candidate spec")
    if candidate_settings["base_checkpoint_sha256"] != PINNED_BASE_CHECKPOINT_SHA256:
        raise ValueError("candidate spec base checkpoint is not pinned")
    if (
        candidate_settings["multitask_checkpoint_sha256"]
        != PINNED_MULTITASK_CHECKPOINT_SHA256
    ):
        raise ValueError("candidate spec multitask checkpoint is not pinned")
    if (
        candidate_settings["training_receipt_sha256"]
        != PINNED_TRAINING_RECEIPT_SHA256
    ):
        raise ValueError("candidate spec training receipt is not pinned")
    expected_code = {
        str(key): str(value)
        for key, value in dict(spec.get("runtime_code_sha256", {})).items()
    }
    initial_code = _runtime_code_digests()
    if expected_code != initial_code:
        raise ValueError("runtime code differs from the frozen candidate spec")

    source = Path(source_run)
    source_receipt_path = source / "run_receipt.json"
    if _sha256_file(source_receipt_path) != expected_source_receipt_sha256:
        raise ValueError("source receipt SHA-256 mismatch")
    source_receipt = _load_json(source_receipt_path)
    source_artifacts = source_receipt.get("artifacts")
    if source_receipt.get("status") != "completed" or not isinstance(
        source_artifacts, Mapping
    ):
        raise ValueError("source run must be complete and sealed")
    source_predictions_path = source / "predictions.jsonl"
    if _sha256_file(source_predictions_path) != source_artifacts.get(
        "predictions.jsonl"
    ):
        raise ValueError("source predictions SHA-256 mismatch")
    source_rows = _load_jsonl(source_predictions_path)
    if len(source_rows) != 115:
        raise ValueError("source evaluation must contain exactly 115 rows")
    if sum(row.get("cohort") == "ok" for row in source_rows) != 27 or sum(
        row.get("cohort") == "ng" for row in source_rows
    ) != 88:
        raise ValueError("source evaluation cohort counts differ from the contract")

    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a mapping")
    if hashlib.sha256(config_bytes).hexdigest() != dict(
        source_receipt["digests"]
    )["config_sha256"]:
        raise ValueError("configuration differs from the sealed source run")
    model_config = dict(config["model"])
    inputs_config = dict(config["inputs"])
    backbone_id = str(model_config["backbone_id"])
    expected_base_sha256 = str(inputs_config["expected_base_checkpoint_sha256"])
    if expected_base_sha256 != PINNED_BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint is not the pinned epoch-3 checkpoint")
    _verify_sha256(base_checkpoint_path, expected_base_sha256)
    training_receipt = verify_training_receipt(
        training_receipt_path,
        expected_receipt_sha256=PINNED_TRAINING_RECEIPT_SHA256,
        expected_multitask_checkpoint_sha256=PINNED_MULTITASK_CHECKPOINT_SHA256,
        expected_base_checkpoint_sha256=PINNED_BASE_CHECKPOINT_SHA256,
    )
    if batch_size != candidate_settings["batch_size"]:
        raise ValueError("batch size CLI value differs from candidate spec")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("candidate evaluation requires exactly one visible CUDA GPU")

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    base = load_timm_probe(
        backbone_id,
        local_files_only=bool(model_config.get("local_files_only", True)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    _load_trusted_epoch3_head(
        base,
        base_checkpoint_path,
        expected_base_sha256,
        expected_backbone_id=backbone_id,
    )
    model = DinoMaskMultiTaskProbe(
        base, decoder_channels=int(model_config.get("decoder_channels", 32))
    )
    checkpoint_metadata = load_multitask_checkpoint(
        model,
        multitask_checkpoint_path,
        expected_checkpoint_sha256=PINNED_MULTITASK_CHECKPOINT_SHA256,
        expected_base_checkpoint_sha256=PINNED_BASE_CHECKPOINT_SHA256,
        expected_backbone_id=backbone_id,
        expected_config_digest=canonical_digest(config),
    )
    frozen_settings = validate_frozen_settings(config, checkpoint_metadata)
    if int(frozen_settings["patch_size"]) != candidate_settings["patch_size"]:
        raise ValueError("runtime patch size differs from candidate spec")
    if int(frozen_settings["stride"]) != candidate_settings["stride"]:
        raise ValueError("runtime stride differs from candidate spec")
    model.to(device).eval()
    threshold = float(candidate_settings["decision_threshold"])

    started_at = _utc_now()
    started = time.perf_counter()
    artifacts: dict[str, str] = {}
    result_rows: list[dict[str, object]] = []
    raw_replay_mismatch_pixels = 0
    try:
        attempt.mkdir(parents=True)
        frozen_spec = attempt / "candidate_spec.json"
        shutil.copyfile(spec_path, frozen_spec)
        artifacts["candidate_spec.json"] = _sha256_file(frozen_spec)

        for index, row in enumerate(source_rows, start=1):
            identifier = str(row["id"])
            cohort = str(row["cohort"])
            image_path = Path(str(row["image_path"]))
            image_bytes = image_path.read_bytes()
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            if image_sha256 != row["image_sha256"]:
                raise RuntimeError(f"source image changed: {identifier}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

            prediction = predict_auxiliary_max_image(
                model,
                rgb,
                patch_size=int(frozen_settings["patch_size"]),
                stride=int(frozen_settings["stride"]),
                batch_size=batch_size,
                device=device,
                use_amp=bool(candidate_settings["use_amp"]),
            )
            replay_u8, _ = quantize_delivered_score_map(prediction.raw_score_map)
            candidate_u8, candidate_delivered = quantize_delivered_score_map(
                prediction.auxiliary_max_score_map
            )

            source_raw_relative = str(dict(row["outputs"])["raw_heatmap"])
            source_raw_path = source / source_raw_relative
            if _sha256_file(source_raw_path) != source_artifacts.get(
                source_raw_relative
            ):
                raise ValueError(f"source raw-map artifact mismatch: {identifier}")
            with Image.open(source_raw_path) as image:
                source_raw_u8 = np.asarray(image.convert("L"), dtype=np.uint8)
            if source_raw_u8.shape != rgb.shape[:2]:
                raise ValueError(f"source raw-map shape mismatch: {identifier}")
            mismatch = int(np.count_nonzero(replay_u8 != source_raw_u8))
            raw_replay_mismatch_pixels += mismatch
            if mismatch:
                raise RuntimeError(f"raw replay differs from sealed source: {identifier}")
            source_raw = source_raw_u8.astype(np.float32) / 255.0

            label_path_value = row.get("label_path")
            ground_truth = (
                load_labelme_mask(
                    str(label_path_value),
                    image_shape=rgb.shape[:2],
                    expected_sha256=str(row["label_sha256"]),
                )
                if label_path_value
                else None
            )
            raw_metrics = score_map_metrics_strict(
                source_raw, threshold=threshold, ground_truth=ground_truth
            )
            candidate_metrics = score_map_metrics_strict(
                candidate_delivered,
                threshold=threshold,
                ground_truth=ground_truth,
            )
            comparison = np.concatenate(
                (
                    rgb,
                    render_mask_overlay(
                        rgb,
                        candidate_delivered,
                        threshold=threshold,
                        alpha=0.55,
                    ),
                ),
                axis=1,
            )
            candidate_relative = (
                f"candidate_heatmaps/{cohort}/{identifier}.png"
            )
            comparison_relative = f"comparisons/{cohort}/{identifier}.jpg"
            candidate_path = attempt / candidate_relative
            comparison_path = attempt / comparison_relative
            _write_image_atomic(
                candidate_path, candidate_u8, format_name="PNG"
            )
            _write_image_atomic(
                comparison_path, comparison, format_name="JPEG"
            )
            artifacts[candidate_relative] = _sha256_file(candidate_path)
            artifacts[comparison_relative] = _sha256_file(comparison_path)
            result_rows.append(
                {
                    "id": identifier,
                    "cohort": cohort,
                    "image_path": str(image_path),
                    "image_sha256": image_sha256,
                    "label_path": label_path_value,
                    "label_sha256": row.get("label_sha256"),
                    "label_status": row["label_status"],
                    "metadata": row.get("metadata", {}),
                    "patch_count": prediction.patch_count,
                    "auxiliary_probability_max": float(
                        prediction.auxiliary_probabilities.max()
                    ),
                    "raw_replay_mismatch_pixels": mismatch,
                    "raw_metrics": raw_metrics,
                    "candidate_metrics": candidate_metrics,
                    "outputs": {
                        "candidate_heatmap": candidate_relative,
                        "comparison": comparison_relative,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(source_rows),
                        "id": identifier,
                        "cohort": cohort,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        results_path = attempt / "results.jsonl"
        _write_jsonl_atomic(results_path, result_rows)
        artifacts["results.jsonl"] = _sha256_file(results_path)
        summary = summarize_auxiliary_max_results(result_rows)
        final_code = _runtime_code_digests()
        if final_code != initial_code:
            raise RuntimeError("runtime code changed during evaluation")
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        receipt: dict[str, object] = {
            "status": "completed",
            "stage": "structural_auxiliary_weighted_max_v1_spike",
            "interpretation": "development_diagnostic_not_sealed_test",
            "candidate_status": (
                "screening_gate_passed"
                if bool(dict(summary["screening_gate"])["passed"])
                else "screening_gate_rejected"
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "counts": {
                "total": len(result_rows),
                "ok": sum(row["cohort"] == "ok" for row in result_rows),
                "ng": sum(row["cohort"] == "ng" for row in result_rows),
                "candidate_heatmaps": len(result_rows),
                "comparisons": len(result_rows),
                "artifacts": len(artifacts),
            },
            "settings": {
                "patch_size": int(frozen_settings["patch_size"]),
                "stride": int(frozen_settings["stride"]),
                "batch_size": batch_size,
                "decision_threshold": threshold,
                "decision_comparator": "strictly_greater_than",
                "visualization_comparator": "strictly_greater_than",
                "formula": spec["formula"],
                "overlay_alpha": 0.55,
            },
            "digests": {
                "candidate_spec_sha256": expected_candidate_spec_sha256,
                "source_receipt_sha256": expected_source_receipt_sha256,
                "source_predictions_sha256": source_artifacts["predictions.jsonl"],
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "base_checkpoint_sha256": PINNED_BASE_CHECKPOINT_SHA256,
                "multitask_checkpoint_sha256": PINNED_MULTITASK_CHECKPOINT_SHA256,
                "training_receipt_sha256": PINNED_TRAINING_RECEIPT_SHA256,
                "code": initial_code,
            },
            "checkpoint_metadata": checkpoint_metadata,
            "training_receipt": {
                "status": training_receipt["status"],
                "quality_gate": training_receipt["quality_gate"],
            },
            "code": {"commit": commit, "dirty": True},
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")
                ),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
            },
            "raw_replay_mismatch_pixels": raw_replay_mismatch_pixels,
            "summary": summary,
            "artifacts": artifacts,
            "execution_failures": 0,
            "failures": 0,
        }
        _write_json_atomic(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen auxiliary-weighted max structural-suppression spike"
    )
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--expected-candidate-spec-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--multitask-checkpoint", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(
        candidate_spec_path=args.candidate_spec,
        expected_candidate_spec_sha256=args.expected_candidate_spec_sha256,
        config_path=args.config,
        base_checkpoint_path=args.base_checkpoint,
        multitask_checkpoint_path=args.multitask_checkpoint,
        training_receipt_path=args.training_receipt,
        source_run=args.source_run,
        expected_source_receipt_sha256=args.expected_source_receipt_sha256,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
