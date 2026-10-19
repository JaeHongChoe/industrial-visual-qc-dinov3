from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import yaml

from .actual_multitask_inference import (
    _sha256_file,
    load_labelme_mask,
    load_multitask_checkpoint,
    predict_multitask_image,
    render_mask_overlay,
    score_map_metrics,
)
from .model import DinoMaskMultiTaskProbe, load_timm_probe
from .refinement_contract import canonical_digest
from .train_multitask import _load_trusted_epoch3_head, _verify_sha256


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PINNED_BASE_CHECKPOINT_SHA256 = "6db93522444871c1ff2cf021e341a46bc855681465f8e71b60bb89eaf1f9c3ab"
PINNED_MULTITASK_CHECKPOINT_SHA256 = "077c95591c01cf67d6126c7c28e85b0615d5ac55d2a340d1855707d053a9cde2"
PINNED_TRAINING_RECEIPT_SHA256 = "4bc178df717a7be2d0c7cb64b52210798ed30bcf76819c670412c566a7b95f86"


@dataclass(frozen=True, slots=True)
class FrozenEvalContract:
    ok_count: int
    ng_count: int
    labeled_ng_count: int
    image_shape: tuple[int, int]
    ok_product: str
    ok_split: str
    ok_date: str
    ok_lot: str


ACTUAL_OK_NG_CONTRACT = FrozenEvalContract(
    ok_count=27,
    ng_count=88,
    labeled_ng_count=80,
    image_shape=(5464, 8192),
    ok_product="PRODUCT_B",
    ok_split="holdout",
    ok_date="2026-08-12",
    ok_lot="PRODUCT_G-G397",
)


@dataclass(frozen=True, slots=True)
class EvalImage:
    identifier: str
    cohort: str
    image_path: Path
    label_path: Path | None
    metadata: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_eval_rows(path: str | Path) -> list[EvalImage]:
    source = Path(path)
    rows: list[EvalImage] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = str(value.get("id", ""))
            if not _SAFE_IDENTIFIER.fullmatch(identifier):
                raise ValueError(f"invalid evaluation id on line {line_number}: {identifier!r}")
            if identifier in seen:
                raise ValueError(f"duplicate evaluation id: {identifier}")
            seen.add(identifier)
            cohort = str(value.get("cohort", "")).casefold()
            if cohort not in {"ok", "ng"}:
                raise ValueError(f"evaluation cohort must be ok or ng on line {line_number}")
            image_path = Path(str(value.get("image_path", "")))
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            raw_label = value.get("label_path")
            label_path = None if raw_label in {None, ""} else Path(str(raw_label))
            if label_path is not None and not label_path.is_file():
                raise FileNotFoundError(label_path)
            metadata = {
                str(key): item
                for key, item in value.items()
                if key not in {"id", "cohort", "image_path", "label_path"}
            }
            rows.append(EvalImage(identifier, cohort, image_path, label_path, metadata))
    if not rows:
        raise ValueError("evaluation manifest is empty")
    return rows


def verify_eval_contract(
    rows: Sequence[EvalImage],
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    contract: FrozenEvalContract,
) -> dict[str, object]:
    manifest_sha256 = _sha256_file(Path(manifest_path))
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "evaluation manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )
    ok_rows = [row for row in rows if row.cohort == "ok"]
    ng_rows = [row for row in rows if row.cohort == "ng"]
    labeled_ng = [row for row in ng_rows if row.label_path is not None]
    if len(ok_rows) != contract.ok_count or len(ng_rows) != contract.ng_count:
        raise ValueError(
            f"evaluation cohort count mismatch: ok={len(ok_rows)}, ng={len(ng_rows)}"
        )
    if len(labeled_ng) != contract.labeled_ng_count:
        raise ValueError(
            "evaluation labeled-NG count mismatch: "
            f"expected {contract.labeled_ng_count}, got {len(labeled_ng)}"
        )
    if any(row.label_path is not None for row in ok_rows):
        raise ValueError("frozen OK cohort must not carry NG labels")
    expected_ok_metadata = {
        "product": contract.ok_product,
        "split": contract.ok_split,
        "date": contract.ok_date,
        "lot": contract.ok_lot,
    }
    for row in ok_rows:
        for key, expected in expected_ok_metadata.items():
            if row.metadata.get(key) != expected:
                raise ValueError(
                    f"frozen OK cohort metadata mismatch for {row.identifier}: "
                    f"{key}={row.metadata.get(key)!r}, expected {expected!r}"
                )
    paths = [str(row.image_path.resolve()) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("evaluation manifest contains duplicate image paths")
    image_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    for row in rows:
        with Image.open(row.image_path) as image:
            actual_shape = (int(image.height), int(image.width))
        if actual_shape != contract.image_shape:
            raise ValueError(
                f"native image size mismatch for {row.identifier}: "
                f"expected {contract.image_shape}, got {actual_shape}"
            )
        image_hashes[row.identifier] = _sha256_file(row.image_path)
        if row.label_path is not None:
            if row.label_path.stem != row.image_path.stem:
                raise ValueError(f"LabelMe/image stem mismatch for {row.identifier}")
            label_payload = json.loads(row.label_path.read_text(encoding="utf-8"))
            label_shape = (
                int(label_payload.get("imageHeight", -1)),
                int(label_payload.get("imageWidth", -1)),
            )
            if label_shape != contract.image_shape:
                raise ValueError(
                    f"LabelMe image size mismatch for {row.identifier}: "
                    f"expected {contract.image_shape}, got {label_shape}"
                )
            label_hashes[row.identifier] = _sha256_file(row.label_path)
    if len(set(image_hashes.values())) != len(rows):
        raise ValueError("evaluation manifest contains duplicate image content")
    return {
        "manifest_sha256": manifest_sha256,
        "counts": {
            "ok": len(ok_rows),
            "ng": len(ng_rows),
            "labeled_ng": len(labeled_ng),
            "unlabeled_ng": len(ng_rows) - len(labeled_ng),
        },
        "image_shape": list(contract.image_shape),
        "unique_image_paths": len(set(paths)),
        "unique_image_sha256": len(set(image_hashes.values())),
        "image_set_sha256": hashlib.sha256(
            "\n".join(sorted(image_hashes.values())).encode("ascii")
        ).hexdigest(),
        "image_sha256_by_id": image_hashes,
        "label_sha256_by_id": label_hashes,
    }


def validate_frozen_settings(
    config: Mapping[str, object], checkpoint_metadata: Mapping[str, object]
) -> dict[str, float | int]:
    config_digest = canonical_digest(config)
    if checkpoint_metadata.get("config_digest") != config_digest:
        raise ValueError("multitask checkpoint configuration digest mismatch")
    data_config = dict(config["data"])
    evaluation_config = dict(config["evaluation"])
    inputs_config = dict(config["inputs"])
    patch_size = int(data_config["patch_size"])
    threshold = float(evaluation_config["mask_threshold"])
    if patch_size != 256:
        raise ValueError("patch_size must remain frozen at 256")
    if threshold != 0.5:
        raise ValueError("mask threshold must remain frozen at 0.5")
    if inputs_config.get("expected_base_checkpoint_sha256") != PINNED_BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint SHA-256 is not the pinned epoch-3 checkpoint")
    return {"patch_size": patch_size, "stride": 128, "mask_threshold": threshold}


def verify_training_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_multitask_checkpoint_sha256: str,
    expected_base_checkpoint_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_receipt_sha256:
        raise ValueError(
            f"training receipt SHA-256 mismatch: expected {expected_receipt_sha256}, got {actual_sha256}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    quality_gate = payload.get("quality_gate")
    if payload.get("status") != "completed" or not isinstance(quality_gate, Mapping):
        raise ValueError("training receipt is not completed")
    if quality_gate.get("passed") is not True or quality_gate.get("failures"):
        raise ValueError("training receipt quality gate did not pass")
    if dict(payload.get("artifacts", {})).get("checkpoint_epoch_001.pt") != (
        expected_multitask_checkpoint_sha256
    ):
        raise ValueError("training receipt checkpoint artifact digest mismatch")
    if dict(payload.get("digests", {})).get("base_checkpoint_sha256") != (
        expected_base_checkpoint_sha256
    ):
        raise ValueError("training receipt base checkpoint digest mismatch")
    return dict(payload)


def _write_image_atomic(path: Path, pixels: np.ndarray, *, format_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image = Image.fromarray(pixels)
        if format_name == "JPEG":
            image.save(temporary, format=format_name, quality=95, subsampling=0)
        else:
            image.save(temporary, format=format_name, compress_level=4)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_rows(
    rows: Sequence[EvalImage],
    model: torch.nn.Module,
    *,
    output_dir: str | Path,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device | str,
    mask_threshold: float,
    patch_gate_threshold: float,
    use_amp: bool,
    progress: Callable[[dict[str, object]], None] | None = None,
    sealed_inputs: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}
    sealed_image_hashes = (
        dict(sealed_inputs.get("image_sha256_by_id", {})) if sealed_inputs is not None else {}
    )
    sealed_label_hashes = (
        dict(sealed_inputs.get("label_sha256_by_id", {})) if sealed_inputs is not None else {}
    )
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        image_bytes = row.image_path.read_bytes()
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        expected_image_sha256 = sealed_image_hashes.get(row.identifier)
        if expected_image_sha256 is not None and image_sha256 != expected_image_sha256:
            raise RuntimeError(f"image input changed after contract preflight: {row.image_path}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prediction = predict_multitask_image(
            model,
            rgb,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            patch_gate_threshold=patch_gate_threshold,
            use_amp=use_amp,
        )
        ground_truth = (
            load_labelme_mask(
                row.label_path,
                image_shape=rgb.shape[:2],
                expected_sha256=sealed_label_hashes.get(row.identifier),
            )
            if row.label_path is not None
            else None
        )
        raw_metrics = score_map_metrics(
            prediction.raw_score_map, threshold=mask_threshold, ground_truth=ground_truth
        )
        gated_metrics = score_map_metrics(
            prediction.gated_score_map, threshold=mask_threshold, ground_truth=ground_truth
        )
        overlay = render_mask_overlay(
            rgb, prediction.gated_score_map, threshold=mask_threshold, alpha=0.55
        )
        comparison = np.concatenate((rgb, overlay), axis=1)
        output_paths = {
            "raw_heatmap": destination / "raw_heatmaps" / row.cohort / f"{row.identifier}.png",
            "gated_heatmap": destination
            / "gated_heatmaps"
            / row.cohort
            / f"{row.identifier}.png",
            "comparison": destination / "comparisons" / row.cohort / f"{row.identifier}.jpg",
        }
        _write_image_atomic(
            output_paths["raw_heatmap"],
            np.rint(prediction.raw_score_map * 255.0).astype(np.uint8),
            format_name="PNG",
        )
        _write_image_atomic(
            output_paths["gated_heatmap"],
            np.rint(prediction.gated_score_map * 255.0).astype(np.uint8),
            format_name="PNG",
        )
        _write_image_atomic(output_paths["comparison"], comparison, format_name="JPEG")
        if ground_truth is not None:
            output_paths["ground_truth"] = (
                destination / "ground_truth" / row.cohort / f"{row.identifier}.png"
            )
            _write_image_atomic(
                output_paths["ground_truth"], ground_truth.astype(np.uint8) * 255, format_name="PNG"
            )
        relative_outputs: dict[str, str] = {}
        for name, output_path in output_paths.items():
            relative = output_path.relative_to(destination).as_posix()
            relative_outputs[name] = relative
            artifacts[relative] = _sha256_file(output_path)
        record: dict[str, object] = {
            "id": row.identifier,
            "cohort": row.cohort,
            "image_path": str(row.image_path),
            "image_sha256": image_sha256,
            "label_path": str(row.label_path) if row.label_path is not None else None,
            "label_sha256": (
                sealed_label_hashes.get(row.identifier)
                if row.label_path is not None and row.identifier in sealed_label_hashes
                else _sha256_file(row.label_path)
                if row.label_path is not None
                else None
            ),
            "label_status": (
                "localized"
                if ground_truth is not None
                else "qualitative_only"
                if row.cohort == "ng"
                else "ok_unlabeled"
            ),
            "metadata": row.metadata,
            "image_shape": [int(value) for value in rgb.shape],
            "patch_count": prediction.patch_count,
            "baseline_probability_max": float(prediction.baseline_probabilities.max()),
            "baseline_patches_above_0_7": int(
                (prediction.baseline_probabilities >= 0.7).sum()
            ),
            "auxiliary_probability_max": float(prediction.auxiliary_probabilities.max()),
            "auxiliary_patches_above_gate": int(
                (prediction.auxiliary_probabilities >= patch_gate_threshold).sum()
            ),
            "raw_metrics": raw_metrics,
            "gated_metrics": gated_metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": relative_outputs,
        }
        predictions.append(record)
        if progress is not None:
            progress(
                {
                    "completed": index,
                    "total": len(rows),
                    "id": row.identifier,
                    "cohort": row.cohort,
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
    return predictions, artifacts


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(
    *,
    config_path: str | Path,
    eval_manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    multitask_checkpoint_path: str | Path,
    training_receipt_path: str | Path,
    expected_eval_manifest_sha256: str,
    expected_training_receipt_sha256: str,
    expected_multitask_sha256: str,
    output_dir: str | Path,
    batch_size: int,
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    if attempt.exists():
        raise FileExistsError(attempt)
    started_at = _utc_now()
    started = time.perf_counter()
    config_bytes = Path(config_path).read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a mapping")
    model_config = dict(config["model"])
    inputs_config = dict(config["inputs"])
    backbone_id = str(model_config["backbone_id"])
    expected_base_sha256 = str(inputs_config["expected_base_checkpoint_sha256"])
    if expected_multitask_sha256 != PINNED_MULTITASK_CHECKPOINT_SHA256:
        raise ValueError("multitask checkpoint SHA-256 is not the pinned job-7008 checkpoint")
    if expected_training_receipt_sha256 != PINNED_TRAINING_RECEIPT_SHA256:
        raise ValueError("training receipt SHA-256 is not the pinned job-7008 receipt")
    if expected_base_sha256 != PINNED_BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint SHA-256 is not the pinned epoch-3 checkpoint")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _verify_sha256(base_checkpoint_path, expected_base_sha256)
    training_receipt = verify_training_receipt(
        training_receipt_path,
        expected_receipt_sha256=expected_training_receipt_sha256,
        expected_multitask_checkpoint_sha256=expected_multitask_sha256,
        expected_base_checkpoint_sha256=expected_base_sha256,
    )
    rows = load_eval_rows(eval_manifest_path)
    contract_audit = verify_eval_contract(
        rows,
        manifest_path=eval_manifest_path,
        expected_manifest_sha256=expected_eval_manifest_sha256,
        contract=ACTUAL_OK_NG_CONTRACT,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("actual evaluation requires exactly one visible CUDA GPU")
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
        expected_checkpoint_sha256=expected_multitask_sha256,
        expected_base_checkpoint_sha256=expected_base_sha256,
        expected_backbone_id=backbone_id,
        expected_config_digest=canonical_digest(config),
    )
    frozen_settings = validate_frozen_settings(config, checkpoint_metadata)
    model.to(device).eval()
    mask_threshold = float(frozen_settings["mask_threshold"])
    patch_gate_threshold = mask_threshold

    project_root = Path(__file__).resolve().parents[2]
    code_commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code_dirty = bool(
        subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    launch_context = {
        "mode": "slurm" if os.environ.get("SLURM_JOB_ID") else "direct",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preexisting_gpu_process_authorized": os.environ.get(
            "PATCHQC_PREEXISTING_GPU_PROCESS_AUTHORIZED"
        ),
        "preexisting_gpu_process_pid": os.environ.get("PATCHQC_PREEXISTING_GPU_PROCESS_PID"),
        "preexisting_gpu_process_job_id": os.environ.get(
            "PATCHQC_PREEXISTING_GPU_PROCESS_JOB_ID"
        ),
    }

    def emit_progress(value: dict[str, object]) -> None:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    predictions, artifacts = evaluate_rows(
        rows,
        model,
        output_dir=attempt,
        patch_size=int(frozen_settings["patch_size"]),
        stride=int(frozen_settings["stride"]),
        batch_size=batch_size,
        device=device,
        mask_threshold=mask_threshold,
        patch_gate_threshold=patch_gate_threshold,
        use_amp=True,
        progress=emit_progress,
        sealed_inputs=contract_audit,
    )
    predictions_path = attempt / "predictions.jsonl"
    _write_jsonl_atomic(predictions_path, predictions)
    artifacts[predictions_path.relative_to(attempt).as_posix()] = _sha256_file(predictions_path)
    cohort_counts = {
        cohort: sum(row.cohort == cohort for row in rows) for cohort in ("ok", "ng")
    }
    localized_ng = [
        row for row in predictions if row["cohort"] == "ng" and row["label_status"] == "localized"
    ]
    receipt: dict[str, object] = {
        "format_version": 1,
        "stage": "actual_ok_ng_multitask_qualitative_eval",
        "status": "completed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {
            **cohort_counts,
            "total": len(rows),
            "localized_ng": len(localized_ng),
            "qualitative_only_ng": sum(
                row["cohort"] == "ng" and row["label_status"] == "qualitative_only"
                for row in predictions
            ),
            "artifacts": len(artifacts),
        },
        "fixed_settings": {
            "patch_size": int(frozen_settings["patch_size"]),
            "stride": int(frozen_settings["stride"]),
            "mask_threshold": mask_threshold,
            "patch_gate_threshold": patch_gate_threshold,
            "baseline_recording_threshold": 0.7,
            "batch_size": batch_size,
            "aggregation": "uniform_mean",
            "display_map": "auxiliary_patch_gated_mask",
        },
        "digests": {
            "config_sha256": config_sha256,
            "eval_manifest_sha256": expected_eval_manifest_sha256,
            "base_checkpoint_sha256": expected_base_sha256,
            "multitask_checkpoint_sha256": expected_multitask_sha256,
            "training_receipt_sha256": expected_training_receipt_sha256,
        },
        "checkpoint_metadata": checkpoint_metadata,
        "training_receipt": {
            "status": training_receipt["status"],
            "quality_gate": training_receipt["quality_gate"],
        },
        "evaluation_contract": contract_audit,
        "code": {"commit": code_commit, "dirty": code_dirty},
        "launch": launch_context,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
        },
        "summary": {
            "localized_ng_hits": sum(
                bool(row["gated_metrics"].get("ground_truth_hit", False)) for row in localized_ng
            ),
            "ok_images_with_activation": sum(
                row["cohort"] == "ok"
                and int(row["gated_metrics"]["predicted_positive_pixels"]) > 0
                for row in predictions
            ),
        },
        "artifacts": artifacts,
        "failures": 0,
        "temporary_files": sum(
            path.is_file() and path.name.startswith(".") and ".tmp" in path.name
            for path in attempt.rglob("*")
        ),
        "interpretation": "qualitative_only_no_ng_tuning",
    }
    if receipt["temporary_files"] != 0:
        raise RuntimeError("temporary files remain in evaluation attempt")
    _write_json_atomic(attempt / "run_receipt.json", receipt)
    attempt.replace(destination)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen actual OK/NG multitask qualitative evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--multitask-checkpoint", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--expected-eval-manifest-sha256", required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--expected-multitask-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(
        config_path=args.config,
        eval_manifest_path=args.eval_manifest,
        base_checkpoint_path=args.base_checkpoint,
        multitask_checkpoint_path=args.multitask_checkpoint,
        training_receipt_path=args.training_receipt,
        expected_eval_manifest_sha256=args.expected_eval_manifest_sha256,
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        expected_multitask_sha256=args.expected_multitask_sha256,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
