from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .dataset import BalancedPatchDataset, collate_patch_batch
from .manifest import load_manifest_records
from .model import DinoMaskMultiTaskProbe, load_timm_probe
from .multitask_train import evaluate_multitask, save_multitask_checkpoint, train_multitask_epoch
from .refinement_contract import (
    build_run_receipt,
    canonical_digest,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


def _as_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration {name} must be a mapping")
    return dict(value)


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return _as_mapping(value, "root")


def _verify_sha256(path: str | Path, expected_sha256: str) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must contain 64 lowercase hexadecimal characters")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {source}: expected {expected}, got {actual}")
    return actual


def _load_trusted_epoch3_head(
    base: Any,
    checkpoint_path: str | Path,
    expected_sha256: str,
    *,
    expected_backbone_id: str,
) -> dict[str, object]:
    """Verify the pinned checkpoint before using its trusted Python pickle payload."""
    _verify_sha256(checkpoint_path, expected_sha256)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("trusted epoch-3 checkpoint must contain a mapping")
    if payload.get("backbone_id") != expected_backbone_id:
        raise ValueError("trusted epoch-3 checkpoint backbone_id mismatch")
    head_state = payload.get("head_state_dict")
    if not isinstance(head_state, Mapping):
        raise ValueError("trusted epoch-3 checkpoint is missing head_state_dict")
    head = getattr(base, "head", None)
    if not isinstance(head, nn.Module):
        raise TypeError("base classifier must expose an nn.Module head")
    head.load_state_dict(dict(head_state), strict=True)
    return payload


def _write_yaml_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_png_atomic(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_checkpoint_atomic(
    model: nn.Module,
    destination: Path,
    *,
    base_checkpoint_path: str,
    base_checkpoint_sha256: str,
    backbone_id: str,
    config_digest: str,
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        save_multitask_checkpoint(
            model,
            temporary,
            base_checkpoint_path=base_checkpoint_path,
            base_checkpoint_sha256=base_checkpoint_sha256,
            backbone_id=backbone_id,
            config_digest=config_digest,
            epoch=epoch,
            metrics=metrics,
        )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _loader(
    dataset: BalancedPatchDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_patch_batch,
        generator=generator,
    )


def _build_data(
    config: Mapping[str, object], manifest_path: str | Path, device: torch.device
) -> tuple[BalancedPatchDataset, DataLoader, BalancedPatchDataset, DataLoader]:
    data_config = _as_mapping(config.get("data"), "data")
    train_config = _as_mapping(config.get("train"), "train")
    train_records = load_manifest_records(manifest_path, split="train")
    holdout_records = load_manifest_records(manifest_path, split="holdout")
    if not train_records or not holdout_records:
        raise RuntimeError("manifest must contain non-empty train and holdout splits")
    seed = int(train_config["seed"])
    common = {
        "patch_size": int(data_config.get("patch_size", 256)),
        "positive_probability": float(train_config.get("positive_probability", 0.5)),
        "source_reuse": int(data_config.get("patches_per_source", 8)),
        "return_mask": True,
    }
    train_dataset = BalancedPatchDataset(
        train_records,
        samples_per_epoch=int(train_config["samples_per_epoch"]),
        seed=seed,
        **common,
    )
    holdout_dataset = BalancedPatchDataset(
        holdout_records,
        samples_per_epoch=int(train_config["evaluation_samples"]),
        seed=seed + 10_000,
        **common,
    )
    loader_options = {
        "batch_size": int(train_config["batch_size"]),
        "workers": int(train_config.get("workers", 0)),
        "device": device,
    }
    return (
        train_dataset,
        _loader(train_dataset, seed=seed, **loader_options),
        holdout_dataset,
        _loader(holdout_dataset, seed=seed + 10_000, **loader_options),
    )


@torch.inference_mode()
def _capture_baseline_reference(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        batch = next(iter(loader))
    except StopIteration as error:
        raise ValueError("baseline reference requires a non-empty holdout loader") from error
    images = batch.get("image")
    if not isinstance(images, torch.Tensor):
        raise TypeError("baseline reference batch is missing image tensors")
    reference_images = images.detach().cpu().clone()
    model.eval()
    reference_logits = model(images.to(device)).baseline_logits
    if not isinstance(reference_logits, torch.Tensor) or reference_logits.ndim != 1:
        raise ValueError("baseline reference logits must have shape [batch]")
    if reference_logits.shape[0] != reference_images.shape[0]:
        raise ValueError("baseline reference images and logits must have matching batches")
    return reference_images, reference_logits.detach().cpu().clone()


@torch.inference_mode()
def _measure_baseline_drift(
    model: nn.Module,
    reference: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, object]:
    reference_images, reference_logits = reference
    model.eval()
    current_logits = model(reference_images.to(device)).baseline_logits
    if not isinstance(current_logits, torch.Tensor) or current_logits.shape != reference_logits.shape:
        raise ValueError("current and reference baseline logits must have matching tensor shapes")
    maximum = float((current_logits.detach().cpu() - reference_logits).abs().max())
    if not np.isfinite(maximum):
        raise ValueError("baseline drift audit produced a non-finite result")
    return {
        "measured": True,
        "examples": int(reference_images.shape[0]),
        "max_abs": maximum,
    }


def _evaluate_quality_gate(
    metrics: Mapping[str, object],
    evaluation_config: Mapping[str, object],
) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "predicted_positive_fraction": {
            "lower_exclusive": 0.0,
            "upper_exclusive": 1.0,
        },
        "minimum_pixel_dice": float(evaluation_config["minimum_pixel_dice"]),
        "minimum_localization_hit_rate": float(
            evaluation_config["minimum_localization_hit_rate"]
        ),
        "maximum_normal_predicted_area_p95": float(
            evaluation_config["maximum_normal_predicted_area_p95"]
        ),
    }
    for name in (
        "minimum_pixel_dice",
        "minimum_localization_hit_rate",
        "maximum_normal_predicted_area_p95",
    ):
        threshold = float(thresholds[name])
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"quality gate threshold {name} must be finite and between zero and one")

    observed = {
        "predicted_positive_fraction": float(metrics["predicted_positive_fraction"]),
        "pixel_dice": float(metrics["pixel_dice"]),
        "positive_localization_hit_rate": float(metrics["positive_localization_hit_rate"]),
        "normal_predicted_area_p95": float(metrics["normal_predicted_area_p95"]),
    }
    if not all(np.isfinite(value) for value in observed.values()):
        raise ValueError("quality gate metrics must be finite")

    failures: list[dict[str, object]] = []
    predicted_fraction = observed["predicted_positive_fraction"]
    if not 0.0 < predicted_fraction < 1.0:
        failures.append(
            {
                "metric": "predicted_positive_fraction",
                "actual": predicted_fraction,
                "required": "strictly_between_0_and_1",
            }
        )
    comparisons = (
        (
            "pixel_dice",
            observed["pixel_dice"],
            "minimum_pixel_dice",
            float(thresholds["minimum_pixel_dice"]),
            "at_least",
        ),
        (
            "positive_localization_hit_rate",
            observed["positive_localization_hit_rate"],
            "minimum_localization_hit_rate",
            float(thresholds["minimum_localization_hit_rate"]),
            "at_least",
        ),
        (
            "normal_predicted_area_p95",
            observed["normal_predicted_area_p95"],
            "maximum_normal_predicted_area_p95",
            float(thresholds["maximum_normal_predicted_area_p95"]),
            "at_most",
        ),
    )
    for metric_name, actual, threshold_name, threshold, direction in comparisons:
        failed = actual < threshold if direction == "at_least" else actual > threshold
        if failed:
            failures.append(
                {
                    "metric": metric_name,
                    "actual": actual,
                    "required": direction,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                }
            )
    return {"passed": not failures, "failures": failures, "thresholds": thresholds}


def _denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype)[:, None, None]
    rgb = (image.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return np.rint(rgb * 255).astype(np.uint8)


def _clean_crop(metadata: Mapping[str, object], patch_size: int) -> np.ndarray:
    source_path = Path(str(metadata.get("image_path", "")))
    box_value = metadata.get("patch_box")
    if not isinstance(box_value, (tuple, list)) or len(box_value) != 4:
        raise ValueError("visualization metadata is missing a native patch_box")
    box = tuple(int(value) for value in box_value)
    with Image.open(source_path) as image:
        crop = np.asarray(image.convert("RGB").crop(box))
    if crop.shape != (patch_size, patch_size, 3):
        raise ValueError("visualization clean crop is not native patch size")
    return crop


@torch.inference_mode()
def _write_visualizations(
    model: nn.Module,
    dataset: BalancedPatchDataset,
    destination: Path,
    *,
    device: torch.device,
    count: int,
) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    model.eval()
    for dataset_index in range(len(dataset)):
        if len(rows) >= count:
            break
        item = dataset[dataset_index]
        label = item.get("label")
        if not isinstance(label, torch.Tensor) or float(label) != 1.0:
            continue
        image = item.get("image")
        mask = item.get("mask")
        metadata = item.get("metadata")
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("visualization item is missing image or mask tensors")
        if not isinstance(metadata, Mapping):
            raise TypeError("visualization item is missing metadata")
        output = model(image.unsqueeze(0).to(device))
        probability = output.mask_logits[0, 0].detach().float().sigmoid().cpu().numpy()
        synthetic = _denormalize(image)
        clean = _clean_crop(metadata, dataset.patch_size)
        ground_truth = np.repeat((mask[0].numpy() > 0.5)[..., None], 3, axis=2).astype(np.uint8) * 255
        red = np.zeros_like(synthetic)
        red[..., 0] = 255
        alpha = np.clip(probability[..., None] * 0.65, 0, 0.65)
        overlay = np.rint(synthetic * (1.0 - alpha) + red * alpha).astype(np.uint8)
        panel = np.concatenate((clean, synthetic, ground_truth, overlay), axis=1)
        filename = f"panel_{len(rows):03d}.png"
        panel_path = destination / filename
        _write_png_atomic(panel_path, panel)
        relative = Path("visualizations") / filename
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(panel_path),
                "bytes": panel_path.stat().st_size,
                "dataset_index": dataset_index,
                "source_path": str(metadata.get("image_path")),
                "patch_box": list(metadata.get("patch_box", ())),
                "synthetic_family": metadata.get("synthetic_family"),
                "panel_order": ["original_clean_crop", "synthetic", "ground_truth", "prediction_overlay"],
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"requested {count} positive visualization panels, produced {len(rows)}")
    write_jsonl_atomic(destination / "index.jsonl", rows)
    return rows


def _gpu_info(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"available": False, "visible_index": None, "uuid": None, "name": None}
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    if isinstance(uuid, bytes):
        uuid = uuid.decode()
    physical_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if uuid is None:
        visible = [token.strip() for token in (physical_visibility or "").split(",") if token.strip()]
        selector = visible[index] if index < len(visible) else None
        if selector and selector.startswith(("GPU-", "MIG-")):
            uuid = selector
        else:
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                indexed_uuids = {
                    physical_index.strip(): physical_uuid.strip()
                    for line in completed.stdout.splitlines()
                    if "," in line
                    for physical_index, physical_uuid in [line.split(",", 1)]
                }
                if selector is not None:
                    uuid = indexed_uuids.get(selector)
                elif len(indexed_uuids) == 1:
                    uuid = next(iter(indexed_uuids.values()))
            except (OSError, subprocess.SubprocessError):
                uuid = None
    return {
        "available": True,
        "visible_index": int(index),
        "physical_visibility": physical_visibility,
        "uuid": str(uuid) if uuid is not None else None,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def _code_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", True
    return commit, dirty


def execute_smoke(
    *,
    config: Mapping[str, object],
    manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    output_dir: str | Path,
    model: nn.Module,
    device: torch.device,
    code_commit: str | None = None,
    code_dirty: bool | None = None,
) -> dict[str, object]:
    started_at = _utc_now_iso()
    config = dict(config)
    model_config = _as_mapping(config.get("model"), "model")
    train_config = _as_mapping(config.get("train"), "train")
    loss_config = _as_mapping(config.get("loss"), "loss")
    evaluation_config = _as_mapping(config.get("evaluation"), "evaluation")
    inputs_config = _as_mapping(config.get("inputs"), "inputs")
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"output directory must be absent or empty, got non-empty path: {destination}")
    manifest_sha256 = _verify_sha256(
        manifest_path, str(inputs_config["expected_manifest_sha256"])
    )
    base_sha256 = _verify_sha256(
        base_checkpoint_path, str(inputs_config["expected_base_checkpoint_sha256"])
    )
    destination.mkdir(parents=True, exist_ok=True)
    config_digest = canonical_digest(config)
    resolved = dict(config)
    resolved["resolved_inputs"] = {
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": manifest_sha256,
        "base_checkpoint_path": str(Path(base_checkpoint_path)),
        "base_checkpoint_sha256": base_sha256,
    }
    _write_yaml_atomic(destination / "resolved_config.yaml", resolved)

    seed = int(train_config["seed"])
    _seed_everything(seed, device)
    model = model.to(device)
    train_dataset, train_loader, holdout_dataset, holdout_loader = _build_data(
        config, manifest_path, device
    )
    baseline_reference = _capture_baseline_reference(
        model,
        holdout_loader,
        device=device,
    )
    loss_kwargs = {
        "auxiliary_weight": float(loss_config["auxiliary_weight"]),
        "mask_bce_weight": float(loss_config["mask_bce_weight"]),
        "mask_dice_weight": float(loss_config["mask_dice_weight"]),
        "max_positive_weight": float(loss_config["max_positive_weight"]),
    }
    mask_threshold = float(evaluation_config["mask_threshold"])
    trainable_builder = getattr(model, "trainable_parameters", None)
    if not callable(trainable_builder):
        raise TypeError("multitask model must provide trainable_parameters()")
    optimizer = torch.optim.AdamW(
        trainable_builder(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    schedule = [
        {
            "epoch": epoch,
            "seed": seed,
            "train_samples": len(train_dataset),
            "evaluation_samples": len(holdout_dataset),
            "batch_size": int(train_config["batch_size"]),
            "learning_rate": float(train_config["learning_rate"]),
            "loss": loss_kwargs,
            "mask_threshold": mask_threshold,
        }
        for epoch in range(1, epochs + 1)
    ]
    write_jsonl_atomic(destination / "schedule.jsonl", schedule)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics_rows: list[dict[str, object]] = []
    checkpoints: list[Path] = []
    final_audit: dict[str, object] | None = None
    final_holdout: Mapping[str, object] | None = None
    use_amp = bool(train_config.get("use_amp", True))
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        training = train_multitask_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_amp=use_amp,
            loss_kwargs=loss_kwargs,
            mask_threshold=mask_threshold,
        )
        holdout = evaluate_multitask(
            model,
            holdout_loader,
            device=device,
            use_amp=use_amp,
            loss_kwargs=loss_kwargs,
            mask_threshold=mask_threshold,
        )
        audit = _measure_baseline_drift(model, baseline_reference, device=device)
        limit = float(evaluation_config["baseline_logit_max_abs_drift"])
        if float(audit["max_abs"]) > limit:
            raise RuntimeError(
                f"baseline logit drift {audit['max_abs']} exceeds configured limit {limit}"
            )
        final_audit = audit
        final_holdout = holdout
        elapsed_epoch = time.perf_counter() - epoch_started
        epoch_examples = int(training["examples"]) + int(holdout["examples"])
        metrics_rows.append(
            {
                "epoch": epoch,
                "train": training,
                "synthetic_holdout": holdout,
                "baseline_drift_audit": audit,
                "elapsed_seconds": elapsed_epoch,
                "samples_per_second": epoch_examples / elapsed_epoch,
            }
        )
        checkpoint = destination / f"checkpoint_epoch_{epoch:03d}.pt"
        _save_checkpoint_atomic(
            model,
            checkpoint,
            base_checkpoint_path=str(Path(base_checkpoint_path)),
            base_checkpoint_sha256=base_sha256,
            backbone_id=str(model_config["backbone_id"]),
            config_digest=config_digest,
            epoch=epoch,
            metrics={
                "train_loss": float(training["loss"]),
                "holdout_loss": float(holdout["loss"]),
                "baseline_logit_max_abs_drift": float(audit["max_abs"]),
                "examples": int(holdout["examples"]),
            },
        )
        checkpoints.append(checkpoint)
    write_jsonl_atomic(destination / "metrics.jsonl", metrics_rows)

    visualization_rows = _write_visualizations(
        model,
        holdout_dataset,
        destination / "visualizations",
        device=device,
        count=int(evaluation_config["visualization_panels"]),
    )
    elapsed = time.perf_counter() - started
    processed = epochs * (len(train_dataset) + len(holdout_dataset))
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    if final_audit is None:
        raise RuntimeError("baseline drift audit was not measured")
    if final_holdout is None:
        raise RuntimeError("synthetic holdout metrics were not measured")
    quality_gate = _evaluate_quality_gate(final_holdout, evaluation_config)
    if code_commit is None or code_dirty is None:
        detected_commit, detected_dirty = _code_state()
        code_commit = detected_commit if code_commit is None else code_commit
        code_dirty = detected_dirty if code_dirty is None else code_dirty
    artifact_paths = [
        destination / "resolved_config.yaml",
        destination / "schedule.jsonl",
        destination / "metrics.jsonl",
        destination / "visualizations/index.jsonl",
        *checkpoints,
        *(destination / row["path"] for row in visualization_rows),
    ]
    artifacts = {
        str(path.relative_to(destination)): sha256_file(path) for path in artifact_paths
    }
    temporary_files = len(list(destination.rglob("*.tmp")))
    receipt = build_run_receipt(
        "synthetic_mask_multitask_smoke",
        digests={
            "config_sha256": config_digest,
            "manifest_sha256": manifest_sha256,
            "base_checkpoint_sha256": base_sha256,
        },
        counts={
            "epochs": epochs,
            "train_samples_per_epoch": len(train_dataset),
            "evaluation_samples_per_epoch": len(holdout_dataset),
            "processed_samples": processed,
            "visualization_panels": len(visualization_rows),
        },
        metrics={
            "elapsed_seconds": elapsed,
            "samples_per_second": processed / elapsed,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "baseline_drift_audit": final_audit,
        },
        failures=0,
        temporary_files=temporary_files,
        seeds={"train": seed, "holdout": seed + 10_000},
        artifacts=artifacts,
        extra={
            "code_commit": code_commit,
            "code_dirty": bool(code_dirty),
            "gpu": _gpu_info(device),
            "backbone_id": str(model_config["backbone_id"]),
            "base_checkpoint_path": str(Path(base_checkpoint_path)),
        },
    )
    receipt["started_at"] = started_at
    receipt["finished_at"] = _utc_now_iso()
    receipt["quality_gate"] = quality_gate
    if not bool(quality_gate["passed"]):
        receipt["status"] = "quality_gate_failed"
    write_json_atomic(destination / "run_receipt.json", receipt)
    return receipt


def run(
    config_path: str | Path,
    manifest_path: str | Path,
    base_checkpoint_path: str | Path,
    output_dir: str | Path,
) -> int:
    config = _load_config(config_path)
    inputs_config = _as_mapping(config.get("inputs"), "inputs")
    model_config = _as_mapping(config.get("model"), "model")
    train_config = _as_mapping(config.get("train"), "train")
    _verify_sha256(manifest_path, str(inputs_config["expected_manifest_sha256"]))
    _verify_sha256(
        base_checkpoint_path, str(inputs_config["expected_base_checkpoint_sha256"])
    )
    if not torch.cuda.is_available():
        raise RuntimeError("multitask smoke requires one Slurm-allocated CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one Slurm-visible GPU, got {torch.cuda.device_count()}")
    if str(model_config.get("provider", "timm")) != "timm":
        raise ValueError("multitask smoke supports only the frozen timm DINOv3 probe")
    device = torch.device("cuda:0")
    _seed_everything(int(train_config["seed"]), device)
    torch.set_float32_matmul_precision("high")
    base = load_timm_probe(
        str(model_config["backbone_id"]),
        local_files_only=bool(model_config.get("local_files_only", True)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    _load_trusted_epoch3_head(
        base,
        base_checkpoint_path,
        str(inputs_config["expected_base_checkpoint_sha256"]),
        expected_backbone_id=str(model_config["backbone_id"]),
    )
    model = DinoMaskMultiTaskProbe(
        base, decoder_channels=int(model_config.get("decoder_channels", 32))
    )
    commit, dirty = _code_state()
    execute_smoke(
        config=config,
        manifest_path=manifest_path,
        base_checkpoint_path=base_checkpoint_path,
        output_dir=output_dir,
        model=model,
        device=device,
        code_commit=commit,
        code_dirty=dirty,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patch-qc-train-multitask")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    return run(args.config, args.manifest, args.base_checkpoint, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
