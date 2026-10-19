"""Bounded native-image A/B scoring with provenance-safe shard output."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .cohort import audit_cohort_reference_isolation
from .core import GridAccumulator, axis_positions, dense_reference_distance_normalized, tail_mean_score


BACKBONE_ID = "vit_large_patch16_dinov3.lvd1689m"
NATIVE_SIZE = (8192, 5464)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass
class RunnerConfig:
    """Frozen runner settings; production defaults preserve the experiment contract."""

    native_size: tuple[int, int] = NATIVE_SIZE
    tile_size: int = 256
    stride: int = 128
    token_step: int = 16
    grid_step: int = 8
    tile_batch_size: int = 8
    reference_chunk_size: int = 1
    reference_image_cache_size: int = 5
    decode_workers: int = 2
    physical_gpu_index: int | None = None
    precision: str = "fp16"
    map_dir: Path | None = None
    config_values: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tile_size != 256 or self.stride != 128:
            raise ValueError("the frozen experiment requires 256-pixel tiles with 128-pixel stride")
        if self.token_step != 16 or self.grid_step != 8:
            raise ValueError("the frozen experiment requires 16-pixel tokens and an 8-pixel grid")
        if min(*self.native_size, self.tile_batch_size, self.reference_chunk_size, self.reference_image_cache_size, self.decode_workers) <= 0:
            raise ValueError("native size and bounded batch sizes must be positive")
        if self.reference_image_cache_size > 5:
            raise ValueError("the frozen full cohort has at most five normal references")
        if self.physical_gpu_index is not None and self.physical_gpu_index < 0:
            raise ValueError("physical GPU index must be non-negative")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError("inference precision must be fp16 or fp32")

    @property
    def scoring_config(self) -> dict[str, object]:
        """Return only settings that can change emitted scores or maps."""
        return {
            "native_size": self.native_size,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "token_step": self.token_step,
            "grid_step": self.grid_step,
            "tile_batch_size": self.tile_batch_size,
            "reference_chunk_size": self.reference_chunk_size,
            "reference_image_cache_size": self.reference_image_cache_size,
            "decode_workers": self.decode_workers,
            "precision": self.precision,
            "config_values": self.config_values,
        }

    @property
    def digest(self) -> str:
        """Identity shared by all shards whose maps are merge-compatible."""
        return _json_digest(self.scoring_config)

    @property
    def execution_digest(self) -> str:
        """Placement-specific identity retained for audit, never used to merge maps."""
        return _json_digest({"scoring_config_sha256": self.digest, "physical_gpu_index": self.physical_gpu_index})


def _default_model_loader(model_name: str, *, local_files_only: bool = True) -> torch.nn.Module:
    """Import the approved offline probe loader without a network fallback."""
    try:
        module = importlib.import_module("patch_qc.model")
    except ImportError as error:
        raise RuntimeError("patch_qc.model must be importable on the Slurm worker") from error
    return module.load_timm_probe(model_name, local_files_only=local_files_only)


def _load_verified_checkpoint(checkpoint: Path) -> tuple[dict[str, object], str]:
    """Load an exact-SHA-verified local checkpoint, retaining its legacy RNG metadata.

    The safe weights-only reader is preferred.  Legacy epoch-3 files contain
    NumPy RNG state that PyTorch's restricted unpickler rejects; only after the
    caller has verified the frozen checkpoint's exact SHA-256 may this function
    retry with Python unpickling.  Never call this helper for an unverified path.
    """
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        mode = "weights_only_true"
    except pickle.UnpicklingError:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mode = "trusted_weights_only_false"
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload, mode


class FrozenDinoEncoder:
    """The epoch-3 classifier and its DINO patch tokens from one model forward."""

    def __init__(
        self,
        checkpoint: Path,
        expected_sha256: str,
        device: torch.device,
        *,
        model_loader: Callable[..., torch.nn.Module] | None = None,
        precision: str = "fp16",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.expected_sha256 = expected_sha256.lower()
        self.device = device
        if precision not in {"fp16", "fp32"}:
            raise ValueError("inference precision must be fp16 or fp32")
        self.precision = precision
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        if self.checkpoint_sha256 != self.expected_sha256:
            raise ValueError(
                f"checkpoint digest mismatch: expected {self.expected_sha256}, got {self.checkpoint_sha256}"
            )

        self.model = (model_loader or _default_model_loader)(BACKBONE_ID, local_files_only=True)
        payload, self.checkpoint_load_mode = _load_verified_checkpoint(self.checkpoint)
        if payload.get("backbone_id") != BACKBONE_ID:
            raise ValueError("checkpoint is not an epoch-3 timm DINOv3 probe checkpoint")
        head_state = payload.get("head_state_dict")
        if not isinstance(head_state, dict):
            raise ValueError("checkpoint is missing head_state_dict")
        self.model.head.load_state_dict(head_state, strict=True)
        self.model.eval().to(self.device)
        self.model_digest = _json_digest(
            {"checkpoint_sha256": self.checkpoint_sha256, "backbone_id": BACKBONE_ID, "epoch": payload.get("epoch"), "precision": self.precision}
        )

    def encode(self, batch: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sigmoid epoch-3 scores and once-normalized FP16 [B,16,16,1024] tokens."""
        array = np.asarray(batch)
        if array.ndim != 4 or array.shape[1:] != (256, 256, 3):
            raise ValueError("encoder expects an RGB [batch,256,256,3] tile batch")
        if not np.isfinite(array).all():
            raise ValueError("encoder input contains non-finite pixels")
        pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=pixels.dtype)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=pixels.dtype)[:, None, None]
        pixels = ((pixels - mean) / std).to(self.device, non_blocking=True)
        with torch.inference_mode(), _inference_autocast(self.device, self.precision):
            logits, patch_tokens = _forward_with_patch_tokens(self.model, pixels)
            probabilities = torch.sigmoid(logits).float()
            if patch_tokens.ndim != 3 or patch_tokens.shape[1] != 16 * 16 or patch_tokens.shape[2] != 1024:
                raise ValueError(f"unexpected DINO patch-token shape: {tuple(patch_tokens.shape)}")
            tokens = functional.normalize(patch_tokens.float(), dim=-1).to(torch.float16).reshape(-1, 16, 16, 1024)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(tokens).all():
            raise ValueError("model produced non-finite probability or token values")
        return probabilities, tokens


def _inference_autocast(device: torch.device, precision: str):
    """Enable the frozen FP16 inference path only on CUDA workers."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and precision == "fp16")


def _forward_with_patch_tokens(model: torch.nn.Module, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Support both checked-in and remote epoch-3 probe forward contracts."""
    forward_with_tokens = getattr(model, "forward_with_tokens", None)
    if callable(forward_with_tokens):
        result = forward_with_tokens(pixels)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("forward_with_tokens must return (logits, patch_tokens)")
        logits, patch_tokens = result
    else:
        encode_with_logits = getattr(model, "encode_with_logits", None)
        if not callable(encode_with_logits):
            raise ValueError("probe model has neither forward_with_tokens nor encode_with_logits")
        result = encode_with_logits(pixels)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("encode_with_logits must return (logits, pooled, patch_tokens)")
        logits, _, patch_tokens = result
    if not isinstance(logits, torch.Tensor) or not isinstance(patch_tokens, torch.Tensor):
        raise ValueError("probe forward contract returned non-tensor logits or patch tokens")
    return logits, patch_tokens


def _verify_source(row: dict[str, object], config: RunnerConfig, label: str) -> Path:
    source = Path(str(row.get("source_path", "")))
    expected = row.get("image_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} is missing a 64-character source SHA-256")
    if not source.is_file():
        raise ValueError(f"{label} source is missing: {source}")
    actual = _sha256(source)
    if actual != expected:
        raise ValueError(f"{label} source digest mismatch: expected {expected}, got {actual}")
    with Image.open(source) as image:
        if image.size != config.native_size:
            raise ValueError(f"{label} geometry mismatch: expected {config.native_size}, got {image.size}")
        if (row.get("width"), row.get("height")) != image.size:
            raise ValueError(f"{label} declared geometry mismatch: expected {image.size}")
    return source


def _decode_verified(source: Path, config: RunnerConfig, label: str) -> np.ndarray:
    """Decode one already validated source; callers retain it only for a bounded chunk."""
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape != (config.native_size[1], config.native_size[0], 3) or not np.isfinite(rgb).all():
        raise ValueError(f"{label} decoded image is invalid or non-finite")
    return rgb


def _verify_and_decode(row: dict[str, object], config: RunnerConfig, label: str) -> np.ndarray:
    return _decode_verified(_verify_source(row, config, label), config, label)


def _decode_reference_cache(sources: Sequence[Path], config: RunnerConfig) -> list[np.ndarray]:
    """Decode the bounded reference bank once, in deterministic source order."""
    if len(sources) > config.reference_image_cache_size:
        raise ValueError(
            f"reference cache holds {config.reference_image_cache_size}, but query requires {len(sources)} references"
        )
    labels = [f"reference[{index}]" for index in range(len(sources))]
    with ThreadPoolExecutor(max_workers=min(config.decode_workers, len(sources) or 1), thread_name_prefix="dense-ref-decode") as pool:
        return list(pool.map(lambda item: _decode_verified(item[0], config, item[1]), zip(sources, labels, strict=True)))


def _tile_boxes(width: int, height: int, config: RunnerConfig) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, config.tile_size, config.tile_size)
        for y in axis_positions(height, config.tile_size, config.stride)
        for x in axis_positions(width, config.tile_size, config.stride)
    ]


def _tile_batches(image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], batch_size: int) -> Iterable[tuple[Sequence[tuple[int, int, int, int]], np.ndarray]]:
    for start in range(0, len(boxes), batch_size):
        selected = boxes[start : start + batch_size]
        tiles = np.stack([image[y : y + height, x : x + width] for x, y, width, height in selected])
        yield selected, tiles


def _normalized_channel_first(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4 or tokens.shape[1:3] != (16, 16) or tokens.dtype != torch.float16:
        raise ValueError("encoder must return normalized FP16 [batch,16,16,width] tokens")
    if not torch.isfinite(tokens).all():
        raise ValueError("encoder tokens are non-finite")
    return tokens.permute(0, 3, 1, 2).contiguous()


def _save_maps(
    destination: Path,
    source_sha256: str,
    baseline_map: np.ndarray,
    baseline_valid: np.ndarray,
    dense_map: np.ndarray,
    dense_valid: np.ndarray,
    config: RunnerConfig,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{source_sha256}.npz"
    metadata = json.dumps(
        {
            "artifact_version": 1,
            "source_sha256": source_sha256,
            "config_sha256": config.digest,
            "native_size": list(config.native_size),
            "tile_size": config.tile_size,
            "stride": config.stride,
            "token_step": config.token_step,
            "grid_step": config.grid_step,
            "methods": ["dense_reference", "epoch3_patch"],
        },
        sort_keys=True,
    )
    np.savez_compressed(
        path,
        baseline_map=baseline_map.astype(np.float32, copy=False),
        baseline_valid_mask=baseline_valid.astype(np.bool_, copy=False),
        dense_map=dense_map.astype(np.float32, copy=False),
        dense_valid_mask=dense_valid.astype(np.bool_, copy=False),
        metadata=np.asarray(metadata),
    )
    _fsync_file(path)
    return path


def _map_artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def score_query(
    query_row: dict[str, object],
    reference_rows: Sequence[dict[str, object]],
    encoder: Any,
    config: RunnerConfig,
) -> dict[str, object]:
    """Score one exact-SHA query using one query tile stream shared by both methods."""
    started = time.monotonic()
    if not reference_rows:
        raise ValueError("a query requires at least one validated normal reference")
    for reference in reference_rows:
        if not isinstance(reference, dict):
            raise ValueError("reference row is not an object")
        query_capture, query_sha = query_row.get("capture_id"), query_row.get("image_sha256")
        reference_capture, reference_sha = reference.get("capture_id"), reference.get("image_sha256")
        if not isinstance(query_capture, str) or not query_capture or not isinstance(reference_capture, str) or not reference_capture:
            raise ValueError("query/reference capture identity is missing")
        if not isinstance(query_sha, str) or not isinstance(reference_sha, str):
            raise ValueError("query/reference image SHA-256 is missing")
        if query_capture == reference_capture:
            raise ValueError("query/reference capture leakage")
        if query_sha == reference_sha:
            raise ValueError("query/reference SHA leakage")
    query = _verify_and_decode(query_row, config, "query")
    reference_sources = [
        _verify_source(row, config, f"reference[{index}]") for index, row in enumerate(reference_rows)
    ]
    references = _decode_reference_cache(reference_sources, config)
    width, height = config.native_size
    boxes = _tile_boxes(width, height, config)
    baseline_grid = GridAccumulator(width, height, step=config.grid_step)
    dense_grid = GridAccumulator(width, height, step=config.grid_step)
    max_tile_batch = 0
    max_reference_chunk = 0

    # Query tokens exist only for the current tile batch; each reference is
    # cropped through the exact same boxes before the next batch starts.
    for selected_boxes, query_tiles in _tile_batches(query, boxes, config.tile_batch_size):
        max_tile_batch = max(max_tile_batch, len(selected_boxes))
        probabilities, query_tokens = encoder.encode(query_tiles)
        probabilities = torch.as_tensor(probabilities).detach().cpu().reshape(-1)
        query_cf = _normalized_channel_first(torch.as_tensor(query_tokens))
        if probabilities.numel() != len(selected_boxes) or not torch.isfinite(probabilities).all():
            raise ValueError("encoder probabilities do not match the query tile batch or are non-finite")
        for box, probability in zip(selected_boxes, probabilities.tolist(), strict=True):
            baseline_grid.add_patch(*box, float(probability))

        best_distance: torch.Tensor | None = None
        for start in range(0, len(references), config.reference_chunk_size):
            chunk = references[start : start + config.reference_chunk_size]
            max_reference_chunk = max(max_reference_chunk, len(chunk))
            for reference in chunk:
                ref_tiles = np.stack([reference[y : y + tile_height, x : x + tile_width] for x, y, tile_width, tile_height in selected_boxes])
                _, reference_tokens = encoder.encode(ref_tiles)
                reference_cf = _normalized_channel_first(torch.as_tensor(reference_tokens)).unsqueeze(0)
                distance = dense_reference_distance_normalized(query_cf, reference_cf, radius=1)
                best_distance = distance if best_distance is None else torch.minimum(best_distance, distance)
        assert best_distance is not None
        for box, values in zip(selected_boxes, best_distance.detach().cpu(), strict=True):
            dense_grid.add_token_tile(box[0], box[1], values, token_step=config.token_step)

    baseline_map, baseline_valid = baseline_grid.finalize()
    dense_map, dense_valid = dense_grid.finalize()
    if not (baseline_valid.any() and dense_valid.any()):
        raise ValueError("no finite grid cells were accumulated")
    if not (np.isfinite(baseline_map[baseline_valid]).all() and np.isfinite(dense_map[dense_valid]).all()):
        raise ValueError("non-finite scalar map cells")
    map_archive = None
    map_artifact = None
    if config.map_dir is not None:
        map_archive = _save_maps(config.map_dir, str(query_row["image_sha256"]), baseline_map, baseline_valid, dense_map, dense_valid, config)
        map_artifact = _map_artifact(map_archive)
    methods = {
        "epoch3_patch": {"score": tail_mean_score(baseline_map), "tile_boxes": boxes},
        "dense_reference": {"score": tail_mean_score(dense_map), "tile_boxes": boxes},
    }
    return {
        "source_sha256": query_row["image_sha256"],
        "source_path": query_row["source_path"],
        "capture_id": query_row.get("capture_id"),
        "cohort": query_row.get("cohort"),
        "product": query_row.get("product"),
        "canonical_condition": query_row.get("canonical_condition"),
        "source_label": query_row.get("source_label"),
        "labeled": query_row.get("labeled"),
        "label_path": query_row.get("label_path"),
        "label_sha256": query_row.get("label_sha256"),
        "reference_membership": [
            {"capture_id": reference.get("capture_id"), "source_path": reference.get("source_path"), "image_sha256": reference.get("image_sha256")}
            for reference in reference_rows
        ],
        "tile_count": len(boxes),
        "reference_count": len(reference_rows),
        "methods": methods,
        "map_archive": str(map_archive) if map_archive is not None else None,
        "map_artifact": map_artifact,
        "telemetry": {
            "max_tile_batch": max_tile_batch,
            "max_reference_chunk": max_reference_chunk,
            "decoded_reference_images": len(references),
            "elapsed_seconds": time.monotonic() - started,
        },
    }


def rank_owns(source_sha256: str, rank: int, world_size: int) -> bool:
    """Assign every valid source SHA to exactly one deterministic rank."""
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world_size")
    if len(source_sha256) != 64:
        raise ValueError("source SHA-256 must have 64 hex characters")
    try:
        return int(source_sha256, 16) % world_size == rank
    except ValueError as error:
        raise ValueError("source SHA-256 must be hexadecimal") from error


def _runtime_digest() -> str:
    return _json_digest({"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()})


def _gpu_telemetry(requested_physical_index: int | None = None) -> dict[str, object]:
    """Capture the controller-selected visible device without touching other GPUs."""
    telemetry: dict[str, object] = {
        "available": bool(torch.cuda.is_available()),
        "visible_cuda_device": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_index": None,
        "requested_physical_index": requested_physical_index,
        "name": None,
        "uuid": None,
        "prelaunch": {"utilization_percent": None, "free_memory_bytes": None, "total_memory_bytes": None},
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }
    if not torch.cuda.is_available():
        return telemetry
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    telemetry["visible_cuda_device"] = device
    telemetry["name"] = properties.name
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    telemetry["prelaunch"] = {"utilization_percent": None, "free_memory_bytes": free_bytes, "total_memory_bytes": total_bytes}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.splitlines()
        rows = [line.split(", ") for line in output if line.strip()]
        requested = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        selected = requested[device].strip() if len(requested) > device else str(device)
        row = next((item for item in rows if item[0] == selected or item[1] == selected), None)
        if row is not None and len(row) == 6:
            telemetry.update({"physical_index": int(row[0]), "uuid": row[1], "name": row[2]})
            telemetry["prelaunch"] = {
                "utilization_percent": int(row[3]),
                "free_memory_bytes": int(float(row[4]) * 1024 * 1024),
                "total_memory_bytes": int(float(row[5]) * 1024 * 1024),
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return telemetry


def _reset_gpu_peaks() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _record_gpu_peaks(telemetry: dict[str, object]) -> None:
    if torch.cuda.is_available():
        telemetry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        telemetry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_shard(
    rows: Sequence[dict[str, object]],
    encoder: Any,
    config: RunnerConfig,
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
    cohort_sha256: str | None = None,
) -> dict[str, object]:
    """Atomically write one rank-owned shard or leave a nonmergeable failure receipt."""
    output_dir = Path(output_dir)
    audit_cohort_reference_isolation(rows)
    owned = [row for row in rows if rank_owns(str(row.get("image_sha256", "")), rank, world_size)]
    source_shas = [str(row["image_sha256"]) for row in owned]
    if len(source_shas) != len(set(source_shas)):
        raise ValueError("a shard cannot contain duplicate query source SHA-256 values")
    final = output_dir / f"shard_{rank}"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {final}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir, prefix=f".shard_{rank}.tmp."))
    started = time.monotonic()
    try:
        encoder_precision = getattr(encoder, "precision", None)
        if encoder_precision != config.precision:
            raise ValueError(
                f"encoder precision {encoder_precision} does not match runner precision {config.precision}"
            )
        gpu = _gpu_telemetry(config.physical_gpu_index)
        if config.physical_gpu_index is not None and gpu["available"] and gpu["physical_index"] != config.physical_gpu_index:
            raise RuntimeError(f"visible GPU physical index {gpu['physical_index']} does not match requested {config.physical_gpu_index}")
        _reset_gpu_peaks()
        local_config = dataclasses.replace(config, map_dir=temporary / "maps")
        scores = []
        peak_allocated = 0
        peak_reserved = 0
        for row in owned:
            refs = row.get("references")
            if not isinstance(refs, list):
                raise ValueError("cohort query row is missing references")
            query_started = time.monotonic()
            _reset_gpu_peaks()
            score = score_query(row, refs, encoder, local_config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            score["telemetry"]["elapsed_seconds"] = time.monotonic() - query_started
            query_gpu = dict(gpu)
            _record_gpu_peaks(query_gpu)
            score["telemetry"].update(
                peak_allocated_bytes=query_gpu["peak_allocated_bytes"],
                peak_reserved_bytes=query_gpu["peak_reserved_bytes"],
            )
            peak_allocated = max(peak_allocated, int(query_gpu["peak_allocated_bytes"]))
            peak_reserved = max(peak_reserved, int(query_gpu["peak_reserved_bytes"]))
            score["map_archive"] = str(final / "maps" / f"{score['source_sha256']}.npz")
            if not isinstance(score.get("map_artifact"), dict):
                raise ValueError("score query did not produce a map artifact")
            score["map_artifact"] = {**score["map_artifact"], "path": score["map_archive"]}
            scores.append(score)
        scores_text = "".join(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n" for score in scores)
        scores_path = temporary / "scores.jsonl"
        scores_path.write_text(scores_text, encoding="utf-8")
        _fsync_file(scores_path)
        model_digest = str(getattr(encoder, "model_digest", "unknown"))
        checkpoint_digest = str(getattr(encoder, "checkpoint_sha256", "unknown"))
        resolved_cohort_digest = cohort_sha256 or _json_digest(list(rows))
        source_set_digest = hashlib.sha256("\n".join(sorted(source_shas)).encode()).hexdigest()
        gpu.update(peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved)
        maps = [{"source_sha256": score["source_sha256"], **score["map_artifact"]} for score in scores]
        digests = {
            "cohort_sha256": resolved_cohort_digest,
            "checkpoint_sha256": checkpoint_digest,
            "model_sha256": model_digest,
            "runtime_sha256": _runtime_digest(),
            "config_sha256": config.digest,
        }
        receipt: dict[str, object] = {
            "rank": rank,
            "world_size": world_size,
            "source_sha256s": sorted(source_shas),
            "source_set_sha256": source_set_digest,
            "counts": {"owned": len(scores), "failures": 0},
            "digests": digests,
            "config_sha256": digests["config_sha256"],
            "execution_config_sha256": config.execution_digest,
            "execution_config": {"physical_gpu_index": config.physical_gpu_index},
            "runtime_sha256": digests["runtime_sha256"],
            "model_sha256": digests["model_sha256"],
            "checkpoint_sha256": digests["checkpoint_sha256"],
            "cohort_sha256": digests["cohort_sha256"],
            "checkpoint_load_mode": str(getattr(encoder, "checkpoint_load_mode", "undeclared")),
            "scores_sha256": hashlib.sha256(scores_text.encode()).hexdigest(),
            "maps": maps,
            "gpu": gpu,
            "precision": config.precision,
            "elapsed_seconds": time.monotonic() - started,
            "nonmergeable": False,
        }
        if local_config.map_dir is not None and local_config.map_dir.exists():
            _fsync_directory(local_config.map_dir)
        _atomic_json(temporary / "shard_receipt.json", receipt)
        _fsync_directory(temporary)
        os.replace(temporary, final)
        _fsync_directory(output_dir)
        return {"scores": scores, "receipt": receipt}
    except BaseException as error:
        shutil.rmtree(temporary, ignore_errors=True)
        _atomic_json(
            output_dir / f"failed_shard_{rank}.json",
            {
                "rank": rank,
                "world_size": world_size,
                "source_sha256s": sorted(source_shas),
                "failures": 1,
                "nonmergeable": True,
                "config_sha256": config.digest,
                "execution_config_sha256": config.execution_digest,
                "execution_config": {"physical_gpu_index": config.physical_gpu_index},
                "runtime_sha256": _runtime_digest(),
                "model_sha256": str(getattr(encoder, "model_digest", "unknown")),
                "checkpoint_sha256": str(getattr(encoder, "checkpoint_sha256", "unknown")),
                "checkpoint_load_mode": str(getattr(encoder, "checkpoint_load_mode", "undeclared")),
                "cohort_sha256": cohort_sha256 or _json_digest(list(rows)),
                "gpu": _gpu_telemetry(config.physical_gpu_index),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def _load_cohort_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    audit_cohort_reference_isolation(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dense-ref-runner")
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--reference-chunk-size", type=int, default=1)
    parser.add_argument("--reference-image-cache-size", type=int, default=5)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("direct CUDA runner requires exactly one controller-selected visible GPU")
    config = RunnerConfig(
        tile_batch_size=args.tile_batch_size,
        reference_chunk_size=args.reference_chunk_size,
        reference_image_cache_size=args.reference_image_cache_size,
        decode_workers=args.decode_workers,
        physical_gpu_index=args.physical_gpu_index,
        precision=args.precision,
    )
    encoder = FrozenDinoEncoder(args.checkpoint, args.checkpoint_sha256, torch.device("cuda:0"), precision=config.precision)
    run_shard(
        _load_cohort_rows(args.cohort),
        encoder,
        config,
        args.output_dir,
        rank=args.rank,
        world_size=args.world_size,
        cohort_sha256=_sha256(args.cohort),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
