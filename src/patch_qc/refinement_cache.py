from __future__ import annotations

import heapq
import hashlib
import io
import json
import time
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import torch
from PIL import Image

from .inference import Box, tile_boxes
from .refinement_contract import (
    CandidateRow,
    PatchKey,
    SourceRow,
    build_run_receipt,
    canonical_digest,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .refinement_encoding import CandidatePolicy, encode_patch_batch, select_candidate_indices


@dataclass(frozen=True, slots=True)
class ImageMeta:
    path: str
    source_sha256: str
    product: str
    group_id: str
    condition_id: str
    split: str


@dataclass(frozen=True, slots=True)
class _LoadedSource:
    meta: ImageMeta
    rgb: np.ndarray


def _load_source(meta: ImageMeta) -> _LoadedSource:
    payload = Path(meta.path).read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if meta.source_sha256 and actual_digest != meta.source_sha256:
        raise RuntimeError(f"source digest changed for {meta.path}")
    resolved = meta if meta.source_sha256 else replace(meta, source_sha256=actual_digest)
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return _LoadedSource(meta=resolved, rgb=rgb)


def _iter_loaded_sources(
    images: Sequence[ImageMeta],
    *,
    workers: int,
    prefetch_sources: int,
) -> Iterator[_LoadedSource]:
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if prefetch_sources <= 0:
        raise ValueError("prefetch_sources must be positive")
    if workers == 0:
        for meta in images:
            yield _load_source(meta)
        return

    pending: deque[Future[_LoadedSource]] = deque()
    image_iter = iter(images)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(prefetch_sources, len(images))):
            pending.append(executor.submit(_load_source, next(image_iter)))
        while pending:
            loaded = pending.popleft().result()
            try:
                pending.append(executor.submit(_load_source, next(image_iter)))
            except StopIteration:
                pass
            yield loaded


def _stack_patch_batch(rgb: np.ndarray, batch_boxes: Sequence[Box]) -> np.ndarray:
    return np.stack(
        [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in batch_boxes],
        axis=0,
    )


def _iter_patch_batches(
    rgb: np.ndarray,
    boxes: Sequence[Box],
    *,
    batch_size: int,
    executor: ThreadPoolExecutor | None,
    prefetch_batches: int,
) -> Iterator[tuple[list[Box], np.ndarray]]:
    if batch_size <= 0 or prefetch_batches <= 0:
        raise ValueError("batch_size and prefetch_batches must be positive")
    batches = [list(boxes[start : start + batch_size]) for start in range(0, len(boxes), batch_size)]
    if executor is None:
        for batch_boxes in batches:
            yield batch_boxes, _stack_patch_batch(rgb, batch_boxes)
        return

    pending: deque[tuple[list[Box], Future[np.ndarray]]] = deque()
    batch_iter = iter(batches)
    for _ in range(min(prefetch_batches, len(batches))):
        batch_boxes = next(batch_iter)
        pending.append((batch_boxes, executor.submit(_stack_patch_batch, rgb, batch_boxes)))
    while pending:
        batch_boxes, future = pending.popleft()
        patches = future.result()
        try:
            next_boxes = next(batch_iter)
            pending.append((next_boxes, executor.submit(_stack_patch_batch, rgb, next_boxes)))
        except StopIteration:
            pass
        yield batch_boxes, patches


class _ReservoirProtocol(Protocol):
    patch_id: str
    product: str
    condition_id: str
    group_id: str


_ReservoirItem = TypeVar("_ReservoirItem", bound=_ReservoirProtocol)


class LowScoreReservoir(Generic[_ReservoirItem]):
    def __init__(
        self,
        capacity_per_product_condition: int,
        capacity_per_group: int,
        seed: int,
    ) -> None:
        if capacity_per_product_condition <= 0 or capacity_per_group <= 0:
            raise ValueError("reservoir capacities must be positive")
        self.capacity_per_product_condition = capacity_per_product_condition
        self.capacity_per_group = capacity_per_group
        self.seed = seed
        self._group_heaps: dict[
            tuple[str, str, str], list[tuple[int, str, _ReservoirItem]]
        ] = defaultdict(list)
        self._group_patch_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    def _priority(self, row: _ReservoirItem) -> str:
        return canonical_digest({"seed": self.seed, "patch_id": row.patch_id})

    def offer(self, row: _ReservoirItem) -> None:
        group_key = (row.product, row.condition_id, row.group_id)
        if row.patch_id in self._group_patch_ids[group_key]:
            return
        priority = int(self._priority(row), 16)
        entry = (-priority, row.patch_id, row)
        heap = self._group_heaps[group_key]
        if len(heap) < self.capacity_per_group:
            heapq.heappush(heap, entry)
            self._group_patch_ids[group_key].add(row.patch_id)
            return
        worst_priority = -heap[0][0]
        worst_patch_id = heap[0][1]
        if (priority, row.patch_id) >= (worst_priority, worst_patch_id):
            return
        removed = heapq.heapreplace(heap, entry)
        self._group_patch_ids[group_key].remove(removed[1])
        self._group_patch_ids[group_key].add(row.patch_id)

    def rows(self) -> list[_ReservoirItem]:
        by_product_condition: dict[tuple[str, str], list[_ReservoirItem]] = defaultdict(list)
        for (product, condition_id, _), heap in self._group_heaps.items():
            by_product_condition[(product, condition_id)].extend(entry[2] for entry in heap)
        return [
            row
            for key in sorted(by_product_condition)
            for row in sorted(
                by_product_condition[key],
                key=lambda item: (self._priority(item), item.patch_id),
            )[: self.capacity_per_product_condition]
        ]


@dataclass(frozen=True, slots=True)
class _FeatureItem:
    patch_id: str
    source_sha256: str
    source_path: str
    product: str
    group_id: str
    condition_id: str
    split: str
    box: tuple[int, int, int, int]
    logit: float
    probability: float
    foreground_fraction: float
    feature: np.ndarray
    source_label: str = "OK"


def _box_tuple(box: Box) -> tuple[int, int, int, int]:
    return box.x0, box.y0, box.x1, box.y1


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


class SelectiveCacheBuilder:
    def __init__(
        self,
        encoder: Any,
        policy: CandidatePolicy,
        batch_size: int,
        seed: int,
        *,
        device: str = "cpu",
        patch_size: int = 256,
        stride: int = 128,
        reservoir_per_product_condition: int = 4096,
        reservoir_per_group: int = 128,
        shard_size: int = 4096,
        model_digest: str = "",
        feature_schema_digest: str = "",
        workers: int = 0,
        prefetch_sources: int = 2,
        prefetch_batches: int = 2,
    ) -> None:
        if batch_size <= 0 or shard_size <= 0:
            raise ValueError("batch_size and shard_size must be positive")
        if workers < 0 or prefetch_sources <= 0 or prefetch_batches <= 0:
            raise ValueError(
                "workers must be non-negative and prefetch counts must be positive"
            )
        self.encoder = encoder
        self.policy = policy
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.shard_size = shard_size
        self.model_digest = model_digest
        self.feature_schema_digest = feature_schema_digest
        self.reservoir_per_product_condition = reservoir_per_product_condition
        self.reservoir_per_group = reservoir_per_group
        self.workers = workers
        self.prefetch_sources = prefetch_sources
        self.prefetch_batches = prefetch_batches

    def _encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.encoder, "encode"):
            logits, features = self.encoder.encode(rgb_batch)
        else:
            logits, features = encode_patch_batch(self.encoder, rgb_batch, self.device)
        logits = np.asarray(logits, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        if logits.shape != (len(rgb_batch),) or features.ndim != 2 or features.shape[0] != len(rgb_batch):
            raise ValueError("encoder must return one logit and one feature vector per patch")
        if not np.isfinite(logits).all() or not np.isfinite(features).all():
            raise ValueError("encoder output must be finite")
        return logits, features

    def _build_image(
        self,
        meta: ImageMeta,
        rgb: np.ndarray,
        reservoir: LowScoreReservoir[_FeatureItem],
        patch_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[list[_FeatureItem], int]:
        boxes = tile_boxes(rgb.shape[1], rgb.shape[0], self.patch_size, self.stride)
        scalar_probabilities: list[float] = []
        candidate_heap: list[tuple[float, int, _FeatureItem]] = []
        feature_by_index: dict[int, _FeatureItem] = {}
        start = 0
        for batch_boxes, patches in _iter_patch_batches(
            rgb,
            boxes,
            batch_size=self.batch_size,
            executor=patch_executor,
            prefetch_batches=self.prefetch_batches,
        ):
            logits, features = self._encode(patches)
            probabilities = (1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))).astype(np.float32)
            for offset, (box, patch, logit, probability, feature) in enumerate(
                zip(batch_boxes, patches, logits, probabilities, features, strict=True)
            ):
                index = start + offset
                box_values = _box_tuple(box)
                item = _FeatureItem(
                    patch_id=PatchKey(meta.source_sha256, box_values).patch_id,
                    source_sha256=meta.source_sha256,
                    source_path=meta.path,
                    product=meta.product,
                    group_id=meta.group_id,
                    condition_id=meta.condition_id,
                    split=meta.split,
                    box=box_values,
                    logit=float(np.float32(logit)),
                    probability=float(np.float32(probability)),
                    foreground_fraction=float((patch.mean(axis=2) > 8).mean()),
                    feature=np.asarray(feature, dtype=np.float32).copy(),
                )
                scalar_probabilities.append(item.probability)
                heapq.heappush(candidate_heap, (item.probability, -index, item))
                feature_by_index[index] = item
                if len(candidate_heap) > self.policy.maximum_candidates:
                    _, removed_index, _ = heapq.heappop(candidate_heap)
                    feature_by_index.pop(-removed_index, None)
                if item.probability < self.policy.probability_floor:
                    reservoir.offer(item)
            start += len(batch_boxes)

        selected_indices = select_candidate_indices(np.asarray(scalar_probabilities), self.policy).tolist()
        selected = [feature_by_index[index] for index in selected_indices]
        return selected, len(boxes)

    def _write_shards(self, output_dir: Path, rows: Sequence[_FeatureItem]) -> list[dict[str, object]]:
        embedding_dir = output_dir / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        index_rows: list[dict[str, object]] = []
        for start in range(0, len(rows), self.shard_size):
            shard_rows = rows[start : start + self.shard_size]
            features = np.stack([row.feature for row in shard_rows]).astype(np.float16)
            patch_ids = np.asarray([row.patch_id for row in shard_rows], dtype=np.str_)
            temporary = embedding_dir / f".shard_{start:08d}.tmp"
            _write_deterministic_npz(temporary, {"features": features, "patch_ids": patch_ids})
            digest = sha256_file(temporary)
            shard_name = f"shard_{digest[:16]}.npz"
            temporary.replace(embedding_dir / shard_name)
            for row_index, row in enumerate(shard_rows):
                index_rows.append(
                    {
                        "feature_key": row.patch_id,
                        "shard": shard_name,
                        "row_index": row_index,
                        "dtype": "float16",
                        "dimension": int(features.shape[1]),
                    }
                )
        return index_rows

    @staticmethod
    def _manifest_row(row: _FeatureItem) -> dict[str, object]:
        return CandidateRow(
            patch_id=row.patch_id,
            source_sha256=row.source_sha256,
            source_path=row.source_path,
            product=row.product,
            group_id=row.group_id,
            condition_id=row.condition_id,
            split=row.split,
            box=row.box,
            logit=row.logit,
            probability=row.probability,
            foreground_fraction=row.foreground_fraction,
            feature_key=row.patch_id,
            source_label=row.source_label,
        ).to_dict()

    def _validate_sources(self, images: Iterable[ImageMeta]) -> list[ImageMeta]:
        rows: list[ImageMeta] = []
        for meta in images:
            actual = sha256_file(meta.path)
            if meta.source_sha256 and actual != meta.source_sha256:
                raise RuntimeError(f"source digest changed for {meta.path}")
            rows.append(meta if meta.source_sha256 else replace(meta, source_sha256=actual))
        return rows

    def _resume(self, images: list[ImageMeta], output_dir: Path) -> dict[str, object]:
        if list(output_dir.rglob("*.tmp")):
            raise RuntimeError("temporary files prevent cache resume")
        receipt_path = output_dir / "candidate_build_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("candidate build receipt is missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {
            "model": self.model_digest,
            "candidate_policy": canonical_digest(asdict(self.policy)),
            "feature_schema": self.feature_schema_digest,
            "source_set": canonical_digest([asdict(meta) for meta in images]),
        }
        for name, digest in expected.items():
            if receipt.get("digests", {}).get(name) != digest:
                raise RuntimeError(f"{name} digest does not match cache receipt")
        return receipt

    def build(
        self,
        images: Iterable[ImageMeta],
        output_dir: Path,
        *,
        resume: bool = False,
        progress_callback: Callable[[int, int, int, str], None] | None = None,
    ) -> dict[str, object]:
        started = time.perf_counter()
        requested_image_rows = list(images)
        if output_dir.exists():
            if not resume:
                raise FileExistsError(output_dir)
            return self._resume(self._validate_sources(requested_image_rows), output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        device = torch.device(self.device)
        track_cuda = device.type == "cuda" and torch.cuda.is_available()
        if track_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        reservoir: LowScoreReservoir[_FeatureItem] = LowScoreReservoir(
            self.reservoir_per_product_condition,
            self.reservoir_per_group,
            self.seed,
        )
        source_rows: list[dict[str, object]] = []
        image_rows: list[ImageMeta] = []
        candidates: list[_FeatureItem] = []
        total_patches = 0
        patch_workers = 1 if self.workers > 0 else 0
        source_workers = self.workers - patch_workers
        patch_executor = ThreadPoolExecutor(max_workers=patch_workers) if patch_workers else None
        loaded_sources = _iter_loaded_sources(
            requested_image_rows,
            workers=source_workers,
            prefetch_sources=self.prefetch_sources,
        )
        try:
            for completed_sources, loaded in enumerate(loaded_sources, start=1):
                meta = loaded.meta
                rgb = loaded.rgb
                image_rows.append(meta)
                source_rows.append(
                    SourceRow(
                        source_sha256=meta.source_sha256,
                        path=meta.path,
                        product=meta.product,
                        group_id=meta.group_id,
                        condition_id=meta.condition_id,
                        split=meta.split,
                        width=rgb.shape[1],
                        height=rgb.shape[0],
                    ).to_dict()
                )
                selected, patch_count = self._build_image(
                    meta,
                    rgb,
                    reservoir,
                    patch_executor=patch_executor,
                )
                candidates.extend(selected)
                total_patches += patch_count
                if progress_callback is not None:
                    progress_callback(
                        completed_sources,
                        len(requested_image_rows),
                        total_patches,
                        meta.path,
                    )
        finally:
            loaded_sources.close()
            if patch_executor is not None:
                patch_executor.shutdown(wait=True, cancel_futures=True)

        candidate_ids = {row.patch_id for row in candidates}
        reservoir_rows = [row for row in reservoir.rows() if row.patch_id not in candidate_ids]
        saved_rows = [*candidates, *reservoir_rows]
        index_rows = self._write_shards(output_dir, saved_rows)

        source_digest = write_jsonl_atomic(output_dir / "source_manifest.jsonl", source_rows)
        candidate_digest = write_jsonl_atomic(
            output_dir / "candidate_manifest.jsonl",
            [self._manifest_row(row) for row in candidates],
        )
        reservoir_digest = write_jsonl_atomic(
            output_dir / "low_score_reservoir.jsonl",
            [self._manifest_row(row) for row in reservoir_rows],
        )
        index_digest = write_jsonl_atomic(output_dir / "embeddings/index.jsonl", index_rows)
        shard_bytes = sum(path.stat().st_size for path in (output_dir / "embeddings").glob("shard_*.npz"))
        digests = {
            "source_manifest": source_digest,
            "candidate_manifest": candidate_digest,
            "reservoir_manifest": reservoir_digest,
            "embedding_index": index_digest,
            "model": self.model_digest,
            "candidate_policy": canonical_digest(asdict(self.policy)),
            "feature_schema": self.feature_schema_digest,
            "source_set": canonical_digest([asdict(meta) for meta in image_rows]),
        }
        elapsed = time.perf_counter() - started
        peak_cuda_memory = (
            int(torch.cuda.max_memory_allocated(device)) if track_cuda else 0
        )
        receipt = build_run_receipt(
            "candidate-cache-build",
            digests=digests,
            counts={
                "sources": len(image_rows),
                "total_patches": total_patches,
                "candidates": len(candidates),
                "reservoir": len(reservoir_rows),
                "saved_embeddings": len(saved_rows),
                "shard_bytes": shard_bytes,
            },
            metrics={
                "elapsed_seconds": elapsed,
                "samples_per_second": total_patches / elapsed,
                "peak_cuda_memory_bytes": peak_cuda_memory,
            },
            failures=0,
            temporary_files=len(list(output_dir.rglob("*.tmp"))),
            seeds={"cache": self.seed},
        )
        receipt["all_patch_embeddings_saved"] = False
        receipt["source_loading"] = {
            "workers": self.workers,
            "source_workers": source_workers,
            "patch_workers": patch_workers,
            "prefetch_sources": self.prefetch_sources,
            "prefetch_batches": self.prefetch_batches,
            "single_read_decode": True,
        }
        write_json_atomic(output_dir / "candidate_build_receipt.json", receipt)
        return receipt


def _candidate_from_dict(value: dict[str, object]) -> CandidateRow:
    row = dict(value)
    row["box"] = tuple(row["box"])
    return CandidateRow(**row)


def _load_shard_features(shard_dir: Path, receipt: dict[str, object]) -> dict[str, np.ndarray]:
    index_path = shard_dir / "embeddings/index.jsonl"
    if sha256_file(index_path) != dict(receipt["digests"])["embedding_index"]:
        raise RuntimeError(f"embedding index digest mismatch: {shard_dir}")
    index_rows = read_jsonl(index_path)
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in index_rows:
        by_file[str(row["shard"])].append(row)
    features: dict[str, np.ndarray] = {}
    for filename, rows in by_file.items():
        path = shard_dir / "embeddings" / filename
        if not path.is_file() or not filename.startswith(f"shard_{sha256_file(path)[:16]}"):
            raise RuntimeError(f"embedding shard digest mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["features"], dtype=np.float16)
            patch_ids = tuple(str(value) for value in archive["patch_ids"].tolist())
        for row in rows:
            row_index = int(row["row_index"])
            feature_key = str(row["feature_key"])
            if row_index < 0 or row_index >= len(values) or patch_ids[row_index] != feature_key:
                raise RuntimeError(f"embedding row identity mismatch: {path}/{row_index}")
            if feature_key in features:
                raise RuntimeError(f"duplicate feature key across shard index: {feature_key}")
            features[feature_key] = values[row_index].astype(np.float32)
    if len(features) != len(index_rows):
        raise RuntimeError(f"embedding index row count mismatch: {shard_dir}")
    return features


def merge_selective_caches(
    shard_dirs: Sequence[Path],
    output_dir: Path,
    *,
    reservoir_per_product_condition: int,
    reservoir_per_group: int,
    shard_size: int,
    seed: int,
    expected_config_digest: str | None = None,
) -> dict[str, object]:
    import shutil

    started = time.perf_counter()
    shards = [Path(path) for path in shard_dirs]
    if len(shards) < 2 or len({path.resolve() for path in shards}) != len(shards):
        raise ValueError("cache merge requires at least two unique shard directories")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    receipts: list[dict[str, object]] = []
    for shard in shards:
        receipt_path = shard / "candidate_build_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"candidate shard receipt is missing: {shard}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "completed"
            or int(receipt.get("failures", -1)) != 0
            or int(receipt.get("temporary_files", -1)) != 0
        ):
            raise RuntimeError(f"candidate shard is not complete: {shard}")
        receipts.append(receipt)

    shared_digest_names = ("model", "candidate_policy", "feature_schema")
    for name in shared_digest_names:
        values = {str(dict(receipt["digests"])[name]) for receipt in receipts}
        if len(values) != 1:
            raise RuntimeError(f"candidate shard digest mismatch for {name}")
    for name in ("candidate_policy", "candidate_policy_calibration_digest", "config_digest"):
        values = {canonical_digest(receipt[name]) for receipt in receipts}
        if len(values) != 1:
            raise RuntimeError(f"candidate shard contract mismatch for {name}")
    if (
        expected_config_digest is not None
        and receipts[0]["config_digest"] != expected_config_digest
    ):
        raise RuntimeError("candidate shard configuration is outside the requested boundary")
    shard_contracts = [receipt.get("cache_shard") for receipt in receipts]
    if any(value is not None for value in shard_contracts):
        if any(not isinstance(value, dict) for value in shard_contracts):
            raise RuntimeError("candidate shard identity is incomplete")
        counts = {int(value["count"]) for value in shard_contracts}
        indices = {int(value["index"]) for value in shard_contracts}
        if counts != {len(shards)} or indices != set(range(len(shards))):
            raise RuntimeError("candidate shard identities are not complete and disjoint")

    source_rows: list[dict[str, object]] = []
    candidate_items: list[_FeatureItem] = []
    reservoir_items: list[_FeatureItem] = []
    seen_sources: set[str] = set()
    seen_paths: set[str] = set()
    seen_patch_ids: set[str] = set()
    input_receipts: dict[str, str] = {}
    for shard, receipt in zip(shards, receipts, strict=True):
        manifest_paths = {
            "source_manifest": shard / "source_manifest.jsonl",
            "candidate_manifest": shard / "candidate_manifest.jsonl",
            "reservoir_manifest": shard / "low_score_reservoir.jsonl",
        }
        for name, path in manifest_paths.items():
            if sha256_file(path) != dict(receipt["digests"])[name]:
                raise RuntimeError(f"candidate shard manifest digest mismatch: {path}")
        feature_map = _load_shard_features(shard, receipt)
        for source in read_jsonl(manifest_paths["source_manifest"]):
            digest = str(source["source_sha256"])
            path = str(source["path"])
            if digest in seen_sources or path in seen_paths or source.get("source_label") != "OK":
                raise RuntimeError("candidate shard source manifests are not disjoint OK sources")
            seen_sources.add(digest)
            seen_paths.add(path)
            source_rows.append(source)

        def item_from(row_value: dict[str, object]) -> _FeatureItem:
            row = _candidate_from_dict(row_value)
            if row.patch_id in seen_patch_ids:
                raise RuntimeError(f"duplicate patch across candidate shards: {row.patch_id}")
            seen_patch_ids.add(row.patch_id)
            feature = feature_map.get(row.feature_key)
            if feature is None:
                raise RuntimeError(f"missing shard feature for {row.feature_key}")
            return _FeatureItem(
                patch_id=row.patch_id,
                source_sha256=row.source_sha256,
                source_path=row.source_path,
                product=row.product,
                group_id=row.group_id,
                condition_id=row.condition_id,
                split=row.split,
                box=row.box,
                logit=row.logit,
                probability=row.probability,
                foreground_fraction=row.foreground_fraction,
                feature=feature,
                source_label=row.source_label,
            )

        candidate_items.extend(
            item_from(value) for value in read_jsonl(manifest_paths["candidate_manifest"])
        )
        reservoir_items.extend(
            item_from(value) for value in read_jsonl(manifest_paths["reservoir_manifest"])
        )
        shard_identity = receipt.get("cache_shard")
        if isinstance(shard_identity, dict):
            identity = f"shard_{int(shard_identity['index']):04d}"
        else:
            identity = shard.parent.name
        if identity in input_receipts:
            raise RuntimeError(f"duplicate candidate shard receipt identity: {identity}")
        input_receipts[identity] = sha256_file(shard / "candidate_build_receipt.json")

    policy = CandidatePolicy(**dict(receipts[0]["candidate_policy"]))
    reservoir: LowScoreReservoir[_FeatureItem] = LowScoreReservoir(
        reservoir_per_product_condition,
        reservoir_per_group,
        seed,
    )
    for item in reservoir_items:
        reservoir.offer(item)
    candidate_ids = {item.patch_id for item in candidate_items}
    merged_reservoir = [item for item in reservoir.rows() if item.patch_id not in candidate_ids]
    candidate_items.sort(key=lambda item: item.patch_id)
    saved_rows = [*candidate_items, *merged_reservoir]
    source_rows.sort(key=lambda row: (str(row["path"]), str(row["source_sha256"])))

    temporary.mkdir(parents=True, exist_ok=False)
    try:
        writer = SelectiveCacheBuilder(
            encoder=None,
            policy=policy,
            batch_size=1,
            seed=seed,
            shard_size=shard_size,
            reservoir_per_product_condition=reservoir_per_product_condition,
            reservoir_per_group=reservoir_per_group,
            model_digest=str(dict(receipts[0]["digests"])["model"]),
            feature_schema_digest=str(dict(receipts[0]["digests"])["feature_schema"]),
        )
        index_rows = writer._write_shards(temporary, saved_rows)
        source_digest = write_jsonl_atomic(temporary / "source_manifest.jsonl", source_rows)
        candidate_digest = write_jsonl_atomic(
            temporary / "candidate_manifest.jsonl",
            [writer._manifest_row(row) for row in candidate_items],
        )
        reservoir_digest = write_jsonl_atomic(
            temporary / "low_score_reservoir.jsonl",
            [writer._manifest_row(row) for row in merged_reservoir],
        )
        index_digest = write_jsonl_atomic(temporary / "embeddings/index.jsonl", index_rows)
        shard_bytes = sum(
            path.stat().st_size for path in (temporary / "embeddings").glob("shard_*.npz")
        )
        elapsed = time.perf_counter() - started
        receipt = build_run_receipt(
            "candidate-cache-merge",
            digests={
                "source_manifest": source_digest,
                "candidate_manifest": candidate_digest,
                "reservoir_manifest": reservoir_digest,
                "embedding_index": index_digest,
                "model": str(dict(receipts[0]["digests"])["model"]),
                "candidate_policy": str(dict(receipts[0]["digests"])["candidate_policy"]),
                "feature_schema": str(dict(receipts[0]["digests"])["feature_schema"]),
                "source_set": canonical_digest(source_rows),
                **{f"input:{name}": digest for name, digest in sorted(input_receipts.items())},
            },
            counts={
                "sources": len(source_rows),
                "total_patches": sum(int(receipt["counts"]["total_patches"]) for receipt in receipts),
                "candidates": len(candidate_items),
                "reservoir": len(merged_reservoir),
                "saved_embeddings": len(saved_rows),
                "shard_bytes": shard_bytes,
                "input_shards": len(shards),
            },
            metrics={
                "elapsed_seconds": elapsed,
                "samples_per_second": sum(
                    int(receipt["counts"]["total_patches"]) for receipt in receipts
                )
                / elapsed,
                "peak_cuda_memory_bytes": 0,
            },
            failures=0,
            temporary_files=len(list(temporary.rglob("*.tmp"))),
            seeds={"cache": seed},
        )
        receipt["all_patch_embeddings_saved"] = False
        receipt["candidate_policy"] = dict(receipts[0]["candidate_policy"])
        receipt["candidate_policy_calibration"] = receipts[0][
            "candidate_policy_calibration"
        ]
        receipt["candidate_policy_calibration_digest"] = receipts[0][
            "candidate_policy_calibration_digest"
        ]
        receipt["config_digest"] = receipts[0]["config_digest"]
        receipt["merged_shards"] = sorted(input_receipts)
        write_json_atomic(temporary / "candidate_build_receipt.json", receipt)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt
