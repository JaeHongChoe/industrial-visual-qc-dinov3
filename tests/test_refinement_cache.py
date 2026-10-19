from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import patch_qc.refinement_cache as refinement_cache
from patch_qc.inference import Box
from patch_qc.refinement_cache import (
    ImageMeta,
    LowScoreReservoir,
    SelectiveCacheBuilder,
    _iter_loaded_sources,
    _iter_patch_batches,
    _load_source,
    merge_selective_caches,
)
from patch_qc.refinement_contract import read_jsonl, sha256_file, write_json_atomic
from patch_qc.refinement_encoding import CandidatePolicy


@dataclass(frozen=True)
class _ReservoirOffer:
    patch_id: str
    product: str
    condition_id: str
    group_id: str
    probability: float


class _FakeEncoder:
    def __init__(self, probabilities: list[float], dimension: int = 8) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.dimension = dimension
        self.offset = 0

    def encode(self, rgb_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(rgb_batch)
        probabilities = self.probabilities[self.offset : self.offset + count]
        self.offset += count
        logits = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
        rows = np.arange(self.offset - count, self.offset, dtype=np.float32)[:, None]
        features = np.repeat(rows, self.dimension, axis=1)
        return logits, features


class _CountingPriorityReservoir(LowScoreReservoir[_ReservoirOffer]):
    def __init__(self) -> None:
        super().__init__(capacity_per_product_condition=4, capacity_per_group=2, seed=17)
        self.priority_calls = 0

    def _priority(self, row: _ReservoirOffer) -> str:
        self.priority_calls += 1
        return super()._priority(row)


def _write_image(path: Path, value: int = 100) -> Path:
    Image.fromarray(np.full((384, 384, 3), value, dtype=np.uint8)).save(path)
    return path


def _meta(path: Path) -> ImageMeta:
    return ImageMeta(
        path=str(path),
        source_sha256=sha256_file(path),
        product="P1",
        group_id="G1",
        condition_id="C",
        split="train",
    )


def _builder() -> SelectiveCacheBuilder:
    return SelectiveCacheBuilder(
        encoder=_FakeEncoder([0.01, 0.02, 0.80, 0.90]),
        policy=CandidatePolicy(0.5, 1, 2),
        batch_size=4,
        seed=17,
        model_digest="1" * 64,
        feature_schema_digest="2" * 64,
    )


def test_low_score_reservoir_caps_group_and_product_condition() -> None:
    reservoir = LowScoreReservoir(capacity_per_product_condition=4, capacity_per_group=2, seed=17)
    for index in range(20):
        reservoir.offer(
            _ReservoirOffer(
                patch_id=f"patch-{index}",
                product="P1",
                condition_id="C1",
                group_id=f"G{index // 5}",
                probability=index / 100,
            )
        )

    rows = reservoir.rows()

    assert len(rows) == 4
    assert max(Counter(row.group_id for row in rows).values()) <= 2
    assert [row.patch_id for row in rows] == [row.patch_id for row in reservoir.rows()]


def test_low_score_reservoir_offer_is_bounded_per_new_row() -> None:
    reservoir = _CountingPriorityReservoir()
    for index in range(100):
        reservoir.offer(
            _ReservoirOffer(
                patch_id=f"patch-{index}",
                product="P1",
                condition_id="C1",
                group_id="G1",
                probability=index / 100,
            )
        )

    assert reservoir.priority_calls <= 110


def test_selective_cache_saves_only_candidates_and_reservoir(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "source.png")
    output = tmp_path / "cache"

    receipt = _builder().build([_meta(image)], output)

    candidates = read_jsonl(output / "candidate_manifest.jsonl")
    reservoir = read_jsonl(output / "low_score_reservoir.jsonl")
    index = read_jsonl(output / "embeddings/index.jsonl")
    assert len(candidates) == 2
    assert len(reservoir) == 2
    assert len(index) == 4
    assert receipt["all_patch_embeddings_saved"] is False
    assert receipt["counts"]["total_patches"] == 4
    assert receipt["counts"]["saved_embeddings"] == 4
    assert receipt["metrics"]["elapsed_seconds"] > 0
    assert receipt["metrics"]["samples_per_second"] > 0
    assert receipt["metrics"]["peak_cuda_memory_bytes"] >= 0
    assert list((output / "embeddings").glob("shard_*.npz"))
    assert not list(output.rglob("*.tmp"))


def test_cache_builder_reports_each_completed_source_with_cumulative_patches(
    tmp_path: Path,
) -> None:
    images = [
        _meta(_write_image(tmp_path / f"source-{index}.png", value=100 + index))
        for index in range(2)
    ]
    builder = SelectiveCacheBuilder(
        encoder=_FakeEncoder([0.01, 0.02, 0.80, 0.90] * 2),
        policy=CandidatePolicy(0.5, 1, 2),
        batch_size=4,
        seed=17,
        model_digest="1" * 64,
        feature_schema_digest="2" * 64,
    )
    updates: list[tuple[int, int, int, str]] = []

    builder.build(
        images,
        tmp_path / "cache",
        progress_callback=lambda completed, total, patches, path: updates.append(
            (completed, total, patches, path)
        ),
    )

    assert updates == [
        (1, 2, 4, str(images[0].path)),
        (2, 2, 8, str(images[1].path)),
    ]


def test_cache_hashes_unsealed_source_once_and_records_the_digest(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "source.png")
    meta = ImageMeta(
        path=str(image),
        source_sha256="",
        product="P1",
        group_id="G1",
        condition_id="C",
        split="train",
    )

    _builder().build([meta], tmp_path / "cache")

    [source] = read_jsonl(tmp_path / "cache/source_manifest.jsonl")
    assert source["source_sha256"] == sha256_file(image)


def test_loaded_source_reads_bytes_once_and_fills_digest(tmp_path: Path, monkeypatch) -> None:
    image = _write_image(tmp_path / "source.png")
    meta = replace(_meta(image), source_sha256="")
    expected_digest = sha256_file(image)
    real_open = Path.open
    reads = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal reads
        if path == image and (not args or "r" in str(args[0])):
            reads += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    loaded = _load_source(meta)

    assert reads == 1
    assert loaded.meta.source_sha256 == expected_digest
    assert loaded.rgb.shape == (384, 384, 3)


def test_loaded_source_rejects_changed_bytes(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "source.png", value=100)
    meta = _meta(image)
    _write_image(image, value=101)

    with pytest.raises(RuntimeError, match="source digest changed"):
        _load_source(meta)


def test_loaded_sources_preserve_order_and_bound_prefetch(tmp_path: Path, monkeypatch) -> None:
    metas = [
        replace(_meta(_write_image(tmp_path / f"source-{index}.png")), group_id=f"G{index}")
        for index in range(3)
    ]
    first_started = threading.Event()
    second_started = threading.Event()
    third_started = threading.Event()
    release_first = threading.Event()
    results = []

    def controlled_load(meta: ImageMeta):
        index = int(Path(meta.path).stem.rsplit("-", 1)[1])
        (first_started, second_started, third_started)[index].set()
        if index == 0:
            assert release_first.wait(timeout=2)
        return refinement_cache._LoadedSource(
            meta=meta,
            rgb=np.full((4, 4, 3), index, dtype=np.uint8),
        )

    monkeypatch.setattr(refinement_cache, "_load_source", controlled_load)

    consumer = threading.Thread(
        target=lambda: results.extend(
            _iter_loaded_sources(metas, workers=3, prefetch_sources=2)
        )
    )
    consumer.start()
    assert first_started.wait(timeout=1)
    assert second_started.wait(timeout=1)
    assert not third_started.wait(timeout=0.1)
    release_first.set()
    consumer.join(timeout=2)

    assert not consumer.is_alive()
    assert third_started.is_set()
    assert [row.meta.path for row in results] == [row.path for row in metas]


def test_patch_batches_preserve_order_and_bound_prefetch(monkeypatch) -> None:
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    boxes = [Box(index, 0, index + 1, 1) for index in range(3)]
    first_started = threading.Event()
    second_started = threading.Event()
    third_started = threading.Event()
    release_first = threading.Event()
    results = []

    def controlled_stack(_rgb: np.ndarray, batch_boxes: list[Box]):
        index = batch_boxes[0].x0
        (first_started, second_started, third_started)[index].set()
        if index == 0:
            assert release_first.wait(timeout=2)
        return np.full((1, 1, 1, 3), index, dtype=np.uint8)

    monkeypatch.setattr(refinement_cache, "_stack_patch_batch", controlled_stack)

    def consume() -> None:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results.extend(
                _iter_patch_batches(
                    rgb,
                    boxes,
                    batch_size=1,
                    executor=executor,
                    prefetch_batches=2,
                )
            )

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert first_started.wait(timeout=1)
    assert second_started.wait(timeout=1)
    assert not third_started.wait(timeout=0.1)
    release_first.set()
    consumer.join(timeout=2)

    assert not consumer.is_alive()
    assert third_started.is_set()
    assert [batch_boxes[0].x0 for batch_boxes, _ in results] == [0, 1, 2]
    assert [int(patches[0, 0, 0, 0]) for _, patches in results] == [0, 1, 2]


def test_prefetch_build_matches_single_worker_artifacts(tmp_path: Path) -> None:
    images = [
        _meta(_write_image(tmp_path / f"source-{index}.png", value=100 + index))
        for index in range(3)
    ]
    probabilities = [0.01, 0.02, 0.80, 0.90] * len(images)

    def build(output: Path, workers: int):
        builder = SelectiveCacheBuilder(
            encoder=_FakeEncoder(probabilities),
            policy=CandidatePolicy(0.5, 1, 2),
            batch_size=4,
            seed=17,
            model_digest="1" * 64,
            feature_schema_digest="2" * 64,
            workers=workers,
            prefetch_sources=2,
            prefetch_batches=2,
        )
        return builder.build(images, output)

    single_output = tmp_path / "single"
    prefetch_output = tmp_path / "prefetch"
    single_receipt = build(single_output, workers=0)
    prefetch_receipt = build(prefetch_output, workers=2)

    for relative in (
        "source_manifest.jsonl",
        "candidate_manifest.jsonl",
        "low_score_reservoir.jsonl",
        "embeddings/index.jsonl",
    ):
        assert (single_output / relative).read_bytes() == (prefetch_output / relative).read_bytes()
    for name in (
        "source_manifest",
        "candidate_manifest",
        "reservoir_manifest",
        "embedding_index",
        "model",
        "candidate_policy",
        "feature_schema",
        "source_set",
    ):
        assert single_receipt["digests"][name] == prefetch_receipt["digests"][name]
    assert single_receipt["counts"]["total_patches"] == 12
    assert prefetch_receipt["counts"]["total_patches"] == 12
    assert single_receipt["source_loading"] == {
        "workers": 0,
        "source_workers": 0,
        "patch_workers": 0,
        "prefetch_sources": 2,
        "prefetch_batches": 2,
        "single_read_decode": True,
    }
    assert prefetch_receipt["source_loading"] == {
        "workers": 2,
        "source_workers": 1,
        "patch_workers": 1,
        "prefetch_sources": 2,
        "prefetch_batches": 2,
        "single_read_decode": True,
    }


def test_cache_resume_rejects_changed_source_bytes(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "source.png", value=100)
    output = tmp_path / "cache"
    original_meta = _meta(image)
    _builder().build([original_meta], output)
    _write_image(image, value=101)

    with pytest.raises(RuntimeError, match="source digest"):
        _builder().build([original_meta], output, resume=True)


def test_cache_refuses_partial_temporary_files_on_resume(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "source.png")
    output = tmp_path / "cache"
    _builder().build([_meta(image)], output)
    (output / ".partial.tmp").write_text("incomplete", encoding="utf-8")

    with pytest.raises(RuntimeError, match="temporary"):
        _builder().build([_meta(image)], output, resume=True)


def test_merge_selective_caches_rebuilds_global_reservoir_and_feature_index(
    tmp_path: Path,
) -> None:
    shard_dirs: list[Path] = []
    for index, value in enumerate((100, 101)):
        image = _write_image(tmp_path / f"source-{index}.png", value=value)
        shard = tmp_path / f"run-{index}" / "candidate_cache"
        builder = _builder()
        receipt = builder.build([_meta(image)], shard)
        receipt["candidate_policy"] = asdict(builder.policy)
        receipt["candidate_policy_calibration"] = []
        receipt["candidate_policy_calibration_digest"] = "3" * 64
        receipt["config_digest"] = "4" * 64
        write_json_atomic(shard / "candidate_build_receipt.json", receipt)
        shard_dirs.append(shard)

    output = tmp_path / "merged"
    receipt = merge_selective_caches(
        shard_dirs,
        output,
        reservoir_per_product_condition=4,
        reservoir_per_group=2,
        shard_size=4,
        seed=17,
    )

    candidates = read_jsonl(output / "candidate_manifest.jsonl")
    reservoir = read_jsonl(output / "low_score_reservoir.jsonl")
    feature_index = read_jsonl(output / "embeddings/index.jsonl")
    sources = read_jsonl(output / "source_manifest.jsonl")
    assert receipt["status"] == "completed"
    assert receipt["counts"] == {
        "sources": 2,
        "total_patches": 8,
        "candidates": 4,
        "reservoir": 2,
        "saved_embeddings": 6,
        "shard_bytes": receipt["counts"]["shard_bytes"],
        "input_shards": 2,
    }
    assert len(sources) == 2
    assert len(candidates) == 4
    assert len(reservoir) == 2
    assert len([name for name in receipt["digests"] if name.startswith("input:")]) == 2
    assert len(receipt["merged_shards"]) == 2
    assert {row["feature_key"] for row in feature_index} == {
        row["feature_key"] for row in [*candidates, *reservoir]
    }
    assert not list(output.rglob("*.tmp"))
