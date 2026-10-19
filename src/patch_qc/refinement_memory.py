from __future__ import annotations

import hashlib
import io
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .refinement_contract import AnchorRow, canonical_digest, sha256_file, write_json_atomic


@dataclass(frozen=True, slots=True)
class CleanAnchorPolicy:
    max_probability: float
    min_tta_stability: float
    min_foreground_fraction: float
    min_distinct_neighbor_groups: int
    max_per_group: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_probability <= 1.0:
            raise ValueError("max_probability must be between 0 and 1")
        if not 0.0 <= self.min_tta_stability <= 1.0:
            raise ValueError("min_tta_stability must be between 0 and 1")
        if not 0.0 <= self.min_foreground_fraction <= 1.0:
            raise ValueError("min_foreground_fraction must be between 0 and 1")
        if self.min_distinct_neighbor_groups <= 0 or self.max_per_group <= 0:
            raise ValueError("group constraints must be positive")


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("features must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def select_clean_anchors(
    rows: Sequence[AnchorRow],
    features: np.ndarray,
    policy: CleanAnchorPolicy,
) -> list[AnchorRow]:
    normalized = _normalize_rows(features)
    if len(rows) != len(normalized):
        raise ValueError("anchor row and feature counts must match")
    eligible_indices = [
        index
        for index, row in enumerate(rows)
        if row.source_label == "OK"
        and row.source_kind == "clean_anchor"
        and row.probability <= policy.max_probability
        and row.tta_stability >= policy.min_tta_stability
        and row.foreground_fraction >= policy.min_foreground_fraction
        and row.roi_relation not in {"edge", "outside"}
    ]
    by_condition: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in eligible_indices:
        row = rows[index]
        by_condition[(row.product, row.condition_id)].append(index)

    dense_indices: list[int] = []
    for indices in by_condition.values():
        neighbor_count = min(len(indices), max(8, policy.min_distinct_neighbor_groups * 2))
        for index in indices:
            similarities = normalized[indices] @ normalized[index]
            order = sorted(
                range(len(indices)),
                key=lambda position: (
                    -float(similarities[position]),
                    rows[indices[position]].source_sha256,
                    rows[indices[position]].box,
                ),
            )
            neighbor_indices = [indices[position] for position in order[:neighbor_count]]
            if len({rows[position].group_id for position in neighbor_indices}) >= policy.min_distinct_neighbor_groups:
                dense_indices.append(index)

    per_group: dict[tuple[str, str, str], int] = defaultdict(int)
    selected: list[AnchorRow] = []
    for index in sorted(
        dense_indices,
        key=lambda position: (
            rows[position].product,
            rows[position].condition_id,
            rows[position].source_sha256,
            rows[position].box,
        ),
    ):
        row = rows[index]
        group_key = (row.product, row.condition_id, row.group_id)
        if per_group[group_key] >= policy.max_per_group:
            continue
        selected.append(row)
        per_group[group_key] += 1
    return selected


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    distance: float
    neighbor_patch_ids: tuple[str, ...]
    neighbor_group_ids: tuple[str, ...]
    fallback_level: str


class NormalMemoryIndex:
    def __init__(
        self,
        features: np.ndarray,
        rows: Sequence[AnchorRow],
        chunk_size: int = 4096,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        normalized = _normalize_rows(features)
        if len(rows) != len(normalized) or not rows:
            raise ValueError("normal memory requires matching non-empty rows and features")
        if any(row.source_label != "OK" or row.source_kind != "clean_anchor" for row in rows):
            raise ValueError("normal memory accepts only selected OK clean anchors")
        self.rows = tuple(rows)
        self.features = normalized.astype(np.float16)
        self.feature_dimension = int(normalized.shape[1])
        self.chunk_size = chunk_size
        self._product_condition: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._product: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self._product_condition[(row.product, row.condition_id)].append(index)
            self._product[row.product].append(index)
        feature_digest = hashlib.sha256(np.ascontiguousarray(self.features).tobytes()).hexdigest()
        self.digest = canonical_digest(
            {
                "feature_digest": feature_digest,
                "patch_ids": [row.patch_id for row in self.rows],
                "dtype": "float16",
                "dimension": self.feature_dimension,
            }
        )

    def _bank(self, product: str, condition_id: str, k: int) -> tuple[list[int], str]:
        condition = self._product_condition.get((product, condition_id), [])
        if len(condition) >= k:
            return condition, "product_condition"
        product_rows = self._product.get(product, [])
        if len(product_rows) >= k:
            return product_rows, "product"
        return list(range(len(self.rows))), "global"

    def query(
        self,
        feature: np.ndarray,
        product: str,
        condition_id: str,
        k: int = 5,
    ) -> MemoryQuery:
        query = np.asarray(feature, dtype=np.float32)
        if query.shape != (self.feature_dimension,):
            raise ValueError(
                f"feature dimension must be {self.feature_dimension}, got {query.shape}"
            )
        if not np.isfinite(query).all():
            raise ValueError("query feature must be finite")
        if k <= 0:
            raise ValueError("k must be positive")
        query /= max(float(np.linalg.norm(query)), 1e-12)
        indices, fallback = self._bank(product, condition_id, k)
        if not indices:
            raise RuntimeError("normal memory is empty")
        scored: list[tuple[float, str, int]] = []
        for start in range(0, len(indices), self.chunk_size):
            chunk_indices = indices[start : start + self.chunk_size]
            chunk = self.features[chunk_indices].astype(np.float32)
            distances = 1.0 - np.clip(chunk @ query, -1.0, 1.0)
            scored.extend(
                (float(distance), self.rows[index].patch_id, index)
                for distance, index in zip(distances, chunk_indices, strict=True)
            )
        selected = sorted(scored, key=lambda item: (item[0], item[1]))[: min(k, len(scored))]
        return MemoryQuery(
            distance=float(np.mean([item[0] for item in selected])),
            neighbor_patch_ids=tuple(self.rows[item[2]].patch_id for item in selected),
            neighbor_group_ids=tuple(self.rows[item[2]].group_id for item in selected),
            fallback_level=fallback,
        )

    @staticmethod
    def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    def save(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_dir / ".normal_memory.tmp"
        try:
            self._write_npz(
                temporary,
                {
                    "features": self.features,
                    "patch_ids": np.asarray([row.patch_id for row in self.rows], dtype=np.str_),
                },
            )
            file_digest = sha256_file(temporary)
            destination = output_dir / f"normal_memory_{file_digest[:16]}.npz"
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        receipt = {
            "format_version": 1,
            "logical_digest": self.digest,
            "file_sha256": file_digest,
            "file": destination.name,
            "dtype": "float16",
            "dimension": self.feature_dimension,
            "rows": len(self.rows),
        }
        write_json_atomic(output_dir / "normal_memory_receipt.json", receipt)
        return receipt

    @classmethod
    def load(
        cls,
        path: Path,
        rows: Sequence[AnchorRow],
        *,
        expected_digest: str | None = None,
        chunk_size: int = 4096,
    ) -> "NormalMemoryIndex":
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float16)
            patch_ids = tuple(str(value) for value in archive["patch_ids"].tolist())
        expected_patch_ids = tuple(row.patch_id for row in rows)
        if patch_ids != expected_patch_ids:
            raise RuntimeError("normal-memory row IDs do not match the supplied anchors")
        memory = cls(features.astype(np.float32), rows, chunk_size=chunk_size)
        if expected_digest is not None and memory.digest != expected_digest:
            raise RuntimeError("normal-memory logical digest mismatch")
        return memory
