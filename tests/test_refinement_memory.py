from __future__ import annotations

from collections import Counter

import numpy as np

from patch_qc.refinement_contract import AnchorRow
from patch_qc.refinement_memory import (
    CleanAnchorPolicy,
    NormalMemoryIndex,
    select_clean_anchors,
)


def _anchor(
    index: int,
    *,
    product: str = "P1",
    condition: str = "C1",
    group: str | None = None,
    probability: float = 0.01,
    stability: float = 0.99,
    foreground: float = 0.9,
    roi_relation: str = "inside",
) -> AnchorRow:
    digest = f"{index + 1:064x}"
    return AnchorRow(
        patch_id=f"patch-{index}",
        source_sha256=digest,
        product=product,
        group_id=group or f"G{index % 3}",
        condition_id=condition,
        box=(0, 0, 256, 256),
        probability=probability,
        tta_stability=stability,
        foreground_fraction=foreground,
        feature_key=f"feature-{index}",
        source_label="OK",
        source_kind="clean_anchor",
        roi_relation=roi_relation,
    )


def test_clean_anchor_selection_excludes_suspicious_and_single_lot_rows() -> None:
    rows = [_anchor(index) for index in range(6)]
    rows.append(_anchor(10, probability=0.80))
    rows[-1] = AnchorRow(**{**rows[-1].to_dict(), "patch_id": "high-score-hidden-chip"})
    rows.extend(_anchor(20 + index, product="P2", group="ONLY") for index in range(3))
    features = np.array(
        [[1.0, index * 0.001, 0.0] for index in range(6)]
        + [[1.0, 0.0, 0.0]]
        + [[0.0, 1.0, index * 0.001] for index in range(3)],
        dtype=np.float32,
    )

    selected = select_clean_anchors(
        rows,
        features,
        CleanAnchorPolicy(
            max_probability=0.05,
            min_tta_stability=0.95,
            min_foreground_fraction=0.50,
            min_distinct_neighbor_groups=3,
            max_per_group=2,
        ),
    )

    assert selected
    assert all(row.source_label == "OK" for row in selected)
    assert all(row.probability <= 0.05 for row in selected)
    assert len({row.group_id for row in selected}) >= 3
    assert max(Counter(row.group_id for row in selected).values()) <= 2
    assert "high-score-hidden-chip" not in {row.patch_id for row in selected}
    assert all(row.product == "P1" for row in selected)


def test_clean_anchor_selection_excludes_roi_edges_and_low_foreground() -> None:
    rows = [_anchor(index) for index in range(6)]
    rows[0] = AnchorRow(**{**rows[0].to_dict(), "roi_relation": "edge"})
    rows[1] = AnchorRow(**{**rows[1].to_dict(), "foreground_fraction": 0.1})
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))

    selected = select_clean_anchors(rows, features, CleanAnchorPolicy(0.05, 0.95, 0.5, 3, 2))

    assert rows[0].patch_id not in {row.patch_id for row in selected}
    assert rows[1].patch_id not in {row.patch_id for row in selected}


def _normal_memory_fixture() -> NormalMemoryIndex:
    rows: list[AnchorRow] = []
    features: list[list[float]] = []
    for index in range(15):
        product = "P1" if index < 10 else "P2"
        condition = "C1" if index < 5 or index >= 10 else "C2"
        rows.append(_anchor(index, product=product, condition=condition, group=f"G{index}"))
        features.append([1.0, index / 100.0, 0.1])
    return NormalMemoryIndex(np.asarray(features, dtype=np.float32), rows, chunk_size=3)


def test_normal_memory_uses_condition_then_product_then_global() -> None:
    memory = _normal_memory_fixture()
    query = np.array([1.0, 0.0, 0.1], dtype=np.float32)

    condition = memory.query(query, "P1", "C1")
    product = memory.query(query, "P1", "UNKNOWN")
    global_query = memory.query(query, "UNKNOWN", "UNKNOWN")

    assert condition.fallback_level == "product_condition"
    assert product.fallback_level == "product"
    assert global_query.fallback_level == "global"
    assert len(condition.neighbor_patch_ids) == 5
    assert len(condition.neighbor_group_ids) == 5
    assert np.isfinite(condition.distance)


def test_normal_memory_rejects_feature_dimension_mismatch() -> None:
    memory = _normal_memory_fixture()

    try:
        memory.query(np.ones(4, dtype=np.float32), "P1", "C1")
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("normal memory must reject a mismatched feature dimension")


def test_normal_memory_persists_float16_content_addressed_bank(tmp_path) -> None:
    memory = _normal_memory_fixture()

    receipt = memory.save(tmp_path)
    bank_path = tmp_path / receipt["file"]
    restored = NormalMemoryIndex.load(bank_path, memory.rows, expected_digest=memory.digest)

    assert bank_path.name.startswith("normal_memory_")
    assert bank_path.suffix == ".npz"
    assert receipt["dtype"] == "float16"
    assert receipt["rows"] == len(memory.rows)
    assert restored.digest == memory.digest
    assert restored.query(np.array([1.0, 0.0, 0.1], dtype=np.float32), "P1", "C1").fallback_level == "product_condition"

    try:
        NormalMemoryIndex.load(bank_path, memory.rows, expected_digest="0" * 64)
    except RuntimeError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("normal-memory load must reject a digest mismatch")
