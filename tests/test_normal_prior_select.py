from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from patch_qc.normal_prior_select import select_visual_references


def _write(path: Path, pixels: np.ndarray) -> str:
    Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _textured() -> np.ndarray:
    value = np.zeros((128, 192), dtype=np.float32)
    cv2.circle(value, (55, 42), 15, 1.0, -1)
    cv2.rectangle(value, (110, 65), (155, 94), 0.55, -1)
    cv2.line(value, (30, 105), (170, 20), 0.8, 5)
    return value


def _shift(value: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(value, matrix, value.shape[::-1])


def test_select_visual_references_keeps_three_registerable_candidates(tmp_path: Path) -> None:
    query_path = tmp_path / "query.png"
    base = _textured()
    query_sha = _write(query_path, base)
    references = []
    candidate_ids = []
    for index, pixels in enumerate(
        [base, _shift(base, 2, 0), _shift(base, -2, 1), np.zeros_like(base)], start=1
    ):
        path = tmp_path / f"ref_{index}.png"
        digest = _write(path, pixels)
        reference_id = f"ref_{index}"
        candidate_ids.append(reference_id)
        references.append(
            {
                "reference_id": reference_id,
                "image_path": str(path),
                "image_sha256": digest,
                "product": "PRODUCT_A",
                "condition_id": "C_L1-07",
                "capture_id": f"capture_{index}",
                "group_id": f"PRODUCT_A::G{index:04d}",
            }
        )
    plan = {
        "schema_version": "normal_prior_reference_plan_v1",
        "queries": [
            {
                "id": "ng_0001",
                "image_path": str(query_path),
                "image_sha256": query_sha,
                "product": "PRODUCT_A",
                "condition_id": "C_L1-07",
                "status": "ready_exact_product_condition",
                "candidate_reference_ids": candidate_ids,
            }
        ],
        "references": references,
    }

    selected = select_visual_references(plan, required_references=3, thumbnail_width=192)

    assert selected["schema_version"] == "normal_prior_selected_v1"
    assert selected["queries"][0]["status"] == "ready_visual_nearest_3"
    assert set(selected["queries"][0]["selected_reference_ids"]) == {"ref_1", "ref_2", "ref_3"}
    assert {row["reference_id"] for row in selected["references"]} == {"ref_1", "ref_2", "ref_3"}
    assert selected["audit"] == {
        "fail_open_queries": 0,
        "ready_queries": 1,
        "selected_unique_references": 3,
        "total_queries": 1,
    }


def test_select_visual_references_fails_open_when_fewer_than_three_align(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "query.png"
    query_sha = _write(query_path, _textured())
    flat_path = tmp_path / "flat.png"
    flat_sha = _write(flat_path, np.zeros_like(_textured()))
    plan = {
        "schema_version": "normal_prior_reference_plan_v1",
        "queries": [
            {
                "id": "ng_0001",
                "image_path": str(query_path),
                "image_sha256": query_sha,
                "product": "PRODUCT_A",
                "condition_id": "C_L1-07",
                "status": "ready_exact_product_condition",
                "candidate_reference_ids": ["flat"],
            }
        ],
        "references": [
            {
                "reference_id": "flat",
                "image_path": str(flat_path),
                "image_sha256": flat_sha,
                "product": "PRODUCT_A",
                "condition_id": "C_L1-07",
                "capture_id": "capture",
                "group_id": "PRODUCT_A::G0001",
            }
        ],
    }

    selected = select_visual_references(plan, required_references=3, thumbnail_width=192)

    assert selected["queries"][0]["status"] == "fail_open_registration_0_of_3"
    assert selected["queries"][0]["selected_reference_ids"] == []
    assert selected["references"] == []
    assert selected["audit"]["fail_open_queries"] == 1
