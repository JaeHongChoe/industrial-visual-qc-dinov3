from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from patch_qc.normal_prior_plan import (
    STRICT_EVALUATION_GROUP_IDS,
    build_reference_plan,
    write_reference_plan,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _image(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _manifest_row(
    root: Path,
    group_id: str,
    *,
    split: str = "train",
    condition_id: str = "C_L1-07",
    payload: bytes = b"reference",
) -> dict[str, object]:
    product, object_id = group_id.split("::", 1)
    filename = f"_{condition_id[0]}_Photo-L1-{condition_id[-2:]}.jpg"
    image_path = _image(root / product / object_id / filename, payload)
    return {
        "capture_id": f"2026-08-01/{product}/{object_id}",
        "date": "2026-08-01",
        "group_id": group_id,
        "images": [
            {
                "condition": {
                    "condition_id": condition_id,
                    "family": condition_id[0],
                    "is_standard": True,
                    "view_index": int(condition_id[-2:]),
                },
                "path": image_path,
            }
        ],
        "object_id": object_id,
        "product": product,
        "source_dir": str(Path(image_path).parent),
        "source_label": "OK",
        "split": split,
    }


def _required_block_rows(root: Path) -> list[dict[str, object]]:
    return [
        _manifest_row(root, group_id, payload=f"blocked-{index}".encode())
        for index, group_id in enumerate(sorted(STRICT_EVALUATION_GROUP_IDS))
    ]


def test_build_reference_plan_uses_only_exact_train_or_calibration_groups(
    tmp_path: Path,
) -> None:
    rows = _required_block_rows(tmp_path / "blocked")
    rows.extend(
        _manifest_row(
            tmp_path / "eligible",
            f"PRODUCT_A::G{100 + index}",
            split="calibration" if index % 2 else "train",
            payload=f"eligible-{index}".encode(),
        )
        for index in range(14)
    )
    rows.append(
        _manifest_row(
            tmp_path / "holdout",
            "PRODUCT_A::G999",
            split="holdout",
            payload=b"must-not-be-selected",
        )
    )
    rows.append(
        _manifest_row(
            tmp_path / "other-view",
            "PRODUCT_A::G998",
            condition_id="C_L1-08",
            payload=b"wrong-view",
        )
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, list(reversed(rows)))
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "ng_0085",
                "image_path": "/sealed/ng_0085___C_Photo-L1-07.jpg",
                "image_sha256": "8" * 64,
                "metadata": {
                    "defect_type": "this field must not drive selection",
                    "relative_source_path": "PRODUCT_A/G500(ignored label)/_C_Photo-L1-07.jpg",
                },
            }
        ],
    )

    plan = build_reference_plan(manifest, predictions)

    query = plan["queries"][0]
    assert query["product"] == "PRODUCT_A"
    assert query["condition_id"] == "C_L1-07"
    assert query["status"] == "ready_exact_product_condition"
    assert len(query["candidate_reference_ids"]) == 12
    selected = {
        reference["reference_id"]: reference for reference in plan["references"]
    }
    assert [
        selected[reference_id]["group_id"]
        for reference_id in query["candidate_reference_ids"]
    ] == [f"PRODUCT_A::G{100 + index}" for index in range(12)]
    assert all(reference["split"] in {"train", "calibration"} for reference in selected.values())
    assert not (set(selected) & set(STRICT_EVALUATION_GROUP_IDS))
    assert plan["blocklist"] == sorted(STRICT_EVALUATION_GROUP_IDS)
    assert plan["audit"]["safe_all_split_capture_rows"] == 16


def test_build_reference_plan_fails_open_for_nonstandard_or_uncovered_query(
    tmp_path: Path,
) -> None:
    rows = _required_block_rows(tmp_path / "blocked")
    rows.append(_manifest_row(tmp_path / "eligible", "PRODUCT_A::G100"))
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "nonstandard",
                "image_path": "/sealed/M1-c__037.jpg",
                "image_sha256": "1" * 64,
                "metadata": {"relative_source_path": "PRODUCT_A/G500/M1-c__037.jpg"},
            },
            {
                "id": "uncovered",
                "image_path": "/sealed/_A_Photo-L1-01.jpg",
                "image_sha256": "2" * 64,
                "metadata": {
                    "product": "NO_REFERENCES",
                    "condition_id": "A_L1-01",
                    "defect_type": "must remain irrelevant",
                },
            },
        ],
    )

    plan = build_reference_plan(manifest, predictions)

    assert [(query["id"], query["status"]) for query in plan["queries"]] == [
        ("nonstandard", "fail_open_nonstandard"),
        ("uncovered", "fail_open_no_exact_references"),
    ]
    assert all(not query["candidate_reference_ids"] for query in plan["queries"])


def test_build_reference_plan_requires_every_strict_block_group(tmp_path: Path) -> None:
    rows = _required_block_rows(tmp_path / "blocked")
    rows.pop()
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [])

    with pytest.raises(ValueError, match="missing strict evaluation group_ids"):
        build_reference_plan(manifest, predictions)


def test_reference_plan_is_deterministic_and_can_be_written(tmp_path: Path) -> None:
    rows = _required_block_rows(tmp_path / "blocked")
    reference_path = tmp_path / "eligible" / "PRODUCT_A" / "G100" / "_C_Photo-L1-07.jpg"
    rows.append(
        _manifest_row(
            tmp_path / "eligible",
            "PRODUCT_A::G100",
            payload=b"sealed-reference",
        )
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "ng_0001",
                "image_path": "/sealed/_C_Photo-L1-07.jpg",
                "image_sha256": "a" * 64,
                "metadata": {
                    "relative_source_path": "PRODUCT_A/G999/_C_Photo-L1-07.jpg"
                },
            }
        ],
    )

    first = build_reference_plan(manifest, predictions, max_candidates=4)
    second = build_reference_plan(manifest, predictions, max_candidates=4)
    destination = tmp_path / "plan.json"
    write_reference_plan(first, destination)

    assert first == second
    assert json.loads(destination.read_text(encoding="utf-8")) == first
    assert first["references"][0]["image_path"] == str(reference_path)
    assert first["references"][0]["image_sha256"] == hashlib.sha256(
        b"sealed-reference"
    ).hexdigest()
    with pytest.raises(ValueError, match="between 1 and 64"):
        build_reference_plan(manifest, predictions, max_candidates=65)
