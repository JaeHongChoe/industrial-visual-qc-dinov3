from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from patch_qc.actual_multitask_eval import (
    PINNED_BASE_CHECKPOINT_SHA256,
    FrozenEvalContract,
    evaluate_rows,
    load_eval_rows,
    validate_frozen_settings,
    verify_eval_contract,
    verify_training_receipt,
)
from patch_qc.refinement_contract import canonical_digest

from test_actual_multitask_inference import _BrightnessModel


def _write_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path)


def test_load_eval_rows_keeps_unlabeled_ng_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    image = tmp_path / "ng.jpg"
    _write_rgb(image, np.zeros((2, 3, 3), dtype=np.uint8))
    manifest = tmp_path / "eval.jsonl"
    row = {"id": "ng_0001", "cohort": "ng", "image_path": str(image), "label_path": None}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = load_eval_rows(manifest)

    assert len(rows) == 1
    assert rows[0].identifier == "ng_0001"
    assert rows[0].label_path is None
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "duplicate evaluation id"):
        load_eval_rows(manifest)


def test_evaluate_rows_writes_native_heatmaps_and_original_overlay_comparison(
    tmp_path: Path,
) -> None:
    ok_image = tmp_path / "ok.png"
    ng_image = tmp_path / "ng.png"
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[:, 2] = 255
    _write_rgb(ok_image, pixels)
    _write_rgb(ng_image, pixels)
    label = tmp_path / "ng.json"
    label.write_text(
        json.dumps(
            {
                "imageWidth": 3,
                "imageHeight": 2,
                "shapes": [
                    {"label": "Bow", "shape_type": "polygon", "points": [[1, 0], [2, 0], [2, 1]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"id": "ok_0001", "cohort": "ok", "image_path": str(ok_image)}),
                json.dumps(
                    {
                        "id": "ng_0001",
                        "cohort": "ng",
                        "image_path": str(ng_image),
                        "label_path": str(label),
                        "defect_type": "spot",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_eval_rows(manifest)

    predictions, artifacts = evaluate_rows(
        rows,
        _BrightnessModel(),
        output_dir=tmp_path / "output",
        patch_size=2,
        stride=1,
        batch_size=2,
        device="cpu",
        mask_threshold=0.5,
        patch_gate_threshold=0.5,
        use_amp=False,
    )

    assert [row["id"] for row in predictions] == ["ok_0001", "ng_0001"]
    assert predictions[0]["label_status"] == "ok_unlabeled"
    assert predictions[1]["label_status"] == "localized"
    assert predictions[1]["gated_metrics"]["ground_truth_hit"] is True
    assert len(artifacts) == 7
    raw = np.asarray(Image.open(tmp_path / "output/raw_heatmaps/ok/ok_0001.png"))
    gated = np.asarray(Image.open(tmp_path / "output/gated_heatmaps/ok/ok_0001.png"))
    comparison = np.asarray(Image.open(tmp_path / "output/comparisons/ok/ok_0001.jpg"))
    assert raw.shape == (2, 3)
    assert gated.shape == (2, 3)
    assert comparison.shape == (2, 6, 3)
    np.testing.assert_array_equal(raw, 128)
    np.testing.assert_array_equal(gated[:, 0], 0)
    assert np.abs(comparison[:, :3].astype(np.int16) - pixels.astype(np.int16)).max() <= 12


def test_verify_eval_contract_pins_manifest_counts_dimensions_and_ok_identity(
    tmp_path: Path,
) -> None:
    ok_image = tmp_path / "ok.png"
    ng_image = tmp_path / "ng.png"
    _write_rgb(ok_image, np.zeros((2, 3, 3), dtype=np.uint8))
    _write_rgb(ng_image, np.full((2, 3, 3), 255, dtype=np.uint8))
    label = tmp_path / "ng.json"
    label.write_text(
        json.dumps(
            {
                "imageWidth": 3,
                "imageHeight": 2,
                "shapes": [
                    {"label": "Bow", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ok_0001",
                        "cohort": "ok",
                        "image_path": str(ok_image),
                        "product": "PRODUCT_B",
                        "split": "holdout",
                        "date": "2026-08-12",
                        "lot": "PRODUCT_G-G397",
                    }
                ),
                json.dumps(
                    {
                        "id": "ng_0001",
                        "cohort": "ng",
                        "image_path": str(ng_image),
                        "label_path": str(label),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    contract = FrozenEvalContract(
        ok_count=1,
        ng_count=1,
        labeled_ng_count=1,
        image_shape=(2, 3),
        ok_product="PRODUCT_B",
        ok_split="holdout",
        ok_date="2026-08-12",
        ok_lot="PRODUCT_G-G397",
    )

    audit = verify_eval_contract(
        load_eval_rows(manifest),
        manifest_path=manifest,
        expected_manifest_sha256=manifest_sha,
        contract=contract,
    )

    assert audit["counts"] == {"ok": 1, "ng": 1, "labeled_ng": 1, "unlabeled_ng": 0}
    assert audit["unique_image_sha256"] == 2
    with np.testing.assert_raises_regex(ValueError, "manifest SHA-256 mismatch"):
        verify_eval_contract(
            load_eval_rows(manifest),
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            contract=contract,
        )


def test_validate_frozen_settings_rejects_non_256_or_non_half_threshold() -> None:
    config = {
        "data": {"patch_size": 256},
        "evaluation": {"mask_threshold": 0.5},
        "inputs": {"expected_base_checkpoint_sha256": PINNED_BASE_CHECKPOINT_SHA256},
    }
    metadata = {"config_digest": canonical_digest(config)}
    settings = validate_frozen_settings(config, metadata)
    assert settings == {"patch_size": 256, "stride": 128, "mask_threshold": 0.5}

    config["data"]["patch_size"] = 128
    metadata["config_digest"] = canonical_digest(config)
    with np.testing.assert_raises_regex(ValueError, "patch_size must remain frozen at 256"):
        validate_frozen_settings(config, metadata)
    config["data"]["patch_size"] = 256
    config["evaluation"]["mask_threshold"] = 0.7
    metadata["config_digest"] = canonical_digest(config)
    with np.testing.assert_raises_regex(ValueError, "mask threshold must remain frozen at 0.5"):
        validate_frozen_settings(config, metadata)


def test_evaluate_rows_rejects_image_changed_after_contract_audit(tmp_path: Path) -> None:
    image = tmp_path / "ok.png"
    _write_rgb(image, np.zeros((2, 3, 3), dtype=np.uint8))
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "ok_0001",
                "cohort": "ok",
                "image_path": str(image),
                "product": "PRODUCT_B",
                "split": "holdout",
                "date": "2026-08-12",
                "lot": "PRODUCT_G-G397",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = FrozenEvalContract(1, 0, 0, (2, 3), "PRODUCT_B", "holdout", "2026-08-12", "PRODUCT_G-G397")
    audit = verify_eval_contract(
        load_eval_rows(manifest),
        manifest_path=manifest,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        contract=contract,
    )
    _write_rgb(image, np.full((2, 3, 3), 255, dtype=np.uint8))

    with np.testing.assert_raises_regex(RuntimeError, "changed after contract preflight"):
        evaluate_rows(
            load_eval_rows(manifest),
            _BrightnessModel(),
            output_dir=tmp_path / "output",
            patch_size=2,
            stride=1,
            batch_size=1,
            device="cpu",
            mask_threshold=0.5,
            patch_gate_threshold=0.5,
            use_amp=False,
            sealed_inputs=audit,
        )


def test_verify_training_receipt_requires_completed_quality_gate_and_checkpoint_entry(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "run_receipt.json"
    value = {
        "status": "completed",
        "quality_gate": {"passed": True, "failures": []},
        "artifacts": {"checkpoint_epoch_001.pt": "d" * 64},
        "digests": {"base_checkpoint_sha256": "a" * 64},
    }
    receipt.write_text(json.dumps(value), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()

    verified = verify_training_receipt(
        receipt,
        expected_receipt_sha256=receipt_sha,
        expected_multitask_checkpoint_sha256="d" * 64,
        expected_base_checkpoint_sha256="a" * 64,
    )
    assert verified["quality_gate"]["passed"] is True
    value["quality_gate"]["passed"] = False
    receipt.write_text(json.dumps(value), encoding="utf-8")
    failed_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with np.testing.assert_raises_regex(ValueError, "quality gate"):
        verify_training_receipt(
            receipt,
            expected_receipt_sha256=failed_sha,
            expected_multitask_checkpoint_sha256="d" * 64,
            expected_base_checkpoint_sha256="a" * 64,
        )
