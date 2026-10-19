from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from patch_qc.raw_primary_export import export_raw_primary_comparisons


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_source_run(
    tmp_path: Path,
    *,
    identifier: str = "ng_0001",
    stage: str = "actual_ok_ng_multitask_qualitative_eval",
    raw_relative: str = "raw_heatmaps/ng/ng_0001.png",
) -> tuple[Path, Path, str]:
    source = tmp_path / "source_run"
    source.mkdir()
    raw_path = source / raw_relative
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(raw_path)
    original_path = tmp_path / "original.jpg"
    Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8)).save(
        original_path,
        quality=100,
        subsampling=0,
    )
    predictions_path = source / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": identifier,
                "cohort": "ng",
                "image_path": str(original_path),
                "image_sha256": _sha256(original_path),
                "outputs": {"raw_heatmap": raw_relative},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_path = source / "run_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "stage": stage,
                "status": "completed",
                "fixed_settings": {"mask_threshold": 0.5},
                "artifacts": {
                    raw_relative: _sha256(raw_path),
                    "predictions.jsonl": _sha256(predictions_path),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, tmp_path / "raw_primary", _sha256(receipt_path)


def _assert_no_export_output(destination: Path) -> None:
    assert not destination.exists()
    assert not destination.with_name(f".{destination.name}.tmp").exists()


def test_cli_exports_only_raw_primary_comparisons_with_verified_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_run"
    raw_path = source / "raw_heatmaps" / "ng" / "ng_0001.png"
    raw_path.parent.mkdir(parents=True)
    original_path = tmp_path / "original.jpg"
    rgb = np.full((8, 8, 3), 180, dtype=np.uint8)
    Image.fromarray(rgb).save(original_path, quality=100, subsampling=0)
    raw = np.zeros((8, 8), dtype=np.uint8)
    raw[2:6, 2:6] = 255
    Image.fromarray(raw).save(raw_path)

    predictions_path = source / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "ng_0001",
                "cohort": "ng",
                "image_path": str(original_path),
                "image_sha256": _sha256(original_path),
                "outputs": {"raw_heatmap": "raw_heatmaps/ng/ng_0001.png"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_path = source / "run_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "stage": "actual_ok_ng_multitask_qualitative_eval",
                "status": "completed",
                "fixed_settings": {"mask_threshold": 0.5},
                "artifacts": {
                    "raw_heatmaps/ng/ng_0001.png": _sha256(raw_path),
                    "predictions.jsonl": _sha256(predictions_path),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "raw_primary"
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "patch_qc.raw_primary_export",
            "--source-run",
            str(source),
            "--output-dir",
            str(destination),
            "--expected-source-receipt-sha256",
            _sha256(receipt_path),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    comparison = destination / "comparisons" / "ng" / "ng_0001.jpg"
    assert comparison.is_file()
    with Image.open(comparison) as image:
        rendered = np.asarray(image.convert("RGB"))
    assert rendered.shape == (8, 16, 3)
    assert (
        np.mean(np.abs(rendered[:, 8:].astype(float) - rendered[:, :8].astype(float)))
        > 1.0
    )
    receipt = json.loads((destination / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["display_map"] == "raw_score_map_8bit"
    assert receipt["counts"] == {"comparisons": 1, "ng": 1, "ok": 0}
    assert receipt["source_run_receipt_sha256"] == _sha256(receipt_path)
    assert sorted(
        path.relative_to(destination).as_posix() for path in destination.rglob("*")
    ) == [
        "comparisons",
        "comparisons/ng",
        "comparisons/ng/ng_0001.jpg",
        "run_receipt.json",
    ]


def test_rejects_traversal_identifier_without_leaving_output(tmp_path: Path) -> None:
    source, destination, receipt_sha256 = _build_source_run(
        tmp_path,
        identifier="../escape",
    )

    with pytest.raises(ValueError, match="invalid prediction id"):
        export_raw_primary_comparisons(
            source,
            destination,
            expected_source_receipt_sha256=receipt_sha256,
        )

    _assert_no_export_output(destination)
    assert not (tmp_path / "escape.jpg").exists()


def test_rejects_absolute_identifier_without_writing_outside_attempt_root(
    tmp_path: Path,
) -> None:
    outside_stem = tmp_path / "absolute_escape"
    outside_output = Path(f"{outside_stem}.jpg")
    source, destination, receipt_sha256 = _build_source_run(
        tmp_path,
        identifier=str(outside_stem),
    )

    with pytest.raises(ValueError, match="invalid prediction id"):
        export_raw_primary_comparisons(
            source,
            destination,
            expected_source_receipt_sha256=receipt_sha256,
        )

    _assert_no_export_output(destination)
    assert not outside_output.exists()


def test_rejects_source_receipt_from_another_stage_without_output(tmp_path: Path) -> None:
    source, destination, receipt_sha256 = _build_source_run(
        tmp_path,
        stage="different_stage",
    )

    with pytest.raises(ValueError, match="source receipt stage"):
        export_raw_primary_comparisons(
            source,
            destination,
            expected_source_receipt_sha256=receipt_sha256,
        )

    _assert_no_export_output(destination)


@pytest.mark.parametrize("raw_path_kind", ["traversal", "absolute"])
def test_rejects_raw_heatmap_paths_outside_source_run_without_output(
    tmp_path: Path,
    raw_path_kind: str,
) -> None:
    outside_raw = tmp_path / f"outside_{raw_path_kind}.png"
    raw_relative = (
        "../outside_traversal.png"
        if raw_path_kind == "traversal"
        else str(outside_raw)
    )
    source, destination, receipt_sha256 = _build_source_run(
        tmp_path,
        raw_relative=raw_relative,
    )
    outside_bytes = outside_raw.read_bytes()

    with pytest.raises(ValueError, match="raw heatmap path"):
        export_raw_primary_comparisons(
            source,
            destination,
            expected_source_receipt_sha256=receipt_sha256,
        )

    _assert_no_export_output(destination)
    assert outside_raw.read_bytes() == outside_bytes
