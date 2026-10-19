import json
from pathlib import Path

import numpy as np
from PIL import Image

from patch_qc.cli import main
from patch_qc.manifest import load_manifest_records


def test_build_manifest_cli_writes_group_safe_ok_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "OK"
    for object_id in ("G001", "G002", "G003"):
        capture = root / "2026.05.01" / "PRODUCT_A" / object_id
        capture.mkdir(parents=True)
        (capture / "_C_Photo-L1-01.jpg").touch()
        (capture / "M1-c__037.jpg").touch()
    output = tmp_path / "artifacts"

    exit_code = main(
        [
            "build-manifest",
            "--root",
            str(root),
            "--output-dir",
            str(output),
            "--start-date",
            "2026-05-01",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    assert len(rows) == 3
    assert {row["split"] for row in rows} == {"train", "calibration", "holdout"}
    assert all(row["source_label"] == "OK" for row in rows)
    assert summary["captures"] == 3
    assert summary["images"] == 6
    assert summary["date_min"] == "2026-05-01"
    assert summary["date_max"] == "2026-05-01"

    train_records = load_manifest_records(output / "manifest.jsonl", split="train")
    assert len(train_records) == 1
    assert train_records[0].images[0].condition.condition_id == "C_L1-01"


def test_synthetic_preview_cli_writes_native_panels_for_each_family_and_severity(tmp_path: Path) -> None:
    image = np.full((320, 320, 3), 248, dtype=np.uint8)
    yy, xx = np.ogrid[:320, :320]
    image[(xx - 160) ** 2 + (yy - 160) ** 2 <= 130**2] = (178, 158, 146)
    image_path = tmp_path / "ok.jpg"
    Image.fromarray(image).save(image_path)
    output = tmp_path / "preview"

    exit_code = main(
        [
            "make-synthetic-preview",
            "--image",
            str(image_path),
            "--output-dir",
            str(output),
            "--seed",
            "31",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_label"] == "OK"
    assert len(summary["families"]) == 6
    assert summary["severities"] == ["subtle", "medium", "strong"]
    for family in summary["families"]:
        for severity in summary["severities"]:
            panel = Image.open(output / f"{family}_{severity}.png")
            assert panel.size == (512, 256)
            location_panel = Image.open(output / f"{family}_{severity}_location.png")
            assert location_panel.size == (512, 256)
