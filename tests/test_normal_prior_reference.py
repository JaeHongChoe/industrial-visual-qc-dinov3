from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from patch_qc.model import MultiTaskOutput
from patch_qc.normal_prior_reference import ReferenceImage, evaluate_reference_rows


class _ConstantMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> MultiTaskOutput:
        logits = torch.full(
            (pixel_values.shape[0],),
            1.0,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        masks = torch.zeros(
            (pixel_values.shape[0], 1, pixel_values.shape[2], pixel_values.shape[3]),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return MultiTaskOutput(logits, logits, masks)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_eval_writes_only_compact_sealed_maps_and_previews(tmp_path: Path) -> None:
    source = tmp_path / "reference.jpg"
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, 6:] = 200
    Image.fromarray(rgb).save(source, quality=100, subsampling=0)
    destination = tmp_path / "reference_run"
    row = ReferenceImage(
        identifier="ref_0001",
        image_path=source,
        image_sha256=_sha256(source),
        product="PRODUCT_A",
        condition_id="C_L1-07",
        capture_id="2026-08-01/PRODUCT_A/G0001",
        group_id="PRODUCT_A::G0001",
    )

    records, artifacts = evaluate_reference_rows(
        [row],
        _ConstantMaskModel(),
        output_dir=destination,
        patch_size=4,
        stride=4,
        batch_size=2,
        device="cpu",
        map_width=6,
        use_amp=False,
    )

    raw_path = destination / "raw_maps" / "ref_0001.png"
    preview_path = destination / "previews" / "ref_0001.jpg"
    assert sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*")) == [
        "previews",
        "previews/ref_0001.jpg",
        "raw_maps",
        "raw_maps/ref_0001.png",
    ]
    with Image.open(raw_path) as image:
        raw = np.asarray(image.convert("L"))
    with Image.open(preview_path) as image:
        preview_shape = (image.height, image.width)
    assert raw.shape == (4, 6)
    assert preview_shape == (4, 6)
    assert set(np.unique(raw)).issubset({127, 128})
    assert records == [
        {
            "id": "ref_0001",
            "product": "PRODUCT_A",
            "condition_id": "C_L1-07",
            "capture_id": "2026-08-01/PRODUCT_A/G0001",
            "group_id": "PRODUCT_A::G0001",
            "image_path": str(source),
            "image_sha256": _sha256(source),
            "native_shape": [8, 12, 3],
            "map_shape": [4, 6],
            "patch_count": 6,
            "baseline_patches_above_0_7": 6,
            "outputs": {
                "raw_map": "raw_maps/ref_0001.png",
                "preview": "previews/ref_0001.jpg",
            },
        }
    ]
    assert artifacts == {
        "previews/ref_0001.jpg": _sha256(preview_path),
        "raw_maps/ref_0001.png": _sha256(raw_path),
    }


def test_reference_eval_rejects_changed_source_before_inference(tmp_path: Path) -> None:
    source = tmp_path / "reference.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(source)
    row = ReferenceImage(
        identifier="ref_0001",
        image_path=source,
        image_sha256="0" * 64,
        product="PRODUCT_A",
        condition_id="C_L1-07",
        capture_id="capture",
        group_id="PRODUCT_A::G0001",
    )

    with np.testing.assert_raises_regex(RuntimeError, "changed after reference planning"):
        evaluate_reference_rows(
            [row],
            _ConstantMaskModel(),
            output_dir=tmp_path / "run",
            patch_size=4,
            stride=4,
            batch_size=1,
            device="cpu",
            map_width=4,
            use_amp=False,
        )

