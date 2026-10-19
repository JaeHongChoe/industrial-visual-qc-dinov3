from pathlib import Path

import numpy as np
from PIL import Image

from patch_qc.inference import Box, PatchPrediction
from patch_qc.visualize import (
    save_native_side_by_side,
    save_prediction_panel,
    save_synthetic_panel,
)


def test_prediction_panel_keeps_native_original_untouched_on_left(tmp_path: Path) -> None:
    original = np.zeros((40, 60, 3), dtype=np.uint8)
    original[..., 0] = np.arange(60, dtype=np.uint8)
    prediction = PatchPrediction(Box(10, 5, 30, 25), logit=2.5, probability=0.924)
    output = tmp_path / "prediction.png"

    save_prediction_panel(original, [prediction], threshold=1.0, output_path=output)

    panel = np.asarray(Image.open(output).convert("RGB"))
    assert panel.shape == (40, 120, 3)
    assert np.array_equal(panel[:, :60], original)
    assert not np.array_equal(panel[:, 60:], original)


def test_synthetic_panel_keeps_native_original_untouched_on_left(tmp_path: Path) -> None:
    original = np.full((32, 32, 3), 100, dtype=np.uint8)
    synthetic = original.copy()
    synthetic[12:20, 12:20] = 150
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True
    output = tmp_path / "synthetic.png"

    save_synthetic_panel(original, synthetic, mask, output)

    panel = np.asarray(Image.open(output).convert("RGB"))
    assert panel.shape == (32, 64, 3)
    assert np.array_equal(panel[:, :32], original)
    assert not np.array_equal(panel[:, 32:], original)


def test_synthetic_panel_can_show_raw_synthetic_without_outline(tmp_path: Path) -> None:
    original = np.full((32, 32, 3), 100, dtype=np.uint8)
    synthetic = original.copy()
    synthetic[12:20, 12:20] = 150
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True
    output = tmp_path / "synthetic_raw.png"

    save_synthetic_panel(original, synthetic, mask, output, draw_outline=False)

    panel = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(panel[:, :32], original)
    assert np.array_equal(panel[:, 32:], synthetic)


def test_native_side_by_side_never_resizes_either_half(tmp_path: Path) -> None:
    left = np.full((37, 53, 3), 10, dtype=np.uint8)
    right = np.full((37, 53, 3), 20, dtype=np.uint8)
    output = tmp_path / "native.png"

    save_native_side_by_side(left, right, output)

    panel = np.asarray(Image.open(output).convert("RGB"))
    assert panel.shape == (37, 106, 3)
    assert np.array_equal(panel[:, :53], left)
    assert np.array_equal(panel[:, 53:], right)
