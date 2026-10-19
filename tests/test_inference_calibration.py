import math

import pytest

from patch_qc.calibration import calibrate_threshold
from patch_qc.inference import Box, tile_boxes


def test_tile_boxes_cover_right_and_bottom_edges_without_duplicates() -> None:
    boxes = tile_boxes(width=500, height=300, patch_size=256, stride=128)

    assert Box(0, 0, 256, 256) in boxes
    assert Box(244, 44, 500, 300) in boxes
    assert max(box.x1 for box in boxes) == 500
    assert max(box.y1 for box in boxes) == 300
    assert len(boxes) == 6
    assert len(set(boxes)) == len(boxes)


def test_tile_boxes_reject_images_smaller_than_native_patch() -> None:
    with pytest.raises(ValueError, match="smaller"):
        tile_boxes(width=255, height=500, patch_size=256, stride=128)


def test_calibrate_threshold_uses_highest_sensitivity_under_fpr_budget() -> None:
    threshold = calibrate_threshold([-2.0, -1.0, 0.0, 1.0, 2.0], target_fpr=0.2)

    assert threshold == 1.0
    assert sum(score > threshold for score in [-2.0, -1.0, 0.0, 1.0, 2.0]) == 1


def test_calibrate_threshold_ignores_nonfinite_scores_but_requires_finite_data() -> None:
    assert calibrate_threshold([math.nan, -1.0, 3.0], target_fpr=0.5) == -1.0
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold([math.nan, math.inf], target_fpr=0.1)
