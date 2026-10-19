import numpy as np

from patch_qc.roi import estimate_placement_mask


def test_estimate_placement_mask_rejects_background_and_erodes_margin() -> None:
    image = np.full((128, 128, 3), 248, dtype=np.uint8)
    image[24:104, 20:108] = (178, 158, 146)

    mask = estimate_placement_mask(image, margin=8)

    assert mask.dtype == np.bool_
    assert mask.shape == (128, 128)
    assert bool(mask[64, 64]) is True
    assert bool(mask[0, 0]) is False
    assert bool(mask[25, 64]) is False
    assert bool(mask[40, 64]) is True


def test_estimate_placement_mask_returns_empty_for_uniform_background() -> None:
    image = np.full((96, 96, 3), 245, dtype=np.uint8)

    mask = estimate_placement_mask(image, margin=4)

    assert not mask.any()


def test_top_plate_geometry_rejects_large_border_fixture() -> None:
    height, width = 320, 480
    image = np.full((height, width, 3), 242, dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    disk = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= 128**2
    image[disk] = (178, 158, 146)
    image[20:280, 5:70] = (45, 50, 55)

    mask = estimate_placement_mask(image, margin=8)

    assert np.mean(mask[disk]) > 0.75
    assert np.mean(mask[30:270, 15:60]) < 0.02
    assert bool(mask[160, 240]) is True
    assert bool(mask[10, 470]) is False
