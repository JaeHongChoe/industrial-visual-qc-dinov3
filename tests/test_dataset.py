from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from patch_qc.capture import CaptureCondition
from patch_qc.dataset import BalancedPatchDataset, collate_patch_batch
from patch_qc.manifest import CaptureRecord, ImageRecord
from patch_qc.synthetic import SyntheticResult


def _record(product: str, object_id: str, conditions: list[str], path_prefix: str = "/unused") -> CaptureRecord:
    images = []
    for index, condition_id in enumerate(conditions):
        family = condition_id.split("_", 1)[0] if not condition_id.startswith("extra") else "extra"
        images.append(
            ImageRecord(
                path=f"{path_prefix}/{product}-{object_id}-{index}.jpg",
                condition=CaptureCondition(
                    family=family,
                    view_index=index + 1,
                    condition_id=condition_id,
                    is_standard=family != "extra",
                ),
            )
        )
    return CaptureRecord(
        capture_id=f"2026-06-01/{product}/{object_id}",
        group_id=f"{product}::{object_id}",
        product=product,
        object_id=object_id,
        date="2026-06-01",
        source_dir=path_prefix,
        images=tuple(images),
    )


def _single_image_record(image_path: Path) -> CaptureRecord:
    record = _record("PRODUCT_A", "G001", ["C_L1-01"], path_prefix=str(image_path.parent))
    return CaptureRecord(
        capture_id=record.capture_id,
        group_id=record.group_id,
        product=record.product,
        object_id=record.object_id,
        date=record.date,
        source_dir=record.source_dir,
        images=(ImageRecord(path=str(image_path), condition=record.images[0].condition),),
    )


class _FixedMaskGenerator:
    families = ("spot",)

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def apply(
        self,
        rgb: np.ndarray,
        valid_mask: np.ndarray,
        family: str,
        rng: np.random.Generator,
    ) -> SyntheticResult:
        return SyntheticResult(
            image=rgb.copy(),
            mask=self.mask.copy(),
            family=family,
            parameters={"fixture": "fixed-mask"},
        )


def test_sample_plan_balances_product_and_condition_not_image_count() -> None:
    records = [
        _record("BIG", "G001", ["A_L1-01"] * 80 + ["extra_1"]),
        _record("SMALL", "G001", ["C_L1-01"]),
    ]
    dataset = BalancedPatchDataset(records, samples_per_epoch=4000, seed=23)

    plans = [dataset.sample_plan(index) for index in range(len(dataset))]
    products = Counter(plan.product for plan in plans)
    big_conditions = Counter(plan.condition_id for plan in plans if plan.product == "BIG")

    assert 0.85 <= products["BIG"] / products["SMALL"] <= 1.15
    assert 0.85 <= big_conditions["A_L1-01"] / big_conditions["extra_1"] <= 1.15


def test_positive_item_contains_tensor_and_auditable_ok_provenance(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.jpg"
    image = np.full((320, 320, 3), 247, dtype=np.uint8)
    yy, xx = np.ogrid[:320, :320]
    image[(xx - 160) ** 2 + (yy - 160) ** 2 <= 130**2] = (178, 158, 146)
    Image.fromarray(image).save(image_path)
    record = _record("PRODUCT_A", "G001", ["C_L1-01"], path_prefix=str(tmp_path))
    original = record.images[0]
    record = CaptureRecord(
        capture_id=record.capture_id,
        group_id=record.group_id,
        product=record.product,
        object_id=record.object_id,
        date=record.date,
        source_dir=record.source_dir,
        images=(ImageRecord(path=str(image_path), condition=original.condition),),
    )
    dataset = BalancedPatchDataset(
        [record],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=1.0,
        seed=5,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 256, 256)
    assert item["image"].dtype == torch.float32
    assert item["label"].item() == 1.0
    assert item["metadata"]["group_id"] == "PRODUCT_A::G001"
    assert item["metadata"]["condition_id"] == "C_L1-01"
    assert len(item["metadata"]["patch_box"]) == 4
    assert item["metadata"]["synthetic_family"] in dataset.generator.families
    assert all("ng" not in str(key).casefold() for key in item["metadata"])


def test_positive_item_preserves_the_exact_synthetic_result_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8)).save(image_path)
    expected_mask = np.zeros((256, 256), dtype=bool)
    expected_mask[17:23, 91:104] = True
    dataset = BalancedPatchDataset(
        [_single_image_record(image_path)],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=1.0,
        seed=5,
        generator=_FixedMaskGenerator(expected_mask),
        placement_mask_fn=lambda rgb, margin: np.ones(rgb.shape[:2], dtype=bool),
        return_mask=True,
    )

    item = dataset[0]

    assert item["mask"].shape == (1, 256, 256)
    assert item["mask"].dtype == torch.float32
    assert torch.equal(item["mask"], torch.from_numpy(expected_mask).unsqueeze(0).float())
    assert set(item) == {"image", "label", "mask", "metadata"}


def test_normal_item_returns_an_all_zero_pixel_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8)).save(image_path)
    dataset = BalancedPatchDataset(
        [_single_image_record(image_path)],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=0.0,
        seed=5,
        return_mask=True,
    )

    item = dataset[0]

    assert item["mask"].shape == (1, 256, 256)
    assert item["mask"].dtype == torch.float32
    assert torch.count_nonzero(item["mask"]).item() == 0


def test_binary_dataset_default_does_not_materialize_a_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8)).save(image_path)
    dataset = BalancedPatchDataset(
        [_single_image_record(image_path)],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=0.0,
        seed=5,
    )

    item = dataset[0]
    batch = collate_patch_batch([item])

    assert "mask" not in item
    assert "mask" not in batch


def test_positive_mask_must_stay_inside_the_patch_placement_roi(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8)).save(image_path)
    invalid_mask = np.zeros((256, 256), dtype=bool)
    invalid_mask[0, 0] = True
    placement = np.zeros((256, 256), dtype=bool)
    placement[32:-32, 32:-32] = True
    dataset = BalancedPatchDataset(
        [_single_image_record(image_path)],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=1.0,
        seed=5,
        generator=_FixedMaskGenerator(invalid_mask),
        placement_mask_fn=lambda rgb, margin: placement,
        return_mask=True,
    )

    with pytest.raises(ValueError, match="placement ROI"):
        dataset[0]


def test_positive_item_builds_placement_mask_from_full_native_image(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.jpg"
    image = np.full((320, 384, 3), 247, dtype=np.uint8)
    image[20:300, 20:364] = (178, 158, 146)
    Image.fromarray(image).save(image_path)
    record = _record("PRODUCT_A", "G001", ["C_L1-01"], path_prefix=str(tmp_path))
    condition = record.images[0].condition
    record = CaptureRecord(
        capture_id=record.capture_id,
        group_id=record.group_id,
        product=record.product,
        object_id=record.object_id,
        date=record.date,
        source_dir=record.source_dir,
        images=(ImageRecord(path=str(image_path), condition=condition),),
    )
    seen_shapes: list[tuple[int, int]] = []

    def full_image_mask(rgb: np.ndarray, margin: int) -> np.ndarray:
        seen_shapes.append(rgb.shape[:2])
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[24:-24, 24:-24] = True
        return mask

    dataset = BalancedPatchDataset(
        [record],
        patch_size=256,
        samples_per_epoch=1,
        positive_probability=1.0,
        seed=5,
        placement_mask_fn=full_image_mask,
    )

    item = dataset[0]

    assert item["label"].item() == 1.0
    assert seen_shapes == [(320, 384)]


def test_positive_crop_is_anchored_to_a_narrow_full_image_roi(tmp_path: Path) -> None:
    image_path = tmp_path / "narrow-product.jpg"
    image = np.full((512, 4096, 3), 242, dtype=np.uint8)
    image[:, 2038:2058] = (178, 158, 146)
    Image.fromarray(image).save(image_path)
    record = _record("NARROW", "G001", ["A_L1-01"], path_prefix=str(tmp_path))
    condition = record.images[0].condition
    record = CaptureRecord(
        capture_id=record.capture_id,
        group_id=record.group_id,
        product=record.product,
        object_id=record.object_id,
        date=record.date,
        source_dir=record.source_dir,
        images=(ImageRecord(path=str(image_path), condition=condition),),
    )

    def narrow_mask(rgb: np.ndarray, margin: int) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[:, 2038:2058] = True
        return mask

    dataset = BalancedPatchDataset(
        [record],
        patch_size=64,
        samples_per_epoch=1,
        positive_probability=1.0,
        seed=5,
        placement_mask_fn=narrow_mask,
    )

    item = dataset[0]

    x0, _, x1, _ = item["metadata"]["patch_box"]
    assert x0 < 2058 and x1 > 2038
    assert item["label"].item() == 1.0


def test_source_reuse_groups_consecutive_native_patches() -> None:
    records = [
        _record("P1", "G001", ["A_L1-01", "B_L1-01"]),
        _record("P2", "G001", ["A_L1-01", "B_L1-01"]),
    ]
    dataset = BalancedPatchDataset(records, samples_per_epoch=8, source_reuse=4, seed=41)

    first_block = [dataset.sample_plan(index).image_path for index in range(4)]
    second_block = [dataset.sample_plan(index).image_path for index in range(4, 8)]

    assert len(set(first_block)) == 1
    assert len(set(second_block)) == 1


def test_collate_patch_batch_keeps_metadata_as_a_list() -> None:
    items = [
        {
            "image": torch.zeros(3, 256, 256),
            "label": torch.tensor(0.0),
            "mask": torch.zeros(1, 256, 256),
            "metadata": {"id": "a"},
        },
        {
            "image": torch.ones(3, 256, 256),
            "label": torch.tensor(1.0),
            "mask": torch.ones(1, 256, 256),
            "metadata": {"id": "b"},
        },
    ]

    batch = collate_patch_batch(items)

    assert batch["image"].shape == (2, 3, 256, 256)
    assert batch["label"].tolist() == [0.0, 1.0]
    assert batch["mask"].shape == (2, 1, 256, 256)
    assert batch["metadata"] == [{"id": "a"}, {"id": "b"}]
