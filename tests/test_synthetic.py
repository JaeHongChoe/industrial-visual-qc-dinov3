import cv2
import numpy as np
import pytest

from patch_qc.synthetic import SyntheticDefectGenerator


@pytest.fixture
def normal_patch() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:256, :256]
    base = np.empty((256, 256, 3), dtype=np.float32)
    base[..., 0] = 178 + x * 0.025
    base[..., 1] = 158 + y * 0.018
    base[..., 2] = 146 + (x + y) * 0.01
    image = np.clip(base, 0, 255).astype(np.uint8)
    valid = np.zeros((256, 256), dtype=bool)
    valid[24:232, 24:232] = True
    return image, valid


@pytest.mark.parametrize(
    "family",
    ["spot", "discolor", "scratch", "pollution", "fiber", "chipping"],
)
def test_each_synthetic_family_changes_only_valid_pixels(
    normal_patch: tuple[np.ndarray, np.ndarray], family: str
) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    result = generator.apply(image, valid, family=family, rng=np.random.default_rng(11))

    changed = np.any(result.image != image, axis=2)
    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert result.mask.shape == valid.shape
    assert result.mask.any()
    assert changed.any()
    assert not changed[~valid].any()
    assert not result.mask[~valid].any()
    assert result.family == family


def test_synthetic_generation_is_seed_deterministic(normal_patch: tuple[np.ndarray, np.ndarray]) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    first = generator.apply(image, valid, family="pollution", rng=np.random.default_rng(91))
    second = generator.apply(image, valid, family="pollution", rng=np.random.default_rng(91))

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.parameters == second.parameters


def test_families_produce_distinct_mask_geometry(normal_patch: tuple[np.ndarray, np.ndarray]) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    masks = {
        generator.apply(image, valid, family=family, rng=np.random.default_rng(7)).mask.tobytes()
        for family in generator.families
    }

    assert len(masks) == len(generator.families)


@pytest.mark.parametrize(
    "family",
    ["spot", "discolor", "scratch", "pollution", "fiber", "chipping"],
)
def test_explicit_severity_controls_visible_strength(
    normal_patch: tuple[np.ndarray, np.ndarray], family: str
) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    subtle = generator.apply(
        image,
        valid,
        family=family,
        severity="subtle",
        rng=np.random.default_rng(101),
    )
    strong = generator.apply(
        image,
        valid,
        family=family,
        severity="strong",
        rng=np.random.default_rng(101),
    )

    subtle_delta = np.abs(subtle.image.astype(np.int16) - image.astype(np.int16)).mean()
    strong_delta = np.abs(strong.image.astype(np.int16) - image.astype(np.int16)).mean()
    assert strong_delta > subtle_delta
    assert subtle.parameters["severity"] == "subtle"
    assert strong.parameters["severity"] == "strong"


def test_chipping_is_a_bright_irregular_cluster_with_severity_scaled_area(
    normal_patch: tuple[np.ndarray, np.ndarray],
) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    subtle = generator.apply(
        image,
        valid,
        family="chipping",
        severity="subtle",
        rng=np.random.default_rng(211),
    )
    strong = generator.apply(
        image,
        valid,
        family="chipping",
        severity="strong",
        rng=np.random.default_rng(211),
    )

    subtle_before = image[subtle.mask].astype(np.float32).mean()
    subtle_after = subtle.image[subtle.mask].astype(np.float32).mean()
    component_count, _ = cv2.connectedComponents(strong.mask.astype(np.uint8))
    assert subtle_after > subtle_before + 1.0
    assert strong.mask.sum() > subtle.mask.sum() * 1.25
    assert component_count - 1 >= 2
    assert subtle.parameters["placement"] == "surface"
    assert strong.parameters["fragments"] > subtle.parameters["fragments"]


def test_random_severity_is_seed_deterministic_and_supported(
    normal_patch: tuple[np.ndarray, np.ndarray],
) -> None:
    image, valid = normal_patch
    generator = SyntheticDefectGenerator()

    first = generator.apply(image, valid, family="spot", rng=np.random.default_rng(37))
    second = generator.apply(image, valid, family="spot", rng=np.random.default_rng(37))

    assert first.parameters["severity"] in generator.severities
    assert first.parameters["severity"] == second.parameters["severity"]


def test_unsupported_severity_is_rejected(normal_patch: tuple[np.ndarray, np.ndarray]) -> None:
    image, valid = normal_patch

    with pytest.raises(ValueError, match="unsupported severity"):
        SyntheticDefectGenerator().apply(
            image,
            valid,
            family="spot",
            severity="extreme",
            rng=np.random.default_rng(9),
        )
