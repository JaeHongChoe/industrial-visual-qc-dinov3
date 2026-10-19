from __future__ import annotations

import cv2
import numpy as np


def _textured_thumbnail() -> np.ndarray:
    image = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(image, (69, 57), 17, 0.8, -1)
    cv2.rectangle(image, (145, 43), (207, 101), 0.45, -1)
    cv2.line(image, (48, 208), (213, 151), 1.0, 7)
    return cv2.GaussianBlur(image, (7, 7), 1.0)


def _translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, image.shape[::-1], flags=cv2.INTER_LINEAR)


def test_phase_registration_recovers_bounded_translation_on_sobel_edges() -> None:
    from patch_qc.normal_prior_postprocess import register_reference_thumbnail

    reference = _textured_thumbnail()
    query = _translate(reference, 3.0, -2.0)
    result = register_reference_thumbnail(reference, query)

    assert result.accepted is True
    assert result.response >= 0.15
    assert result.overlap_fraction >= 0.95
    np.testing.assert_allclose(result.shift_xy, (3.0, -2.0), atol=0.05)
    assert np.mean(np.abs(result.aligned_reference - query)) < 0.005


def test_phase_registration_fails_open_when_shift_exceeds_compact_bound() -> None:
    from patch_qc.normal_prior_postprocess import register_reference_thumbnail

    reference = _textured_thumbnail()
    result = register_reference_thumbnail(reference, _translate(reference, 12.0, 0.0))

    assert result.accepted is False
    assert result.reason == "shift_out_of_bounds"
    np.testing.assert_array_equal(result.aligned_reference, reference)


def test_phase_registration_fails_open_on_low_response() -> None:
    from patch_qc.normal_prior_postprocess import register_reference_thumbnail

    flat = np.zeros((256, 256), dtype=np.float32)
    result = register_reference_thumbnail(flat, flat)

    assert result.accepted is False
    assert result.reason == "response_below_minimum"
    np.testing.assert_array_equal(result.aligned_reference, flat)


def test_sift_fallback_recovers_bounded_similarity_transform() -> None:
    from patch_qc.normal_prior_postprocess import register_reference_thumbnail

    rng = np.random.default_rng(7)
    reference = np.zeros((256, 256), dtype=np.float32)
    for _ in range(30):
        x, y = rng.integers(25, 231, size=2)
        radius = int(rng.integers(3, 10))
        value = float(rng.uniform(0.3, 1.0))
        cv2.circle(reference, (int(x), int(y)), radius, value, -1)
    matrix = cv2.getRotationMatrix2D((128, 128), 23.0, 1.04).astype(np.float32)
    matrix[:, 2] += np.asarray([12.0, -9.0], dtype=np.float32)
    query = cv2.warpAffine(reference, matrix, (256, 256))

    result = register_reference_thumbnail(reference, query, allow_feature_fallback=True)

    assert result.accepted is True
    assert result.method == "sift_similarity"
    assert result.feature_inliers >= 6
    assert result.feature_inlier_ratio >= 0.25
    assert 0.8 <= result.scale <= 1.2
    assert np.mean(np.abs(result.aligned_reference - query)) < 0.03


def test_binary_recurrence_prior_requires_two_of_three_aligned_references() -> None:
    from patch_qc.normal_prior_postprocess import aggregate_binary_recurrence_prior

    query = _textured_thumbnail()
    maps = [np.zeros_like(query) for _ in range(3)]
    maps[0][80:84, 100:104] = 0.9
    maps[1][80:84, 100:104] = 0.7
    maps[2][120:124, 130:134] = 0.9
    result = aggregate_binary_recurrence_prior(
        query, [query, query, query], maps, dilation_radius_compact_px=0
    )

    assert result.usable is True
    assert result.accepted_count == 3
    np.testing.assert_array_equal(result.prior[80:84, 100:104], np.ones((4, 4)))
    np.testing.assert_array_equal(result.prior[120:124, 130:134], np.zeros((4, 4)))


def test_aggregation_exposes_maximum_continuous_reference_ceiling() -> None:
    from patch_qc.normal_prior_postprocess import aggregate_binary_recurrence_prior

    query = _textured_thumbnail()
    maps = [np.zeros_like(query) for _ in range(3)]
    maps[0][80, 100] = 0.55
    maps[1][80, 100] = 0.60
    maps[2][80, 100] = 0.20

    result = aggregate_binary_recurrence_prior(
        query, [query, query, query], maps, dilation_radius_compact_px=0
    )

    assert result.prior[80, 100] == 1.0
    assert result.reference_score_ceiling[80, 100] == np.float32(0.60)


def test_binary_recurrence_prior_fails_open_if_any_registration_fails() -> None:
    from patch_qc.normal_prior_postprocess import aggregate_binary_recurrence_prior

    query = _textured_thumbnail()
    activation = np.ones_like(query)
    result = aggregate_binary_recurrence_prior(
        query, [query, query, np.zeros_like(query)], [activation] * 3
    )

    assert result.usable is False
    assert result.accepted_count == 2
    np.testing.assert_array_equal(result.prior, np.zeros_like(query))


def test_b_and_c_scores_stay_between_conservative_bounds() -> None:
    from patch_qc.normal_prior_postprocess import (
        apply_case_support_boost,
        apply_recurrence_suppression,
    )

    a_score = np.asarray([[0.2, 0.8], [1.0, 0.6]], dtype=np.float32)
    recurrence = np.asarray([[1.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    b_score = apply_recurrence_suppression(a_score, recurrence)
    c_score = apply_case_support_boost(a_score, b_score, support=1.0)

    np.testing.assert_allclose(b_score, [[0.1, 0.4], [1.0, 0.45]], atol=1e-6)
    np.testing.assert_allclose(c_score, [[0.125, 0.5], [1.0, 0.4875]], atol=1e-6)
    assert np.all(0.5 * a_score <= b_score)
    assert np.all(b_score <= a_score)
    assert np.all(b_score <= c_score)
    assert np.all(c_score <= a_score)


def test_reference_relative_peak_preserves_only_novel_recurrent_peak() -> None:
    from patch_qc.normal_prior_postprocess import (
        apply_recurrence_suppression,
        reference_relative_local_peak_mask,
    )

    query = np.full((5, 5), 0.20, dtype=np.float32)
    query[2, 1:4] = 0.70
    query[2, 2] = 0.80
    reference_ceiling = np.full((5, 5), 0.60, dtype=np.float32)
    recurrence = np.ones((5, 5), dtype=np.float32)

    preserve = reference_relative_local_peak_mask(
        query, reference_ceiling, radius_px=1
    )
    suppressed = apply_recurrence_suppression(
        query, recurrence, preserve_mask=preserve
    )

    assert preserve.dtype == np.bool_
    assert preserve[2, 2]
    assert not preserve[2, 1]
    assert suppressed[2, 2] == np.float32(0.80)
    assert suppressed[2, 1] == np.float32(0.35)


def test_build_peak_candidate_maps_compact_gate_back_to_native_coordinates() -> None:
    from patch_qc.normal_prior_postprocess import (
        build_reference_relative_peak_candidate,
    )

    compact_query = np.full((3, 3), 0.20, dtype=np.float32)
    compact_query[1, 0] = 0.70
    compact_query[1, 1] = 0.80
    raw = np.kron(compact_query, np.ones((2, 2), dtype=np.float32))
    compact_prior = np.ones((3, 3), dtype=np.float32)
    compact_ceiling = np.full((3, 3), 0.60, dtype=np.float32)

    result = build_reference_relative_peak_candidate(
        raw,
        compact_prior,
        compact_ceiling,
        radius_compact_px=1,
        suppression_strength=0.5,
    )

    assert result.baseline.shape == raw.shape
    assert result.candidate.shape == raw.shape
    assert result.compact_preserve_mask.shape == compact_prior.shape
    assert result.compact_preserve_mask.dtype == np.bool_
    assert result.compact_preserve_mask[1, 1]
    assert not result.compact_preserve_mask[1, 0]
    np.testing.assert_allclose(result.baseline[2:4, 2:4], 0.40, atol=1e-6)
    np.testing.assert_allclose(result.candidate[2:4, 2:4], 0.80, atol=1e-6)
    np.testing.assert_allclose(result.candidate[2:4, 0:2], 0.35, atol=1e-6)


def test_equal_to_reference_local_peak_is_not_preserved() -> None:
    from patch_qc.normal_prior_postprocess import (
        reference_relative_local_peak_mask,
    )

    query = np.zeros((3, 3), dtype=np.float32)
    query[1, 1] = 0.80
    ceiling = np.zeros((3, 3), dtype=np.float32)
    ceiling[1, 1] = 0.80

    preserve = reference_relative_local_peak_mask(query, ceiling, radius_px=1)

    assert not preserve[1, 1]


def test_nearby_reference_ceiling_blocks_shift_tolerant_false_preservation() -> None:
    from patch_qc.normal_prior_postprocess import (
        reference_relative_local_peak_mask,
    )

    query = np.zeros((3, 3), dtype=np.float32)
    query[1, 1] = 0.80
    ceiling = np.zeros((3, 3), dtype=np.float32)
    ceiling[1, 2] = 0.90

    preserve = reference_relative_local_peak_mask(query, ceiling, radius_px=1)

    assert not preserve[1, 1]


def test_nonmaximum_novel_pixel_is_not_preserved() -> None:
    from patch_qc.normal_prior_postprocess import (
        reference_relative_local_peak_mask,
    )

    query = np.zeros((3, 3), dtype=np.float32)
    query[1, 1] = 0.80
    query[1, 2] = 0.90
    ceiling = np.full((3, 3), 0.10, dtype=np.float32)

    preserve = reference_relative_local_peak_mask(query, ceiling, radius_px=1)

    assert not preserve[1, 1]
    assert preserve[1, 2]


def test_failed_aggregation_exposes_zero_reference_ceiling() -> None:
    from patch_qc.normal_prior_postprocess import aggregate_binary_recurrence_prior

    query = _textured_thumbnail()
    activation = np.ones_like(query)
    result = aggregate_binary_recurrence_prior(
        query, [query, query, np.zeros_like(query)], [activation] * 3
    )

    assert result.usable is False
    np.testing.assert_array_equal(
        result.reference_score_ceiling, np.zeros_like(query)
    )


def test_render_comparison_keeps_original_on_left_and_overlay_on_right() -> None:
    from patch_qc.normal_prior_postprocess import render_original_overlay_comparison

    rgb = np.full((4, 5, 3), 100, dtype=np.uint8)
    score = np.zeros((4, 5), dtype=np.float32)
    score[1:3, 2:4] = 1.0
    comparison = render_original_overlay_comparison(rgb, score, threshold=0.5)

    assert comparison.shape == (4, 10, 3)
    np.testing.assert_array_equal(comparison[:, :5], rgb)
    assert not np.array_equal(comparison[:, 5:], rgb)
