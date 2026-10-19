from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from patch_qc.actual_multitask_inference import (
    load_multitask_checkpoint,
    load_labelme_mask,
    predict_multitask_image,
    render_mask_overlay,
    score_map_metrics,
    stitch_patch_masks,
)
from patch_qc.inference import Box
from patch_qc.model import MultiTaskOutput


class _BrightnessModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> MultiTaskOutput:
        logits = pixel_values.mean(dim=(1, 2, 3)) + self.anchor
        masks = torch.zeros(
            (pixel_values.shape[0], 1, pixel_values.shape[2], pixel_values.shape[3]),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return MultiTaskOutput(logits, logits, masks)


class _HeadContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask_head = nn.Linear(2, 1)

    def multitask_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"mask_head": self.mask_head.state_dict()}


class _NaNModel(_BrightnessModel):
    def forward(self, pixel_values: torch.Tensor) -> MultiTaskOutput:
        output = super().forward(pixel_values)
        return MultiTaskOutput(
            output.baseline_logits * torch.nan,
            output.auxiliary_logits,
            output.mask_logits,
        )


def test_stitch_patch_masks_averages_overlap_and_gates_by_patch_score() -> None:
    boxes = [Box(0, 0, 2, 2), Box(1, 0, 3, 2)]
    masks = np.asarray(
        [
            [[0.1, 0.8], [0.2, 0.6]],
            [[0.4, 0.9], [0.8, 0.3]],
        ],
        dtype=np.float32,
    )

    raw, gated = stitch_patch_masks(
        image_shape=(2, 3),
        boxes=boxes,
        masks=masks,
        patch_scores=np.asarray([0.4, 0.9], dtype=np.float32),
        gate_threshold=0.5,
    )

    np.testing.assert_allclose(raw, [[0.1, 0.6, 0.9], [0.2, 0.7, 0.3]])
    np.testing.assert_allclose(gated, [[0.0, 0.2, 0.9], [0.0, 0.4, 0.3]])


def test_stitch_patch_masks_rejects_a_box_outside_the_image() -> None:
    with np.testing.assert_raises_regex(ValueError, "outside image bounds"):
        stitch_patch_masks(
            image_shape=(2, 2),
            boxes=[Box(1, 0, 3, 2)],
            masks=np.ones((1, 2, 2), dtype=np.float32),
            patch_scores=np.asarray([1.0], dtype=np.float32),
            gate_threshold=0.5,
        )


def test_predict_multitask_image_preserves_tail_and_is_batch_size_invariant() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, 2] = 255
    model = _BrightnessModel()

    one_at_a_time = predict_multitask_image(
        model,
        rgb,
        patch_size=2,
        stride=1,
        batch_size=1,
        device="cpu",
        patch_gate_threshold=0.5,
        use_amp=False,
    )
    one_batch = predict_multitask_image(
        model,
        rgb,
        patch_size=2,
        stride=1,
        batch_size=2,
        device="cpu",
        patch_gate_threshold=0.5,
        use_amp=False,
    )

    assert one_at_a_time.boxes == (Box(0, 0, 2, 2), Box(1, 0, 3, 2))
    assert one_at_a_time.patch_count == 2
    np.testing.assert_allclose(one_at_a_time.raw_score_map, 0.5)
    np.testing.assert_allclose(one_at_a_time.gated_score_map[:, 0], 0.0)
    np.testing.assert_allclose(one_at_a_time.gated_score_map[:, 1], 0.25)
    np.testing.assert_allclose(one_at_a_time.gated_score_map[:, 2], 0.5)
    np.testing.assert_allclose(one_at_a_time.raw_score_map, one_batch.raw_score_map)
    np.testing.assert_allclose(one_at_a_time.gated_score_map, one_batch.gated_score_map)
    np.testing.assert_allclose(
        one_at_a_time.auxiliary_probabilities, one_batch.auxiliary_probabilities
    )


def test_predict_multitask_image_rejects_non_finite_outputs() -> None:
    with np.testing.assert_raises_regex(RuntimeError, "non-finite"):
        predict_multitask_image(
            _NaNModel(),
            np.zeros((2, 2, 3), dtype=np.uint8),
            patch_size=2,
            stride=1,
            batch_size=1,
            device="cpu",
            patch_gate_threshold=0.5,
            use_amp=False,
        )


def test_load_multitask_checkpoint_verifies_digest_chain_and_loads_strictly(
    tmp_path: Path,
) -> None:
    source = _HeadContainer()
    with torch.no_grad():
        source.mask_head.weight.fill_(2.0)
        source.mask_head.bias.fill_(3.0)
    checkpoint = tmp_path / "multitask.pt"
    torch.save(
        {
            "format_version": 1,
            "base_checkpoint_sha256": "a" * 64,
            "backbone_id": "test-backbone",
            "config_digest": "b" * 64,
            "multitask_state_dict": source.multitask_state_dict(),
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    destination = _HeadContainer()
    with torch.no_grad():
        destination.mask_head.weight.zero_()
        destination.mask_head.bias.zero_()

    metadata = load_multitask_checkpoint(
        destination,
        checkpoint,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_base_checkpoint_sha256="a" * 64,
        expected_backbone_id="test-backbone",
        expected_config_digest="b" * 64,
    )

    np.testing.assert_allclose(destination.mask_head.weight.detach(), 2.0)
    np.testing.assert_allclose(destination.mask_head.bias.detach(), 3.0)
    assert metadata["config_digest"] == "b" * 64


def test_load_multitask_checkpoint_rejects_hash_before_pickle_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with np.testing.assert_raises_regex(ValueError, "SHA-256 mismatch"):
        load_multitask_checkpoint(
            _HeadContainer(),
            checkpoint,
            expected_checkpoint_sha256="0" * 64,
            expected_base_checkpoint_sha256="a" * 64,
            expected_backbone_id="test-backbone",
            expected_config_digest="b" * 64,
        )


def test_load_multitask_checkpoint_rejects_config_digest_mismatch(tmp_path: Path) -> None:
    source = _HeadContainer()
    checkpoint = tmp_path / "multitask.pt"
    torch.save(
        {
            "format_version": 1,
            "base_checkpoint_sha256": "a" * 64,
            "backbone_id": "test-backbone",
            "config_digest": "b" * 64,
            "multitask_state_dict": source.multitask_state_dict(),
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with np.testing.assert_raises_regex(ValueError, "configuration digest mismatch"):
        load_multitask_checkpoint(
            _HeadContainer(),
            checkpoint,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_base_checkpoint_sha256="a" * 64,
            expected_backbone_id="test-backbone",
            expected_config_digest="c" * 64,
        )


def test_load_labelme_mask_rasterizes_polygon_and_rectangle(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.json"
    annotation.write_text(
        json.dumps(
            {
                "imageWidth": 6,
                "imageHeight": 5,
                "shapes": [
                    {
                        "label": "Bow",
                        "shape_type": "polygon",
                        "points": [[1, 1], [3, 1], [3, 3], [1, 3]],
                    },
                    {
                        "label": "Bow",
                        "shape_type": "rectangle",
                        "points": [[4, 0], [5, 1]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    mask = load_labelme_mask(annotation, image_shape=(5, 6))

    assert mask.dtype == np.bool_
    assert mask.shape == (5, 6)
    assert mask[2, 2]
    assert mask[0, 4]
    assert not mask[4, 0]


def test_load_labelme_mask_rejects_mismatched_image_size(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.json"
    annotation.write_text(
        json.dumps({"imageWidth": 8, "imageHeight": 5, "shapes": []}), encoding="utf-8"
    )

    with np.testing.assert_raises_regex(ValueError, "image size mismatch"):
        load_labelme_mask(annotation, image_shape=(5, 6))


def test_score_map_metrics_reports_ok_activation_and_labelme_localization() -> None:
    score_map = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    ground_truth = np.asarray([[False, True], [False, True]])

    metrics = score_map_metrics(score_map, threshold=0.5, ground_truth=ground_truth)

    assert metrics["predicted_positive_pixels"] == 2
    assert metrics["predicted_positive_fraction"] == 0.5
    assert metrics["maximum_score"] == np.float32(0.8)
    assert metrics["ground_truth_pixels"] == 2
    assert metrics["ground_truth_hit"] is True
    assert metrics["ground_truth_recall"] == 0.5
    assert metrics["ground_truth_iou"] == 1 / 3
    assert metrics["maximum_score_inside_ground_truth"] == np.float32(0.7)


def test_render_mask_overlay_keeps_pixels_below_threshold_unchanged() -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    score_map = np.asarray([[0.2, 0.5], [0.75, 1.0]], dtype=np.float32)

    overlay = render_mask_overlay(rgb, score_map, threshold=0.5, alpha=0.5)

    np.testing.assert_array_equal(overlay[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(overlay[0, 1], rgb[0, 1])
    assert not np.array_equal(overlay[1, 0], rgb[1, 0])
    assert not np.array_equal(overlay[1, 1], rgb[1, 1])
