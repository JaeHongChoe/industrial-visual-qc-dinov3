from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patch_qc.multitask_train import (
    MultitaskMetricAccumulator,
    compute_multitask_loss,
    evaluate_multitask,
    positive_soft_dice_loss,
    save_multitask_checkpoint,
    train_multitask_epoch,
)


def _output(
    auxiliary_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    baseline_logits: torch.Tensor | None = None,
) -> SimpleNamespace:
    if baseline_logits is None:
        baseline_logits = torch.zeros_like(auxiliary_logits)
    return SimpleNamespace(
        baseline_logits=baseline_logits,
        auxiliary_logits=auxiliary_logits,
        mask_logits=mask_logits,
    )


def test_all_normal_multitask_loss_is_finite_and_skips_positive_dice() -> None:
    labels = torch.zeros(2)
    masks = torch.zeros(2, 1, 4, 4)
    result = compute_multitask_loss(
        _output(torch.zeros(2), torch.full_like(masks, -2.0)),
        labels,
        masks,
        max_positive_weight=8.0,
    )

    assert torch.isfinite(result.total)
    assert float(result.mask_dice_loss) == 0.0
    assert float(result.mask_positive_weight) == 1.0


def test_sparse_mask_positive_weight_is_capped() -> None:
    labels = torch.ones(1)
    masks = torch.zeros(1, 1, 4, 4)
    masks[0, 0, 0, 0] = 1.0

    result = compute_multitask_loss(
        _output(torch.zeros(1), torch.zeros_like(masks)),
        labels,
        masks,
        max_positive_weight=8.0,
    )

    assert float(result.mask_positive_weight) == 8.0


@pytest.mark.parametrize(
    ("label", "mask_value"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_multitask_loss_rejects_label_mask_mismatch(label: float, mask_value: float) -> None:
    labels = torch.tensor([label])
    masks = torch.full((1, 1, 2, 2), mask_value)

    with pytest.raises(ValueError, match="label and mask"):
        compute_multitask_loss(_output(torch.zeros(1), torch.zeros_like(masks)), labels, masks)


def test_multitask_loss_rejects_non_binary_or_non_finite_masks() -> None:
    labels = torch.ones(1)
    for invalid in (0.5, float("nan")):
        masks = torch.zeros(1, 1, 2, 2)
        masks[0, 0, 0, 0] = invalid
        with pytest.raises(ValueError, match="binary and finite"):
            compute_multitask_loss(_output(torch.zeros(1), torch.zeros_like(masks)), labels, masks)


def test_positive_soft_dice_ignores_normal_samples() -> None:
    masks = torch.zeros(2, 1, 2, 2)
    masks[0, 0, 0, 0] = 1.0
    logits = torch.full_like(masks, -20.0)
    logits[0, 0, 0, 0] = 20.0
    logits[1] = 20.0

    loss = positive_soft_dice_loss(logits, masks)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_metric_accumulator_reports_hand_checked_pixel_counts_and_baseline_drift() -> None:
    labels = torch.tensor([1.0, 0.0])
    masks = torch.zeros(2, 1, 2, 2)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 0, 0, 1] = 1.0
    logits = torch.full_like(masks, -20.0)
    logits[0, 0, 0, 0] = 20.0  # one true positive, one false negative
    logits[0, 0, 1, 0] = 20.0  # one false positive
    logits[1, 0, 1, 1] = 20.0  # one false positive on the normal sample
    output = _output(
        auxiliary_logits=torch.tensor([20.0, -20.0]),
        mask_logits=logits,
        baseline_logits=torch.tensor([0.2, -0.4]),
    )
    accumulator = MultitaskMetricAccumulator(mask_threshold=0.5)

    accumulator.update(
        output,
        labels,
        masks,
        reference_baseline_logits=torch.tensor([0.1, -0.4]),
    )
    metrics = accumulator.compute()

    assert metrics["examples"] == 2
    assert metrics["auxiliary_accuracy"] == 1.0
    assert metrics["auxiliary_precision"] == 1.0
    assert metrics["auxiliary_recall"] == 1.0
    assert metrics["pixel_precision"] == pytest.approx(1 / 3)
    assert metrics["pixel_recall"] == pytest.approx(1 / 2)
    assert metrics["pixel_iou"] == pytest.approx(1 / 4)
    assert metrics["pixel_dice"] == pytest.approx(2 / 5)
    assert metrics["positive_localization_hit_rate"] == 1.0
    assert metrics["normal_predicted_area_mean"] == pytest.approx(1 / 4)
    assert metrics["normal_predicted_area_p95"] == pytest.approx(1 / 4)
    assert metrics["normal_predicted_area_p99"] == pytest.approx(1 / 4)
    assert metrics["predicted_positive_fraction"] == pytest.approx(3 / 8)
    assert metrics["baseline_logit_max_abs_drift"] == pytest.approx(0.1)


@pytest.mark.parametrize(("logit", "expected"), [(-20.0, 0.0), (20.0, 1.0)])
def test_metric_accumulator_reports_all_zero_and_all_one_prediction_fractions(
    logit: float, expected: float
) -> None:
    masks = torch.zeros(2, 1, 2, 2)
    accumulator = MultitaskMetricAccumulator(mask_threshold=0.5)

    accumulator.update(
        _output(torch.zeros(2), torch.full_like(masks, logit)),
        torch.zeros(2),
        masks,
    )

    assert accumulator.compute()["predicted_positive_fraction"] == expected


def test_metric_accumulator_does_not_claim_unmeasured_baseline_drift() -> None:
    output = _output(torch.tensor([-20.0]), torch.full((1, 1, 2, 2), -20.0))
    accumulator = MultitaskMetricAccumulator()
    accumulator.update(output, torch.zeros(1), torch.zeros(1, 1, 2, 2))

    metrics = accumulator.compute()

    assert "baseline_logit_max_abs_drift" not in metrics


class _TinyMultitaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.base.weight.requires_grad_(False)
        self.mask_head = nn.Conv2d(3, 1, kernel_size=1)
        self.auxiliary_head = nn.Linear(1, 1)

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        with torch.no_grad():
            baseline = self.base(images).mean((1, 2, 3))
        mask_logits = self.mask_head(images)
        pooled = mask_logits.mean((2, 3))
        return _output(self.auxiliary_head(pooled).squeeze(1), mask_logits, baseline)

    def trainable_parameters(self):
        yield from self.mask_head.parameters()
        yield from self.auxiliary_head.parameters()

    def multitask_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "mask_head": self.mask_head.state_dict(),
            "auxiliary_head": self.auxiliary_head.state_dict(),
        }


def test_one_training_step_updates_new_branch_without_changing_frozen_base() -> None:
    torch.manual_seed(7)
    model = _TinyMultitaskModel()
    base_before = model.base.weight.detach().clone()
    branch_before = model.mask_head.weight.detach().clone()
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)
    batch = {
        "image": torch.randn(2, 3, 4, 4),
        "label": torch.tensor([0.0, 1.0]),
        "mask": torch.cat((torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))),
    }

    metrics = train_multitask_epoch(
        model,
        [batch],
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert metrics["examples"] == 2
    assert math.isfinite(float(metrics["loss"]))
    assert torch.equal(model.base.weight, base_before)
    assert not torch.equal(model.mask_head.weight, branch_before)
    assert model.base.weight.grad is None


def test_training_rejects_an_optimizer_that_contains_frozen_base_parameters() -> None:
    model = _TinyMultitaskModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "image": torch.randn(2, 3, 4, 4),
        "label": torch.tensor([0.0, 1.0]),
        "mask": torch.cat((torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))),
    }

    with pytest.raises(ValueError, match="frozen parameter"):
        train_multitask_epoch(
            model,
            [batch],
            optimizer,
            device=torch.device("cpu"),
            use_amp=False,
        )


def test_evaluation_reports_loss_without_updating_parameters() -> None:
    torch.manual_seed(11)
    model = _TinyMultitaskModel()
    branch_before = model.mask_head.weight.detach().clone()
    batch = {
        "image": torch.randn(2, 3, 4, 4),
        "label": torch.tensor([0.0, 1.0]),
        "mask": torch.cat((torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))),
    }

    metrics = evaluate_multitask(model, [batch], device=torch.device("cpu"), use_amp=False)

    assert metrics["examples"] == 2
    assert math.isfinite(float(metrics["loss"]))
    assert torch.equal(model.mask_head.weight, branch_before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert "by_family_severity" not in metrics


class _EchoMaskModel(nn.Module):
    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        mask_logits = images[:, :1]
        return _output(torch.zeros(images.shape[0]), mask_logits)


def test_evaluation_reports_positive_metrics_nested_by_synthetic_family_and_severity() -> None:
    images = torch.full((4, 3, 2, 2), -20.0)
    masks = torch.zeros(4, 1, 2, 2)

    masks[0, 0, 0, :2] = 1.0
    images[0, 0, 0, 0] = 20.0  # one TP, one FN
    images[0, 0, 1, 0] = 20.0  # one FP
    masks[1, 0, 0, 0] = 1.0
    images[1, 0, 0, 0] = 20.0  # one TP
    masks[2, 0, 1, 1] = 1.0  # missed positive
    images[3, 0, 1, 1] = 20.0  # normal prediction must not enter a positive group

    batch = {
        "image": images,
        "label": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "mask": masks,
        "metadata": [
            {"synthetic_family": "spot", "synthetic_parameters": {"severity": "subtle"}},
            {"synthetic_family": "spot", "synthetic_parameters": {"severity": "subtle"}},
            {"synthetic_family": "fiber", "synthetic_parameters": {"severity": "strong"}},
            {"synthetic_family": None, "synthetic_parameters": None},
        ],
    }

    metrics = evaluate_multitask(
        _EchoMaskModel(),
        [batch],
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert metrics["by_family_severity"] == {
        "fiber": {
            "strong": {
                "examples": 1,
                "pixel_dice": 0.0,
                "pixel_recall": 0.0,
                "positive_localization_hit_rate": 0.0,
            }
        },
        "spot": {
            "subtle": {
                "examples": 2,
                "pixel_dice": pytest.approx(2 / 3),
                "pixel_recall": pytest.approx(2 / 3),
                "positive_localization_hit_rate": 1.0,
            }
        },
    }


def test_evaluation_honors_loss_weights_and_mask_threshold() -> None:
    model = _TinyMultitaskModel()
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.zero_()
        model.auxiliary_head.weight.zero_()
        model.auxiliary_head.bias.zero_()
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "label": torch.zeros(1),
        "mask": torch.zeros(1, 1, 4, 4),
    }

    metrics = evaluate_multitask(
        model,
        [batch],
        device=torch.device("cpu"),
        use_amp=False,
        loss_kwargs={"auxiliary_weight": 0.0, "mask_bce_weight": 0.0, "mask_dice_weight": 0.0},
        mask_threshold=0.75,
    )

    assert metrics["loss"] == 0.0
    assert metrics["normal_predicted_area_mean"] == 0.0


def test_checkpoint_contains_only_new_heads_and_base_provenance(tmp_path: Path) -> None:
    model = _TinyMultitaskModel()
    destination = tmp_path / "checkpoint_epoch_001.pt"

    save_multitask_checkpoint(
        model,
        destination,
        base_checkpoint_path="/models/epoch_003.pt",
        base_checkpoint_sha256="a" * 64,
        backbone_id="vit-large-test",
        config_digest="b" * 64,
        epoch=1,
        metrics={"loss": 0.25, "examples": 2},
    )

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert payload["base_checkpoint_sha256"] == "a" * 64
    assert payload["config_digest"] == "b" * 64
    assert payload["epoch"] == 1
    assert set(payload["multitask_state_dict"]) == {"mask_head", "auxiliary_head"}
    assert all("base" not in key for key in payload["multitask_state_dict"])


def test_checkpoint_rejects_non_hex_digests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        save_multitask_checkpoint(
            _TinyMultitaskModel(),
            tmp_path / "bad.pt",
            base_checkpoint_path="/models/epoch_003.pt",
            base_checkpoint_sha256="z" * 64,
            backbone_id="vit-large-test",
            config_digest="b" * 64,
            epoch=1,
            metrics={"loss": 0.25},
        )
