from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Protocol

import torch
from torch import nn
from torch.nn import functional as F


class _MultitaskOutput(Protocol):
    baseline_logits: torch.Tensor
    auxiliary_logits: torch.Tensor
    mask_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class _MetricOutputView:
    baseline_logits: torch.Tensor
    auxiliary_logits: torch.Tensor
    mask_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class MultitaskLoss:
    total: torch.Tensor
    auxiliary_bce: torch.Tensor
    mask_bce: torch.Tensor
    mask_dice_loss: torch.Tensor
    mask_positive_weight: torch.Tensor


def positive_soft_dice_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return soft Dice loss over positive samples without rewarding empty masks."""
    if mask_logits.shape != masks.shape or mask_logits.ndim != 4:
        raise ValueError("mask logits and targets must share [batch, 1, height, width] shape")
    positive_samples = masks.flatten(1).sum(1) > 0
    if not bool(positive_samples.any()):
        return mask_logits.sum() * 0.0
    probabilities = mask_logits[positive_samples].sigmoid().flatten(1)
    targets = masks[positive_samples].float().flatten(1)
    intersection = (probabilities * targets).sum(1)
    denominator = probabilities.sum(1) + targets.sum(1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def compute_multitask_loss(
    output: _MultitaskOutput,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    auxiliary_weight: float = 0.25,
    mask_bce_weight: float = 1.0,
    mask_dice_weight: float = 1.0,
    max_positive_weight: float = 64.0,
) -> MultitaskLoss:
    labels = labels.float()
    masks = masks.float()
    if labels.ndim != 1 or masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("labels must be [batch] and masks must be [batch, 1, height, width]")
    if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary and finite")
    if not bool(torch.isfinite(masks).all()) or not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("masks must be binary and finite")
    mask_labels = masks.flatten(1).any(1).to(labels.dtype)
    if not torch.equal(mask_labels, labels):
        raise ValueError("label and mask presence must agree")
    if output.auxiliary_logits.shape != labels.shape:
        raise ValueError("auxiliary logits and labels must have the same shape")
    if output.mask_logits.shape != masks.shape:
        raise ValueError("mask logits and targets must have the same shape")
    if max_positive_weight < 1.0:
        raise ValueError("max_positive_weight must be at least one")

    positive_pixels = masks.sum()
    if float(positive_pixels.detach()) > 0.0:
        negative_pixels = masks.numel() - positive_pixels
        positive_weight = (negative_pixels / positive_pixels).clamp(
            min=1.0,
            max=max_positive_weight,
        )
    else:
        positive_weight = masks.new_tensor(1.0)

    auxiliary_bce = F.binary_cross_entropy_with_logits(output.auxiliary_logits, labels)
    mask_bce = F.binary_cross_entropy_with_logits(
        output.mask_logits,
        masks,
        pos_weight=positive_weight,
    )
    mask_dice = positive_soft_dice_loss(output.mask_logits, masks)
    total = (
        auxiliary_weight * auxiliary_bce
        + mask_bce_weight * mask_bce
        + mask_dice_weight * mask_dice
    )
    return MultitaskLoss(
        total=total,
        auxiliary_bce=auxiliary_bce,
        mask_bce=mask_bce,
        mask_dice_loss=mask_dice,
        mask_positive_weight=positive_weight.detach(),
    )


class MultitaskMetricAccumulator:
    def __init__(self, *, mask_threshold: float = 0.5) -> None:
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be between zero and one")
        self.mask_threshold = mask_threshold
        self.examples = 0
        self.auxiliary_correct = 0
        self.auxiliary_true_positive = 0
        self.auxiliary_false_positive = 0
        self.auxiliary_false_negative = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.positive_examples = 0
        self.positive_hits = 0
        self.predicted_positive_pixels = 0
        self.predicted_pixels = 0
        self.normal_areas: list[float] = []
        self.baseline_max_drift = 0.0
        self.baseline_drift_measured = False

    @torch.no_grad()
    def update(
        self,
        output: _MultitaskOutput,
        labels: torch.Tensor,
        masks: torch.Tensor,
        *,
        reference_baseline_logits: torch.Tensor | None = None,
    ) -> None:
        labels = labels.detach().float()
        masks = masks.detach().bool()
        predicted = output.mask_logits.detach().sigmoid() >= self.mask_threshold
        if predicted.shape != masks.shape:
            raise ValueError("predicted and target masks must have the same shape")
        if output.auxiliary_logits.shape != labels.shape:
            raise ValueError("auxiliary logits and labels must have the same shape")

        self.examples += int(labels.numel())
        auxiliary_predicted = output.auxiliary_logits.detach() > 0
        auxiliary_target = labels > 0.5
        self.auxiliary_correct += int((auxiliary_predicted == auxiliary_target).sum())
        self.auxiliary_true_positive += int((auxiliary_predicted & auxiliary_target).sum())
        self.auxiliary_false_positive += int((auxiliary_predicted & ~auxiliary_target).sum())
        self.auxiliary_false_negative += int((~auxiliary_predicted & auxiliary_target).sum())
        self.true_positive += int((predicted & masks).sum())
        self.false_positive += int((predicted & ~masks).sum())
        self.false_negative += int((~predicted & masks).sum())
        self.predicted_positive_pixels += int(predicted.sum())
        self.predicted_pixels += predicted.numel()

        positive_samples = masks.flatten(1).any(1)
        overlap = (predicted & masks).flatten(1).any(1)
        self.positive_examples += int(positive_samples.sum())
        self.positive_hits += int((positive_samples & overlap).sum())
        normal_samples = ~positive_samples
        if bool(normal_samples.any()):
            areas = predicted[normal_samples].float().flatten(1).mean(1).cpu().tolist()
            self.normal_areas.extend(float(value) for value in areas)

        if reference_baseline_logits is not None:
            if reference_baseline_logits.shape != output.baseline_logits.shape:
                raise ValueError("reference and output baseline logits must have the same shape")
            drift = (output.baseline_logits.detach() - reference_baseline_logits.detach()).abs().max()
            self.baseline_max_drift = max(self.baseline_max_drift, float(drift))
            self.baseline_drift_measured = True

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def compute(self) -> dict[str, float | int]:
        if self.examples == 0:
            raise ValueError("no predictions were accumulated")
        precision = self._ratio(self.true_positive, self.true_positive + self.false_positive)
        recall = self._ratio(self.true_positive, self.true_positive + self.false_negative)
        iou = self._ratio(
            self.true_positive,
            self.true_positive + self.false_positive + self.false_negative,
        )
        dice = self._ratio(
            2 * self.true_positive,
            2 * self.true_positive + self.false_positive + self.false_negative,
        )
        if self.normal_areas:
            normal = torch.tensor(self.normal_areas, dtype=torch.float64)
            normal_mean = float(normal.mean())
            normal_p95 = float(torch.quantile(normal, 0.95))
            normal_p99 = float(torch.quantile(normal, 0.99))
        else:
            normal_mean = normal_p95 = normal_p99 = 0.0
        metrics: dict[str, float | int] = {
            "examples": self.examples,
            "auxiliary_accuracy": self._ratio(self.auxiliary_correct, self.examples),
            "auxiliary_precision": self._ratio(
                self.auxiliary_true_positive,
                self.auxiliary_true_positive + self.auxiliary_false_positive,
            ),
            "auxiliary_recall": self._ratio(
                self.auxiliary_true_positive,
                self.auxiliary_true_positive + self.auxiliary_false_negative,
            ),
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_iou": iou,
            "pixel_dice": dice,
            "positive_localization_hit_rate": self._ratio(
                self.positive_hits,
                self.positive_examples,
            ),
            "predicted_positive_fraction": self._ratio(
                self.predicted_positive_pixels,
                self.predicted_pixels,
            ),
            "normal_predicted_area_mean": normal_mean,
            "normal_predicted_area_p95": normal_p95,
            "normal_predicted_area_p99": normal_p99,
        }
        if self.baseline_drift_measured:
            metrics["baseline_logit_max_abs_drift"] = self.baseline_max_drift
        return metrics


def train_multitask_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = True,
    loss_kwargs: Mapping[str, float] | None = None,
    mask_threshold: float = 0.5,
) -> dict[str, float | int]:
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if not optimizer_parameters:
        raise ValueError("optimizer must contain trainable parameters")
    if any(not parameter.requires_grad for parameter in optimizer_parameters):
        raise ValueError("optimizer must not contain a frozen parameter")
    model.train()
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Module):
        classifier.eval()
    accumulator = MultitaskMetricAccumulator(mask_threshold=mask_threshold)
    configured_loss = dict(loss_kwargs or {})
    total_examples = 0
    summed = {"loss": 0.0, "auxiliary_bce": 0.0, "mask_bce": 0.0, "mask_dice_loss": 0.0}
    for batch in loader:
        images = batch.get("image")
        labels = batch.get("label")
        masks = batch.get("mask")
        if not all(isinstance(value, torch.Tensor) for value in (images, labels, masks)):
            raise TypeError("loader batches must contain tensor image, label, and mask fields")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(images)
            losses = compute_multitask_loss(output, labels, masks, **configured_loss)
        losses.total.backward()
        optimizer.step()
        count = int(labels.numel())
        total_examples += count
        summed["loss"] += float(losses.total.detach()) * count
        summed["auxiliary_bce"] += float(losses.auxiliary_bce.detach()) * count
        summed["mask_bce"] += float(losses.mask_bce.detach()) * count
        summed["mask_dice_loss"] += float(losses.mask_dice_loss.detach()) * count
        accumulator.update(output, labels, masks)
    if total_examples == 0:
        raise ValueError("loader produced no examples")
    metrics = accumulator.compute()
    metrics.update({key: value / total_examples for key, value in summed.items()})
    return metrics


@torch.inference_mode()
def evaluate_multitask(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    *,
    device: torch.device,
    use_amp: bool = True,
    loss_kwargs: Mapping[str, float] | None = None,
    mask_threshold: float = 0.5,
) -> dict[str, object]:
    model.eval()
    accumulator = MultitaskMetricAccumulator(mask_threshold=mask_threshold)
    grouped_accumulators: dict[tuple[str, str], MultitaskMetricAccumulator] = {}
    configured_loss = dict(loss_kwargs or {})
    total_examples = 0
    summed = {"loss": 0.0, "auxiliary_bce": 0.0, "mask_bce": 0.0, "mask_dice_loss": 0.0}
    for batch in loader:
        images = batch.get("image")
        labels = batch.get("label")
        masks = batch.get("mask")
        if not all(isinstance(value, torch.Tensor) for value in (images, labels, masks)):
            raise TypeError("loader batches must contain tensor image, label, and mask fields")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(images)
            losses = compute_multitask_loss(output, labels, masks, **configured_loss)
        count = int(labels.numel())
        total_examples += count
        summed["loss"] += float(losses.total) * count
        summed["auxiliary_bce"] += float(losses.auxiliary_bce) * count
        summed["mask_bce"] += float(losses.mask_bce) * count
        summed["mask_dice_loss"] += float(losses.mask_dice_loss) * count
        accumulator.update(output, labels, masks)
        metadata = batch.get("metadata")
        if isinstance(metadata, list):
            for index, item in enumerate(metadata[:count]):
                if float(labels[index]) <= 0.5 or not isinstance(item, Mapping):
                    continue
                family = item.get("synthetic_family")
                parameters = item.get("synthetic_parameters")
                severity = parameters.get("severity") if isinstance(parameters, Mapping) else None
                if not isinstance(family, str) or not isinstance(severity, str):
                    continue
                group = grouped_accumulators.setdefault(
                    (family, severity),
                    MultitaskMetricAccumulator(mask_threshold=mask_threshold),
                )
                group.update(
                    _MetricOutputView(
                        baseline_logits=output.baseline_logits[index : index + 1],
                        auxiliary_logits=output.auxiliary_logits[index : index + 1],
                        mask_logits=output.mask_logits[index : index + 1],
                    ),
                    labels[index : index + 1],
                    masks[index : index + 1],
                )
    if total_examples == 0:
        raise ValueError("loader produced no examples")
    metrics: dict[str, object] = dict(accumulator.compute())
    metrics.update({key: value / total_examples for key, value in summed.items()})
    if grouped_accumulators:
        grouped: dict[str, dict[str, dict[str, float | int]]] = {}
        selected = (
            "examples",
            "pixel_dice",
            "pixel_recall",
            "positive_localization_hit_rate",
        )
        for (family, severity), group in sorted(grouped_accumulators.items()):
            group_metrics = group.compute()
            grouped.setdefault(family, {})[severity] = {
                key: group_metrics[key] for key in selected
            }
        metrics["by_family_severity"] = grouped
    return metrics


def save_multitask_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    base_checkpoint_path: str,
    base_checkpoint_sha256: str,
    backbone_id: str,
    config_digest: str,
    epoch: int,
    metrics: dict[str, float | int],
) -> None:
    sha256_pattern = re.compile(r"[0-9a-f]{64}")
    if not sha256_pattern.fullmatch(base_checkpoint_sha256) or not sha256_pattern.fullmatch(
        config_digest
    ):
        raise ValueError("checkpoint and config digests must be SHA-256 hex strings")
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("checkpoint metrics must be finite")
    state_builder = getattr(model, "multitask_state_dict", None)
    if not callable(state_builder):
        raise TypeError("model must provide multitask_state_dict()")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "base_checkpoint_path": base_checkpoint_path,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "backbone_id": backbone_id,
            "config_digest": config_digest,
            "epoch": epoch,
            "metrics": metrics,
            "multitask_state_dict": state_builder(),
        },
        destination,
    )
