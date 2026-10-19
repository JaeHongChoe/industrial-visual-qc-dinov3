from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .model import DinoBinaryClassifier


def token_grad_cam(
    model: DinoBinaryClassifier,
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw logits and gradient-times-token maps at the input resolution."""
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    inputs = pixel_values.detach().clone().requires_grad_(True)
    logits, patch_tokens = model.forward_with_tokens(inputs)
    patch_count = patch_tokens.shape[1]
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"patch token count {patch_count} does not form a square grid")

    patch_tokens.retain_grad()
    logits.sum().backward()
    gradients = patch_tokens.grad
    if gradients is None:
        raise RuntimeError("patch-token gradients were not retained")
    relevance = torch.relu((patch_tokens * gradients).sum(dim=-1)).reshape(-1, 1, side, side)
    flat = relevance.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None, None]
    maximum = flat.max(dim=1).values[:, None, None, None]
    relevance = (relevance - minimum) / (maximum - minimum).clamp_min(1e-8)
    cams = F.interpolate(
        relevance,
        size=pixel_values.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return logits.detach(), cams.detach()
