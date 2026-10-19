from __future__ import annotations

import math
import types
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .model import DinoBinaryClassifier


def select_legrad_blocks(blocks: Sequence[object], layer_start: int = -4) -> list[object]:
    start = layer_start if layer_start >= 0 else len(blocks) + layer_start
    if start < 0 or start >= len(blocks):
        raise ValueError(f"layer_start {layer_start} is outside {len(blocks)} blocks")
    return list(blocks[start:])


def relevance_from_attention_gradients(
    attention_gradients: Iterable[torch.Tensor],
    *,
    num_prefix_tokens: int,
) -> torch.Tensor:
    gradients = list(attention_gradients)
    if not gradients:
        raise ValueError("at least one attention-gradient tensor is required")
    expected_shape = gradients[0].shape
    if len(expected_shape) != 4:
        raise ValueError("attention gradients must be [batch, heads, query, key]")
    if any(gradient.shape != expected_shape for gradient in gradients):
        raise ValueError("attention-gradient shapes must match")
    patch_count = expected_shape[-1] - num_prefix_tokens
    side = math.isqrt(patch_count)
    if patch_count <= 0 or side * side != patch_count:
        raise ValueError("patch-token count must form a square grid")
    relevance = torch.zeros(
        (expected_shape[0], patch_count),
        dtype=gradients[0].dtype,
        device=gradients[0].device,
    )
    for gradient in gradients:
        layer_relevance = gradient.clamp_min(0).mean(dim=1).mean(dim=1)
        relevance.add_(layer_relevance[:, num_prefix_tokens:])
    minimum = relevance.min(dim=1, keepdim=True).values
    maximum = relevance.max(dim=1, keepdim=True).values
    return (relevance - minimum) / (maximum - minimum).clamp_min(1e-8)


def _capturing_eva_attention_forward(
    self: nn.Module,
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    batch, token_count, channels = x.shape
    gate = self.gate(x).sigmoid() if getattr(self, "gate", None) is not None else None
    if self.qkv is not None:
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            qkv = (
                self.qkv(x) + qkv_bias
                if self.qkv_bias_separate
                else F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            )
        qkv = qkv.reshape(batch, token_count, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
    else:
        query = self.q_proj(x).reshape(batch, token_count, self.num_heads, -1).transpose(1, 2)
        key = self.k_proj(x).reshape(batch, token_count, self.num_heads, -1).transpose(1, 2)
        value = self.v_proj(x).reshape(batch, token_count, self.num_heads, -1).transpose(1, 2)
    query, key = self.q_norm(query), self.k_norm(key)
    if rope is not None:
        from timm.models.eva import apply_rot_embed_cat

        prefix = self.num_prefix_tokens
        rotate_half = getattr(self, "rotate_half", False)
        query = torch.cat(
            (
                query[:, :, :prefix],
                apply_rot_embed_cat(query[:, :, prefix:], rope, half=rotate_half),
            ),
            dim=2,
        ).type_as(value)
        key = torch.cat(
            (
                key[:, :, :prefix],
                apply_rot_embed_cat(key[:, :, prefix:], rope, half=rotate_half),
            ),
            dim=2,
        ).type_as(value)
    attention = (query * self.scale) @ key.transpose(-2, -1)
    if is_causal:
        attention = attention + attention.new_full((token_count, token_count), float("-inf")).triu_(1)
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attention_bias = torch.zeros_like(attn_mask, dtype=attention.dtype)
            attention_bias.masked_fill_(~attn_mask, float("-inf"))
            attention = attention + attention_bias
        else:
            attention = attention + attn_mask
    attention = self.attn_drop(attention.softmax(dim=-1))
    self._legrad_attention_map = attention
    output = attention @ value
    output = output.transpose(1, 2).reshape(batch, token_count, channels)
    output = self.norm(output)
    if gate is not None:
        output = output * gate
    return self.proj_drop(self.proj(output))


@contextmanager
def capture_eva_attention(blocks: Iterable[nn.Module]) -> Iterator[None]:
    records: list[tuple[nn.Module, bool, object | None]] = []
    try:
        for block in blocks:
            attention = block.attn
            had_instance_forward = "forward" in attention.__dict__
            prior = attention.__dict__.get("forward")
            records.append((attention, had_instance_forward, prior))
            attention.forward = types.MethodType(_capturing_eva_attention_forward, attention)
        yield
    finally:
        for attention, had_instance_forward, prior in records:
            attention.__dict__.pop("_legrad_attention_map", None)
            if had_instance_forward:
                attention.forward = prior
            else:
                attention.__dict__.pop("forward", None)


def token_legrad(
    model: DinoBinaryClassifier,
    pixel_values: torch.Tensor,
    *,
    layer_start: int = -4,
) -> tuple[torch.Tensor, torch.Tensor]:
    timm_backbone = model.backbone.backbone
    selected_blocks = select_legrad_blocks(list(timm_backbone.blocks), layer_start)
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    inputs = pixel_values.detach().clone().requires_grad_(True)
    features: dict[int, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(
            lambda _module, _args, output, index=index: features.__setitem__(index, output)
        )
        for index, block in enumerate(selected_blocks)
    ]
    try:
        with capture_eva_attention(selected_blocks):
            logits, _ = model.forward_with_tokens(inputs)
            gradients: list[torch.Tensor] = []
            for index, block in enumerate(selected_blocks):
                tokens = timm_backbone.norm(features[index])
                patch_tokens = tokens[:, model.num_prefix_tokens :]
                pooled = torch.cat((tokens[:, 0], patch_tokens.mean(dim=1)), dim=1)
                layer_logits = model.head(pooled).squeeze(-1)
                gradients.append(
                    torch.autograd.grad(
                        layer_logits.sum(),
                        block.attn._legrad_attention_map,
                        retain_graph=True,
                    )[0].detach()
                )
            relevance = relevance_from_attention_gradients(
                gradients, num_prefix_tokens=model.num_prefix_tokens
            )
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    side = math.isqrt(relevance.shape[1])
    cams = F.interpolate(
        relevance.reshape(-1, 1, side, side),
        size=pixel_values.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return logits.detach(), cams.detach()
