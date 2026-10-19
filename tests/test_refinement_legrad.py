from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patch_qc.model import DinoBinaryClassifier, TimmBackboneAdapter
from patch_qc.refinement_legrad import (
    capture_eva_attention,
    relevance_from_attention_gradients,
    select_legrad_blocks,
    token_legrad,
)


def test_legrad_selects_final_four_blocks_and_normalizes_patch_map() -> None:
    blocks = [object() for _ in range(8)]
    assert select_legrad_blocks(blocks, -4) == blocks[4:]
    gradients = [torch.zeros(2, 3, 21, 21) for _ in range(4)]
    gradients[0][:, :, :, 5] = 2.0

    relevance = relevance_from_attention_gradients(gradients, num_prefix_tokens=5)

    assert relevance.shape == (2, 16)
    assert torch.isfinite(relevance).all()
    assert torch.all((0.0 <= relevance) & (relevance <= 1.0))
    with pytest.raises(ValueError, match="outside"):
        select_legrad_blocks(blocks, -9)


class _TinyEvaAttention(nn.Module):
    def __init__(self, channels: int = 8, heads: int = 2) -> None:
        super().__init__()
        self.num_heads = heads
        self.num_prefix_tokens = 1
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.qkv_bias_separate = False
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Identity()
        self.norm = nn.Identity()
        self.gate = None
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attention @ value).transpose(1, 2).reshape(batch, tokens, channels))


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyEvaAttention()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.attn(values)


class _TinyTimmBackbone(nn.Module):
    num_features = 8
    num_prefix_tokens = 1

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=16, stride=16, bias=False)
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
        self.norm = nn.LayerNorm(8)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        values = torch.cat((patches.mean(1, keepdim=True), patches), dim=1)
        for block in self.blocks:
            values = block(values)
        return self.norm(values)


def test_capture_restores_attention_forward_even_after_exception() -> None:
    block = _TinyBlock()
    assert "forward" not in block.attn.__dict__

    with pytest.raises(RuntimeError, match="stop"):
        with capture_eva_attention([block]):
            assert "forward" in block.attn.__dict__
            raise RuntimeError("stop")

    assert "forward" not in block.attn.__dict__
    assert "_legrad_attention_map" not in block.attn.__dict__


def test_token_legrad_runs_final_four_eva_blocks_and_restores_model_mode() -> None:
    torch.manual_seed(7)
    backbone = TimmBackboneAdapter(_TinyTimmBackbone())
    model = DinoBinaryClassifier(backbone, hidden_size=8, dropout=0.0)
    model.train()

    logits, cams = token_legrad(model, torch.randn(1, 3, 256, 256), layer_start=-4)

    assert logits.shape == (1,)
    assert cams.shape == (1, 256, 256)
    assert torch.isfinite(cams).all()
    assert float(cams.min()) >= 0.0
    assert float(cams.max()) <= 1.0
    assert model.training is True
