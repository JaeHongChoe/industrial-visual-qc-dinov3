from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional


class DinoBinaryClassifier(nn.Module):
    """Frozen DINO-style backbone with a compact binary classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        num_prefix_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_prefix_tokens = num_prefix_tokens or self._infer_prefix_tokens(backbone)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _infer_prefix_tokens(backbone: nn.Module) -> int:
        direct = getattr(backbone, "num_prefix_tokens", None)
        if direct is not None:
            return int(direct)
        config = getattr(backbone, "config", None)
        registers = getattr(config, "num_register_tokens", getattr(config, "num_registers", 0))
        return 1 + int(registers or 0)

    def _validated_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"backbone tokens must have shape [batch, tokens, {self.hidden_size}], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] <= self.num_prefix_tokens:
            raise ValueError("backbone returned no patch tokens")
        return tokens

    def encode_with_logits(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs: Any = self.backbone(pixel_values=pixel_values)
        tokens = self._validated_tokens(outputs.last_hidden_state)
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        pooled = torch.cat((tokens[:, 0], patch_tokens.mean(dim=1)), dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, pooled, patch_tokens

    def forward_with_tokens(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, patch_tokens = self.encode_with_logits(pixel_values)
        return logits, patch_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_tokens(pixel_values)
        return logits


class MultiTaskOutput(NamedTuple):
    baseline_logits: torch.Tensor
    auxiliary_logits: torch.Tensor
    mask_logits: torch.Tensor


def _conv_block(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.GELU(),
    )


class DinoMaskMultiTaskProbe(nn.Module):
    """Frozen binary probe with a trainable RGB/DINO localization branch."""

    def __init__(self, base: DinoBinaryClassifier, *, decoder_channels: int = 32) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

        self.token_projection = nn.Conv2d(base.hidden_size, decoder_channels, kernel_size=1)
        self.rgb_256 = _conv_block(3, decoder_channels)
        self.rgb_128 = _conv_block(decoder_channels, decoder_channels, stride=2)
        self.rgb_64 = _conv_block(decoder_channels, decoder_channels, stride=2)
        self.fuse_64 = _conv_block(decoder_channels * 2, decoder_channels)
        self.fuse_128 = _conv_block(decoder_channels * 2, decoder_channels)
        self.fuse_256 = _conv_block(decoder_channels * 2, decoder_channels)
        self.auxiliary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 1),
        )
        self.mask_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def train(self, mode: bool = True) -> DinoMaskMultiTaskProbe:
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def multitask_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        names = (
            "token_projection",
            "rgb_256",
            "rgb_128",
            "rgb_64",
            "fuse_64",
            "fuse_128",
            "fuse_256",
            "auxiliary_head",
            "mask_head",
        )
        return {name: getattr(self, name).state_dict() for name in names}

    def forward(self, pixel_values: torch.Tensor) -> MultiTaskOutput:
        with torch.no_grad():
            baseline_logits, _, patch_tokens = self.base.encode_with_logits(pixel_values)

        patch_count = patch_tokens.shape[1]
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(f"DINO patch tokens must form a square grid, got {patch_count} tokens")

        semantic = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], self.base.hidden_size, side, side
        )
        semantic = self.token_projection(semantic)
        rgb_256 = self.rgb_256(pixel_values)
        rgb_128 = self.rgb_128(rgb_256)
        rgb_64 = self.rgb_64(rgb_128)

        fused_64 = self.fuse_64(
            torch.cat(
                (functional.interpolate(semantic, size=rgb_64.shape[-2:], mode="bilinear", align_corners=False), rgb_64),
                dim=1,
            )
        )
        auxiliary_logits = self.auxiliary_head(fused_64).squeeze(-1)
        fused_128 = self.fuse_128(
            torch.cat(
                (functional.interpolate(fused_64, size=rgb_128.shape[-2:], mode="bilinear", align_corners=False), rgb_128),
                dim=1,
            )
        )
        fused_256 = self.fuse_256(
            torch.cat(
                (functional.interpolate(fused_128, size=rgb_256.shape[-2:], mode="bilinear", align_corners=False), rgb_256),
                dim=1,
            )
        )
        mask_logits = self.mask_head(fused_256)
        return MultiTaskOutput(baseline_logits, auxiliary_logits, mask_logits)


class TimmBackboneAdapter(nn.Module):
    """Expose timm VisionTransformer features through the Hugging Face-style token contract."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(getattr(backbone, "num_features"))
        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        tokens = self.backbone.forward_features(pixel_values)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("timm backbone forward_features must return [batch, tokens, hidden]")
        return SimpleNamespace(last_hidden_state=tokens)


def load_timm_probe(
    model_name: str,
    *,
    local_files_only: bool = True,
    dropout: float = 0.1,
) -> DinoBinaryClassifier:
    """Load a cached timm DINOv3 model with dynamic 256-pixel positional embeddings."""
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import timm

        timm_backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
    except ImportError as error:
        raise RuntimeError("timm is required to load the cached DINOv3 backbone") from error
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    adapter = TimmBackboneAdapter(timm_backbone)
    return DinoBinaryClassifier(
        adapter,
        hidden_size=adapter.hidden_size,
        dropout=dropout,
        freeze_backbone=True,
        num_prefix_tokens=adapter.num_prefix_tokens,
    )


def load_huggingface_probe(
    model_id: str,
    *,
    token: str | None = None,
    local_files_only: bool = True,
    dropout: float = 0.1,
) -> DinoBinaryClassifier:
    """Load an approved DINOv3 backbone without silently bypassing gated access."""
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError("transformers is required to load DINOv3") from error

    backbone = AutoModel.from_pretrained(
        model_id,
        token=token,
        local_files_only=local_files_only,
    )
    hidden_size = int(backbone.config.hidden_size)
    return DinoBinaryClassifier(backbone, hidden_size=hidden_size, dropout=dropout, freeze_backbone=True)
