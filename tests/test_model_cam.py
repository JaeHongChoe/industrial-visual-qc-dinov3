from types import SimpleNamespace

import pytest
import torch
from torch import nn

import patch_qc.model as model_module
from patch_qc.cam import token_grad_cam
from patch_qc.model import DinoBinaryClassifier, TimmBackboneAdapter


class TinyPatchBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        feature_map = self.projection(pixel_values)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        cls_token = patch_tokens.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.cat((cls_token, patch_tokens), dim=1))


def _multitask_probe(*, dropout: float = 0.0) -> tuple[DinoBinaryClassifier, nn.Module]:
    base = DinoBinaryClassifier(
        TinyPatchBackbone(),
        hidden_size=8,
        dropout=dropout,
        freeze_backbone=False,
    )
    probe = model_module.DinoMaskMultiTaskProbe(base, decoder_channels=8)
    return base, probe


def test_multitask_probe_returns_attribute_accessible_native_outputs() -> None:
    _, model = _multitask_probe()

    output = model(torch.randn(2, 3, 256, 256))

    assert isinstance(output, model_module.MultiTaskOutput)
    assert output.baseline_logits.shape == (2,)
    assert output.auxiliary_logits.shape == (2,)
    assert output.mask_logits.shape == (2, 1, 256, 256)


def test_multitask_probe_rejects_non_square_patch_token_grid() -> None:
    class NonSquareBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            patch_tokens = self.scale * torch.ones(
                pixel_values.shape[0], 6, 8, device=pixel_values.device
            )
            cls_token = patch_tokens.mean(dim=1, keepdim=True)
            return SimpleNamespace(last_hidden_state=torch.cat((cls_token, patch_tokens), dim=1))

    base = DinoBinaryClassifier(
        NonSquareBackbone(), hidden_size=8, dropout=0.0, freeze_backbone=False
    )
    model = model_module.DinoMaskMultiTaskProbe(base, decoder_channels=8)

    with pytest.raises(ValueError, match="square"):
        model(torch.randn(1, 3, 256, 256))


def test_multitask_probe_freezes_entire_base_and_exposes_only_new_parameters() -> None:
    base, model = _multitask_probe()

    trainable = model.trainable_parameters()
    base_parameter_ids = {id(parameter) for parameter in base.parameters()}

    assert isinstance(trainable, tuple)
    assert trainable
    assert all(parameter.requires_grad is False for parameter in base.backbone.parameters())
    assert all(parameter.requires_grad is False for parameter in base.head.parameters())
    assert all(parameter.requires_grad is True for parameter in trainable)
    assert not (base_parameter_ids & {id(parameter) for parameter in trainable})
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_multitask_state_dict_contains_only_new_branch_modules() -> None:
    _, model = _multitask_probe()

    state = model.multitask_state_dict()

    assert set(state) == {
        "token_projection",
        "rgb_256",
        "rgb_128",
        "rgb_64",
        "fuse_64",
        "fuse_128",
        "fuse_256",
        "auxiliary_head",
        "mask_head",
    }
    assert all("base" not in key for key in state)


def test_multitask_probe_backward_populates_gradients_only_in_new_branch() -> None:
    base, model = _multitask_probe()
    output = model(torch.randn(2, 3, 256, 256))

    (output.baseline_logits.sum() + output.auxiliary_logits.sum() + output.mask_logits.mean()).backward()

    assert all(parameter.grad is None for parameter in base.parameters())
    assert all(parameter.grad is not None for parameter in model.trainable_parameters())


def test_multitask_probe_preserves_eval_epoch3_logits_while_branch_trains() -> None:
    torch.manual_seed(41)
    base, model = _multitask_probe(dropout=0.8)
    batch = torch.randn(2, 3, 256, 256)
    base.eval()
    with torch.no_grad():
        expected = base(batch)

    model.train()
    actual = model(batch).baseline_logits

    assert model.training is True
    assert base.training is False
    assert torch.equal(actual, expected)


def test_binary_classifier_returns_one_logit_and_freezes_backbone() -> None:
    backbone = TinyPatchBackbone()
    model = DinoBinaryClassifier(backbone, hidden_size=8, dropout=0.0, freeze_backbone=True)
    batch = torch.randn(2, 3, 256, 256)

    logits = model(batch)

    assert logits.shape == (2,)
    assert all(parameter.requires_grad is False for parameter in backbone.parameters())
    assert any(parameter.requires_grad is True for parameter in model.head.parameters())


def test_encode_with_logits_preserves_forward_and_cam_contract() -> None:
    torch.manual_seed(3)
    model = DinoBinaryClassifier(TinyPatchBackbone(), hidden_size=8, dropout=0.0)
    batch = torch.randn(2, 3, 256, 256)

    logits, pooled, tokens = model.encode_with_logits(batch)

    assert logits.shape == (2,)
    assert pooled.shape == (2, 16)
    assert tokens.shape == (2, 256, 8)
    assert torch.equal(model(batch), logits)
    cam_logits, cams = token_grad_cam(model, batch[:1])
    assert cam_logits.shape == (1,)
    assert cams.shape == (1, 256, 256)


def test_token_grad_cam_returns_normalized_native_patch_maps() -> None:
    torch.manual_seed(9)
    model = DinoBinaryClassifier(TinyPatchBackbone(), hidden_size=8, dropout=0.0)
    batch = torch.randn(2, 3, 256, 256)

    logits, cams = token_grad_cam(model, batch)

    assert logits.shape == (2,)
    assert cams.shape == (2, 256, 256)
    assert torch.isfinite(cams).all()
    assert float(cams.min()) >= 0.0
    assert float(cams.max()) <= 1.0


def test_classifier_rejects_non_square_patch_token_grid() -> None:
    class NonSquareBackbone(nn.Module):
        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=torch.ones(pixel_values.shape[0], 7, 4))

    model = DinoBinaryClassifier(NonSquareBackbone(), hidden_size=4, dropout=0.0)

    try:
        token_grad_cam(model, torch.randn(1, 3, 256, 256))
    except ValueError as error:
        assert "square" in str(error)
    else:
        raise AssertionError("non-square patch token grids must be rejected")


def test_timm_adapter_exposes_patch_tokens_to_shared_binary_head() -> None:
    class TinyTimmBackbone(nn.Module):
        num_features = 8
        num_prefix_tokens = 1

        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(3, 8, kernel_size=16, stride=16, bias=False)

        def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
            patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
            return torch.cat((patches.mean(1, keepdim=True), patches), dim=1)

    adapter = TimmBackboneAdapter(TinyTimmBackbone())
    model = DinoBinaryClassifier(adapter, hidden_size=adapter.hidden_size, dropout=0.0)

    logits = model(torch.randn(2, 3, 256, 256))

    assert adapter.hidden_size == 8
    assert adapter.num_prefix_tokens == 1
    assert logits.shape == (2,)
