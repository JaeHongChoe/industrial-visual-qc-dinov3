import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from patch_qc.model import DinoBinaryClassifier
from patch_qc.refinement_encoding import (
    CandidatePolicy,
    _normalize_batch_on_device,
    calibrate_candidate_policy,
    encode_patch_batch,
    select_candidate_indices,
)
from patch_qc.inference import _normalize_batch


class TinyPatchBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        patch_tokens = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat((patch_tokens.mean(1, keepdim=True), patch_tokens), dim=1))


def test_encode_patch_batch_returns_float32_logits_and_pooled_features() -> None:
    torch.manual_seed(11)
    model = DinoBinaryClassifier(TinyPatchBackbone(), hidden_size=8, dropout=0.0)
    model.train()
    rgb = np.zeros((2, 256, 256, 3), dtype=np.uint8)

    logits, features = encode_patch_batch(model, rgb, "cpu")

    assert logits.shape == (2,)
    assert features.shape == (2, 16)
    assert logits.dtype == np.float32
    assert features.dtype == np.float32
    assert np.isfinite(logits).all()
    assert np.isfinite(features).all()
    assert model.training is True


def test_device_normalization_matches_frozen_cpu_contract() -> None:
    rng = np.random.default_rng(17)
    rgb = rng.integers(0, 256, size=(3, 256, 256, 3), dtype=np.uint8)

    actual = _normalize_batch_on_device(rgb, torch.device("cpu"))

    assert torch.equal(actual, _normalize_batch(rgb))


def test_encode_patch_batch_rejects_unknown_precision() -> None:
    model = DinoBinaryClassifier(TinyPatchBackbone(), hidden_size=8, dropout=0.0)
    rgb = np.zeros((1, 256, 256, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="precision"):
        encode_patch_batch(model, rgb, "cpu", precision="int8")


def test_candidate_selection_keeps_floor_union_top_m_and_cap() -> None:
    values = np.array([0.01, 0.20, 0.35, 0.40, 0.90], dtype=np.float32)
    policy = CandidatePolicy(probability_floor=0.30, minimum_top_m=2, maximum_candidates=3)

    assert select_candidate_indices(values, policy).tolist() == [4, 3, 2]


def test_candidate_selection_breaks_equal_score_ties_by_source_order() -> None:
    values = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    policy = CandidatePolicy(probability_floor=0.9, minimum_top_m=2, maximum_candidates=2)

    assert select_candidate_indices(values, policy).tolist() == [0, 1]


def test_policy_calibration_uses_no_real_ng_and_meets_synthetic_recall() -> None:
    clean = [{"source_label": "OK", "probabilities": [0.01, 0.02, 0.40], "target_indices": []}]
    synthetic = [
        {"source_label": "SYNTHETIC", "probabilities": [0.05, 0.60, 0.10], "target_indices": [1]}
    ]

    policy = calibrate_candidate_policy(
        clean,
        synthetic,
        floors=(0.2, 0.5, 0.7),
        top_m_values=(1, 2),
        maximum_candidates=4,
        target_local_recall=1.0,
    )

    assert select_candidate_indices(np.array(synthetic[0]["probabilities"]), policy).tolist()[0] == 1
    with pytest.raises(ValueError, match="NG"):
        calibrate_candidate_policy([{"source_label": "NG"}], synthetic, (0.5,), (1,), 4, 1.0)


def test_policy_calibration_rejects_an_unreachable_recall_target() -> None:
    synthetic = [
        {"source_label": "SYNTHETIC", "probabilities": [0.9, 0.1], "target_indices": [1]}
    ]

    with pytest.raises(ValueError, match="target local recall"):
        calibrate_candidate_policy([], synthetic, (0.8,), (1,), 1, 1.0)
