from __future__ import annotations

from pathlib import Path

import yaml


def test_refinement_config_preserves_label_free_native_contract() -> None:
    config = yaml.safe_load(Path("configs/refinement_v1.yaml").read_text(encoding="utf-8"))

    assert config["version"] == "candidate_refinement_v1"
    assert config["model"]["checkpoint_sha256"] == "6db93522444871c1ff2cf021e341a46bc855681465f8e71b60bb89eaf1f9c3ab"
    assert config["model"]["backbone_id"] == "vit_large_patch16_dinov3.lvd1689m"
    assert config["model"]["precision"] == "fp16"
    assert config["data"]["patch_size"] == 256
    assert config["data"]["stride"] == 128
    assert config["candidate_policy"]["floor_grid"] == [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    assert config["candidate_policy"]["top_m_grid"] == [5, 10, 20, 40]
    assert config["cache"]["reservoir_per_product_condition"] == 512
    assert config["cache"]["reservoir_per_group"] == 2
    assert config["ranking"]["default_top_k"] == 3
    assert config["ranking"]["expanded_top_k"] == 5
    assert config["training"]["manual_labels"] == 0
    assert config["evaluation"]["real_ng_usage"] == "sealed_evaluation_only"
    assert config["runtime"]["required_visible_gpus"] == 1


def test_refinement_config_keeps_display_threshold_out_of_ranking() -> None:
    config = yaml.safe_load(Path("configs/refinement_v1.yaml").read_text(encoding="utf-8"))

    assert "display_threshold" not in config["ranking"]
    assert 0.0 <= config["display"]["threshold"] <= 1.0
