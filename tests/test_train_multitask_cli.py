from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn


def _module():
    return importlib.import_module("patch_qc.train_multitask")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(tmp_path: Path) -> Path:
    rows = []
    for index, split in enumerate(("train", "holdout"), start=1):
        pixels = np.full((320, 320, 3), 248, dtype=np.uint8)
        yy, xx = np.ogrid[:320, :320]
        pixels[(xx - 160) ** 2 + (yy - 160) ** 2 <= 130**2] = (
            170 + index,
            150 + index,
            140 + index,
        )
        image_path = tmp_path / f"{split}.png"
        Image.fromarray(pixels).save(image_path)
        rows.append(
            {
                "capture_id": f"2026-08-{index:02d}/PRODUCT_A/G{index:03d}",
                "group_id": f"PRODUCT_A::G{index:03d}",
                "product": "PRODUCT_A",
                "object_id": f"G{index:03d}",
                "date": f"2026-08-{index:02d}",
                "source_dir": str(tmp_path),
                "source_label": "OK",
                "split": split,
                "images": [
                    {
                        "path": str(image_path),
                        "condition": {
                            "family": "C",
                            "view_index": 1,
                            "condition_id": "C_L1-01",
                            "is_standard": True,
                        },
                    }
                ],
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


class _FakeBaseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.projection.weight.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).mean((1, 2, 3))


class _FakeDinoMaskMultiTaskProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _FakeBaseClassifier()
        self.mask_head = nn.Conv2d(3, 1, kernel_size=1)
        self.auxiliary_head = nn.Linear(1, 1)

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        with torch.no_grad():
            baseline = self.base(images)
        learned_logits = self.mask_head(images)
        columns = torch.arange(images.shape[-1], device=images.device)
        pattern = torch.where(columns.remainder(2) == 0, -2.0, 2.0)
        mask_logits = learned_logits * 0.01 + pattern.view(1, 1, 1, -1)
        auxiliary_logits = self.auxiliary_head(mask_logits.mean((2, 3))).squeeze(1)
        return SimpleNamespace(
            baseline_logits=baseline,
            auxiliary_logits=auxiliary_logits,
            mask_logits=mask_logits,
        )

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.mask_head.parameters()) + tuple(self.auxiliary_head.parameters())

    def multitask_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "mask_head": self.mask_head.state_dict(),
            "auxiliary_head": self.auxiliary_head.state_dict(),
        }


class _CollapsedMaskProbe(_FakeDinoMaskMultiTaskProbe):
    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        with torch.no_grad():
            baseline = self.base(images)
        learned_logits = self.mask_head(images)
        mask_logits = learned_logits * 0.0 - 20.0
        auxiliary_logits = self.auxiliary_head(mask_logits.mean((2, 3))).squeeze(1)
        return SimpleNamespace(
            baseline_logits=baseline,
            auxiliary_logits=auxiliary_logits,
            mask_logits=mask_logits,
        )


def _smoke_config(manifest: Path, base_checkpoint: Path) -> dict[str, object]:
    return {
        "data": {"patch_size": 256, "patches_per_source": 1},
        "model": {"backbone_id": "tiny-dino", "decoder_channels": 4},
        "train": {
            "seed": 19,
            "epochs": 1,
            "samples_per_epoch": 2,
            "evaluation_samples": 2,
            "positive_probability": 1.0,
            "batch_size": 1,
            "workers": 0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "use_amp": False,
        },
        "loss": {
            "auxiliary_weight": 0.4,
            "mask_bce_weight": 0.8,
            "mask_dice_weight": 1.2,
            "max_positive_weight": 7.0,
        },
        "evaluation": {
            "mask_threshold": 0.4,
            "visualization_panels": 1,
            "baseline_logit_max_abs_drift": 1e-5,
            "minimum_pixel_dice": 0.0,
            "minimum_localization_hit_rate": 0.0,
            "maximum_normal_predicted_area_p95": 1.0,
        },
        "inputs": {
            "expected_manifest_sha256": _sha256(manifest),
            "expected_base_checkpoint_sha256": _sha256(base_checkpoint),
        },
    }


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"trusted-input")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _module()._verify_sha256(source, "0" * 64)


def test_checkpoint_hash_is_verified_before_trusted_pickle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    module = _module()
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unsafe load was reached")),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._load_trusted_epoch3_head(
            _FakeBaseClassifier(),
            checkpoint,
            "f" * 64,
            expected_backbone_id="tiny-dino",
        )


def test_trusted_checkpoint_uses_weights_only_false_and_strict_head_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    base = SimpleNamespace(head=nn.Linear(2, 1))
    expected_weight = torch.tensor([[2.0, -3.0]])
    expected_bias = torch.tensor([0.75])
    checkpoint = tmp_path / "epoch_003.pt"
    torch.save(
        {
            "backbone_id": "tiny-dino",
            "head_state_dict": {"weight": expected_weight, "bias": expected_bias},
        },
        checkpoint,
    )
    original_load = torch.load
    load_kwargs: list[dict[str, object]] = []

    def recorded_load(*args, **kwargs):
        load_kwargs.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", recorded_load)
    module._load_trusted_epoch3_head(
        base,
        checkpoint,
        _sha256(checkpoint),
        expected_backbone_id="tiny-dino",
    )

    assert load_kwargs == [{"map_location": "cpu", "weights_only": False}]
    assert torch.equal(base.head.weight, expected_weight)
    assert torch.equal(base.head.bias, expected_bias)

    malformed = tmp_path / "malformed.pt"
    torch.save(
        {"backbone_id": "tiny-dino", "head_state_dict": {"weight": expected_weight}},
        malformed,
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        module._load_trusted_epoch3_head(
            SimpleNamespace(head=nn.Linear(2, 1)),
            malformed,
            _sha256(malformed),
            expected_backbone_id="tiny-dino",
        )


def test_gpu_info_resolves_slurm_visible_uuid_when_torch_properties_omit_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(module.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        name="NVIDIA H100 80GB HBM3",
        total_memory=80 * 1024**3,
    ))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, GPU-zero\n3, GPU-assigned\n",
        ),
    )

    info = module._gpu_info(torch.device("cuda:0"))

    assert info["uuid"] == "GPU-assigned"
    assert info["physical_visibility"] == "3"


def test_run_seeds_new_multitask_branch_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    config = {
        "model": {"provider": "timm", "backbone_id": "tiny-dino", "decoder_channels": 4},
        "train": {"seed": 20260831},
        "inputs": {
            "expected_manifest_sha256": _sha256(manifest),
            "expected_base_checkpoint_sha256": _sha256(checkpoint),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    initial_states: list[torch.Tensor] = []

    class FakeProbe:
        def __init__(self, base, *, decoder_channels: int) -> None:
            self.initial_state = torch.rand(decoder_channels)

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(module, "load_timm_probe", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "_load_trusted_epoch3_head", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "DinoMaskMultiTaskProbe", FakeProbe)
    monkeypatch.setattr(module, "_code_state", lambda: ("test-commit", False))
    monkeypatch.setattr(
        module,
        "execute_smoke",
        lambda **kwargs: initial_states.append(kwargs["model"].initial_state.clone()),
    )

    torch.manual_seed(1)
    module.run(config_path, manifest, checkpoint, tmp_path / "run-1")
    torch.manual_seed(2)
    module.run(config_path, manifest, checkpoint, tmp_path / "run-2")

    assert torch.equal(initial_states[0], initial_states[1])


def test_baseline_reference_detects_a_base_mutation_after_capture() -> None:
    module = _module()
    model = _FakeDinoMaskMultiTaskProbe()
    loader = [{"image": torch.ones(2, 3, 4, 4)}]
    reference = module._capture_baseline_reference(
        model,
        loader,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        model.base.projection.weight.add_(1.0)
    audit = module._measure_baseline_drift(
        model,
        reference,
        device=torch.device("cpu"),
    )

    assert audit["measured"] is True
    assert audit["examples"] == 2
    assert audit["max_abs"] == pytest.approx(3.0)


def test_execute_smoke_writes_auditable_one_epoch_artifacts_from_real_manifest(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = _write_manifest(tmp_path)
    base_checkpoint = tmp_path / "epoch_003.pt"
    torch.save({"head_state_dict": {}}, base_checkpoint)
    output = tmp_path / "run"
    config = _smoke_config(manifest, base_checkpoint)

    receipt = module.execute_smoke(
        config=config,
        manifest_path=manifest,
        base_checkpoint_path=base_checkpoint,
        output_dir=output,
        model=_FakeDinoMaskMultiTaskProbe(),
        device=torch.device("cpu"),
        code_commit="test-commit",
    )

    expected = {
        "metrics.jsonl",
        "schedule.jsonl",
        "checkpoint_epoch_001.pt",
        "run_receipt.json",
        "resolved_config.yaml",
        "visualizations/index.jsonl",
    }
    assert all((output / relative).is_file() for relative in expected)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    schedule = [json.loads(line) for line in (output / "schedule.jsonl").read_text().splitlines()]
    visualizations = [
        json.loads(line) for line in (output / "visualizations/index.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == len(schedule) == 1
    assert metrics[0]["epoch"] == schedule[0]["epoch"] == 1
    assert metrics[0]["train"]["examples"] == 2
    assert metrics[0]["synthetic_holdout"]["examples"] == 2
    assert metrics[0]["baseline_drift_audit"] == {
        "examples": 1,
        "max_abs": 0.0,
        "measured": True,
    }
    assert schedule[0]["seed"] == 19
    assert schedule[0]["loss"]["max_positive_weight"] == 7.0

    checkpoint = torch.load(
        output / "checkpoint_epoch_001.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["base_checkpoint_sha256"] == _sha256(base_checkpoint)
    assert set(checkpoint["multitask_state_dict"]) == {"mask_head", "auxiliary_head"}

    assert len(visualizations) == 1
    panel = output / visualizations[0]["path"]
    assert panel.is_file()
    assert panel.suffix == ".png"
    assert Image.open(panel).size == (1024, 256)
    assert visualizations[0]["sha256"] == _sha256(panel)

    persisted_receipt = json.loads((output / "run_receipt.json").read_text())
    assert persisted_receipt == receipt
    assert receipt["status"] == "completed"
    assert receipt["digests"]["manifest_sha256"] == _sha256(manifest)
    assert receipt["digests"]["base_checkpoint_sha256"] == _sha256(base_checkpoint)
    assert receipt["extra"]["code_commit"] == "test-commit"
    assert receipt["extra"]["gpu"]["available"] is False
    assert receipt["metrics"]["peak_memory_allocated_bytes"] == 0
    assert receipt["metrics"]["elapsed_seconds"] > 0
    assert receipt["metrics"]["samples_per_second"] > 0
    assert receipt["metrics"]["baseline_drift_audit"]["measured"] is True
    assert receipt["metrics"]["baseline_drift_audit"]["examples"] == 1
    assert receipt["quality_gate"] == {
        "passed": True,
        "failures": [],
        "thresholds": {
            "predicted_positive_fraction": {
                "lower_exclusive": 0.0,
                "upper_exclusive": 1.0,
            },
            "minimum_pixel_dice": 0.0,
            "minimum_localization_hit_rate": 0.0,
            "maximum_normal_predicted_area_p95": 1.0,
        },
    }
    started_at = datetime.fromisoformat(receipt["started_at"].replace("Z", "+00:00"))
    finished_at = datetime.fromisoformat(receipt["finished_at"].replace("Z", "+00:00"))
    assert started_at.tzinfo == timezone.utc
    assert finished_at.tzinfo == timezone.utc
    assert started_at <= finished_at
    assert not list(output.rglob("*.tmp"))


def test_execute_smoke_preserves_artifacts_when_quality_gate_detects_collapse(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = _write_manifest(tmp_path)
    base_checkpoint = tmp_path / "epoch_003.pt"
    torch.save({"head_state_dict": {}}, base_checkpoint)
    output = tmp_path / "collapsed-run"

    receipt = module.execute_smoke(
        config=_smoke_config(manifest, base_checkpoint),
        manifest_path=manifest,
        base_checkpoint_path=base_checkpoint,
        output_dir=output,
        model=_CollapsedMaskProbe(),
        device=torch.device("cpu"),
        code_commit="test-commit",
    )

    assert receipt["status"] == "quality_gate_failed"
    assert receipt["quality_gate"]["passed"] is False
    assert [failure["metric"] for failure in receipt["quality_gate"]["failures"]] == [
        "predicted_positive_fraction"
    ]
    assert (output / "checkpoint_epoch_001.pt").is_file()
    assert (output / "metrics.jsonl").is_file()
    assert (output / "visualizations/index.jsonl").is_file()
    assert json.loads((output / "run_receipt.json").read_text()) == receipt


def test_execute_smoke_rejects_a_non_empty_output_directory_without_writing(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = _write_manifest(tmp_path)
    base_checkpoint = tmp_path / "epoch_003.pt"
    torch.save({"head_state_dict": {}}, base_checkpoint)
    output = tmp_path / "existing-run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing artifact", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        module.execute_smoke(
            config=_smoke_config(manifest, base_checkpoint),
            manifest_path=manifest,
            base_checkpoint_path=base_checkpoint,
            output_dir=output,
            model=_FakeDinoMaskMultiTaskProbe(),
            device=torch.device("cpu"),
            code_commit="test-commit",
        )

    assert list(output.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "existing artifact"
