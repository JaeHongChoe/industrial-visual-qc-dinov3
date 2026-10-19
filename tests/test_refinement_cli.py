from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import patch_qc.refinement_cli as refinement_cli

from patch_qc.refinement_cli import (
    _audit_source_digest,
    _load_signal_bundles,
    _parse_review_candidates,
    _partition_image_specs,
    _validate_rank_request,
    build_parser,
    _select_limited_image_specs,
    _save_signal_bundles,
    main as refinement_main,
)
from patch_qc.refinement_contract import SignalRow, sha256_file
from patch_qc.refinement_encoding import CandidatePolicy
from patch_qc.refinement_signals import SignalBundle, SignalSchema


CONFIG = "configs/refinement_v1.yaml"


def test_calibration_audit_resolves_an_unsealed_source_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source-bytes")

    assert _audit_source_digest(str(source), "") == sha256_file(source)
    assert _audit_source_digest(str(source), "a" * 64) == "a" * 64


def test_smoke_source_selection_preserves_cross_lot_product_condition_buckets() -> None:
    specs: list[dict[str, str]] = []
    for split in ("train", "calibration", "holdout"):
        specs.append(
            {
                "path": f"/{split}/other.jpg",
                "product": "P0",
                "group_id": f"{split}-other",
                "condition_id": "C0",
                "split": split,
            }
        )
        for index in range(4):
            specs.append(
                {
                    "path": f"/{split}/target-{index}.jpg",
                    "product": "P1",
                    "group_id": f"{split}-G{index}",
                    "condition_id": "C1",
                    "split": split,
                }
            )

    selected = _select_limited_image_specs(specs, 12)

    for split in ("train", "calibration", "holdout"):
        rows = [row for row in selected if row["split"] == split]
        assert len(rows) == 4
        assert {(row["product"], row["condition_id"]) for row in rows} == {("P1", "C1")}
        assert len({row["group_id"] for row in rows}) == 4


def test_cache_shards_are_deterministic_disjoint_and_complete() -> None:
    specs = [
        {
            "path": f"/data/source-{index:03d}.jpg",
            "product": f"P{index % 3}",
            "group_id": f"G{index}",
            "condition_id": f"C{index % 5}",
            "split": "train",
        }
        for index in range(23)
    ]

    shards = [_partition_image_specs(specs, 4, index) for index in range(4)]

    paths = [row["path"] for shard in shards for row in shard]
    assert sorted(paths) == sorted(row["path"] for row in specs)
    assert len(paths) == len(set(paths))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert shards == [_partition_image_specs(specs, 4, index) for index in range(4)]


def test_cache_source_loading_arguments_validate_bounds() -> None:
    parser = build_parser()
    required = [
        "build-cache",
        "--run-root",
        "/tmp/run",
        "--source-manifest",
        "/tmp/manifest.jsonl",
        "--checkpoint",
        "/tmp/epoch_003.pt",
    ]

    parsed = parser.parse_args(
        [
            *required,
            "--workers",
            "12",
            "--prefetch-sources",
            "2",
            "--prefetch-batches",
            "2",
            "--progress-interval-seconds",
            "30",
        ]
    )

    assert parsed.workers == 12
    assert parsed.prefetch_sources == 2
    assert parsed.prefetch_batches == 2
    assert parsed.progress_interval_seconds == 30
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--workers", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--prefetch-sources", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--prefetch-batches", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--progress-interval-seconds", "0"])


@pytest.mark.parametrize(
    ("value", "expected_returncode"),
    (("30", 0), ("1", 0), ("0", 2), ("-1", 2), ("abc", 2)),
)
def test_positive_integer_runtime_validation_rejects_invalid_progress_interval(
    value: str,
    expected_returncode: int,
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_positive_integer "$2" "progress interval"',
            "bash",
            str(Path("slurm/refinement_runtime.sh").resolve()),
            value,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_returncode, result.stderr


def test_build_cache_writes_stage_progress_and_seals_it_in_the_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.png"
    Image.fromarray(np.full((384, 384, 3), 100, dtype=np.uint8)).save(source)
    source_manifest = tmp_path / "source_manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "source_label": "OK",
                "product": "P1",
                "group_id": "G1",
                "split": "train",
                "images": [
                    {
                        "path": str(source),
                        "condition": {"condition_id": "C1"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "version": "candidate_refinement_v1",
                "model": {"checkpoint_sha256": "1" * 64, "precision": "fp32"},
                "runtime": {"batch_size": 4, "seed": 17},
                "data": {"patch_size": 256, "stride": 128},
                "cache": {
                    "shard_size": 16,
                    "reservoir_per_product_condition": 8,
                    "reservoir_per_group": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint")

    class _Model:
        hidden_size = 4

    monkeypatch.setattr(
        refinement_cli,
        "_load_frozen_model",
        lambda _config, _checkpoint: (_Model(), "cpu"),
    )
    monkeypatch.setattr(
        refinement_cli,
        "_calibrate_cache_policy",
        lambda _config, _rows, _encoder: (CandidatePolicy(0.5, 1, 2), [{"sample": 1}]),
    )

    import patch_qc.refinement_encoding as refinement_encoding

    monkeypatch.setattr(
        refinement_encoding,
        "encode_patch_batch",
        lambda _model, rgb_batch, _device, precision: (
            np.zeros(len(rgb_batch), dtype=np.float32),
            np.zeros((len(rgb_batch), 8), dtype=np.float32),
        ),
    )
    run_root = tmp_path / "run"

    assert (
        refinement_main(
            [
                "build-cache",
                "--run-root",
                str(run_root),
                "--config",
                str(config),
                "--source-manifest",
                str(source_manifest),
                "--checkpoint",
                str(checkpoint),
                "--progress-interval-seconds",
                "30",
            ]
        )
        == 0
    )

    progress_path = run_root / "progress.jsonl"
    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [(event["stage"], event["status"]) for event in events] == [
        ("model_load", "started"),
        ("model_load", "completed"),
        ("calibration", "started"),
        ("calibration", "completed"),
        ("cache_build", "started"),
        ("cache_build", "running"),
        ("cache_build", "completed"),
    ]
    receipt = json.loads(
        (run_root / "candidate_cache/candidate_build_receipt.json").read_text()
    )
    assert receipt["progress"] == {
        "events": 7,
        "file": "../progress.jsonl",
        "interval_seconds": 30,
        "sha256": sha256_file(progress_path),
    }
    assert events[-1]["completed_sources"] == receipt["counts"]["sources"] == 1
    assert events[-1]["total_patches"] == receipt["counts"]["total_patches"] == 4


@pytest.mark.parametrize(
    (
        "allocated_cpus",
        "workers",
        "prefetch_sources",
        "prefetch_batches",
        "expected_returncode",
    ),
    (
        (12, 12, 2, 2, 0),
        (12, 0, 2, 2, 0),
        (12, 13, 2, 2, 2),
        (12, 12, 0, 2, 2),
        (12, 12, 2, 0, 2),
    ),
)
def test_source_loading_resource_validation_respects_slurm_cpu_boundary(
    allocated_cpus: int,
    workers: int,
    prefetch_sources: int,
    prefetch_batches: int,
    expected_returncode: int,
) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_refinement_source_loading "$2" "$3" "$4" "$5"',
            "bash",
            str(Path("slurm/refinement_runtime.sh").resolve()),
            str(allocated_cpus),
            str(workers),
            str(prefetch_sources),
            str(prefetch_batches),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_returncode, result.stderr


def _receipt(path: Path, stage: str, digest_value: str, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": stage,
                "status": "completed",
                "failures": 0,
                "temporary_files": 0,
                "digests": {"shared_manifest": digest_value},
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_freeze_requires_complete_upstream_receipts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="candidate_build_receipt"):
        refinement_main(["freeze-policy", "--run-root", str(tmp_path), "--config", CONFIG])


def test_freeze_rejects_failed_or_digest_mismatched_receipts(tmp_path: Path) -> None:
    _receipt(tmp_path / "candidate_cache/candidate_build_receipt.json", "candidate-cache-build", "a" * 64)
    _receipt(tmp_path / "anchors/anchor_build_receipt.json", "anchor-build", "b" * 64)
    _receipt(tmp_path / "refiner/fit_receipt.json", "refiner-fit", "a" * 64)

    with pytest.raises(RuntimeError, match="digest"):
        refinement_main(["freeze-policy", "--run-root", str(tmp_path), "--config", CONFIG])


def test_freeze_writes_display_independent_policy_with_receipt_hashes(tmp_path: Path) -> None:
    digest = "a" * 64
    _receipt(
        tmp_path / "candidate_cache/candidate_build_receipt.json",
        "candidate-cache-build",
        digest,
        candidate_policy={"probability_floor": 0.3, "minimum_top_m": 20, "maximum_candidates": 64},
    )
    _receipt(
        tmp_path / "anchors/anchor_build_receipt.json",
        "anchor-build",
        digest,
        signal_schema_digest="b" * 64,
        normal_memory_digest="c" * 64,
    )
    _receipt(
        tmp_path / "refiner/fit_receipt.json",
        "refiner-fit",
        digest,
        selected_refiner="deterministic_fusion",
        selected_refiner_digest="d" * 64,
    )

    assert refinement_main(["freeze-policy", "--run-root", str(tmp_path), "--config", CONFIG]) == 0
    frozen = json.loads((tmp_path / "frozen_policy.json").read_text(encoding="utf-8"))

    assert frozen["selected_refiner"] == "deterministic_fusion"
    assert frozen["ranking"]["default_top_k"] == 3
    assert "display_threshold" not in json.dumps(frozen["ranking"])
    assert set(frozen["upstream_receipt_sha256"]) == {
        "candidate_build_receipt",
        "anchor_build_receipt",
        "fit_receipt",
    }


def test_sealed_evaluate_rejects_unfrozen_policy(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="frozen policy"):
        refinement_main(
            ["sealed-evaluate", "--run-root", str(tmp_path), "--labelme-root", str(tmp_path / "NG")]
        )


def test_rank_images_rejects_unfrozen_policy_before_model_loading(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-1",
                        "image_path": str(tmp_path / "case-1.png"),
                        "product": "P1",
                        "group_id": "sealed-G1",
                        "condition_id": "unknown",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="frozen policy"):
        refinement_main(
            [
                "rank-images",
                "--run-root",
                str(tmp_path),
                "--request",
                str(request),
                "--checkpoint",
                str(tmp_path / "epoch_003.pt"),
            ]
        )


def test_rank_request_rejects_label_fields_and_duplicate_case_ids() -> None:
    valid = {
        "case_id": "case-1",
        "image_path": "/sealed/case-1.png",
        "product": "P1",
        "group_id": "sealed-G1",
        "condition_id": "unknown",
    }
    with pytest.raises(ValueError, match="label-free"):
        _validate_rank_request({"cases": [{**valid, "shapes": []}]})
    with pytest.raises(ValueError, match="unique"):
        _validate_rank_request({"cases": [valid, valid]})


def test_all_refinement_slurm_entrypoints_have_valid_bash_syntax() -> None:
    for name in (
        "refinement_smoke.sbatch",
        "refinement_build.sbatch",
        "refinement_build_shard.sbatch",
        "refinement_cache_benchmark.sbatch",
        "refinement_fit.sbatch",
        "refinement_rank.sbatch",
        "refinement_merge.sbatch",
        "refinement_sealed_eval.sbatch",
    ):
        result = subprocess.run(["bash", "-n", f"slurm/{name}"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_all_refinement_slurm_entrypoints_allow_isolated_project_root() -> None:
    for name in (
        "refinement_smoke.sbatch",
        "refinement_build.sbatch",
        "refinement_build_shard.sbatch",
        "refinement_cache_benchmark.sbatch",
        "refinement_fit.sbatch",
        "refinement_rank.sbatch",
        "refinement_merge.sbatch",
        "refinement_sealed_eval.sbatch",
    ):
        script = Path("slurm", name).read_text(encoding="utf-8")
        assert "PATCHQC_REFINEMENT_PROJECT_ROOT" in script


def test_all_refinement_slurm_entrypoints_monitor_only_the_allocated_gpu() -> None:
    for name in (
        "refinement_smoke.sbatch",
        "refinement_build.sbatch",
        "refinement_build_shard.sbatch",
        "refinement_cache_benchmark.sbatch",
        "refinement_fit.sbatch",
        "refinement_rank.sbatch",
    ):
        script = Path("slurm", name).read_text(encoding="utf-8")
        assert 'nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}"' in script


def test_signal_artifact_round_trip_preserves_schema_and_float32_values(tmp_path: Path) -> None:
    schema = SignalSchema.default(2)
    row = SignalRow(
        patch_id="patch-1",
        source_sha256="1" * 64,
        product="P1",
        group_id="G1",
        condition_id="C1",
        box=(0, 0, 256, 256),
        probability=0.1,
        roi_relation="inside",
        spatial_support=1,
        signal_key="signal-1",
        feature_dimension=schema.feature_dimension,
        schema_digest=schema.digest,
        source_label="OK",
        source_kind="clean_anchor",
    )
    bundle = SignalBundle(row, np.arange(schema.feature_dimension, dtype=np.float32))

    receipt = _save_signal_bundles(tmp_path, "clean", [bundle])
    restored = _load_signal_bundles(tmp_path, "clean")

    assert receipt["rows"] == 1
    assert receipt["schema_digest"] == schema.digest
    assert restored[0].row == row
    assert restored[0].values.dtype == np.float32
    assert np.array_equal(restored[0].values, bundle.values)


def test_review_request_parser_preserves_native_box_and_review_state() -> None:
    rows = _parse_review_candidates(
        [
            {
                "component_id": "component-1",
                "patch_ids": ["patch-1"],
                "box": {"x0": 128, "y0": 256, "x1": 384, "y1": 512},
                "score": 0.9,
                "probability": 0.8,
                "support": 1,
                "roi_relation": "edge",
                "review_state": "REVIEW",
            }
        ]
    )

    assert rows[0].box.x1 - rows[0].box.x0 == 256
    assert rows[0].box.y1 - rows[0].box.y0 == 256
    assert rows[0].review_state == "REVIEW"
