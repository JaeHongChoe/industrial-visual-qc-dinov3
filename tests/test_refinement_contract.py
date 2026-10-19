from pathlib import Path

import pytest

from patch_qc.refinement_contract import (
    AnchorRow,
    CandidateRow,
    PatchKey,
    SignalRow,
    SourceRow,
    assert_training_source,
    build_run_receipt,
    canonical_digest,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_canonical_digest_ignores_mapping_order() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})


def test_write_jsonl_atomic_returns_digest_and_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "candidate_manifest.jsonl"

    digest = write_jsonl_atomic(path, [{"id": 1}, {"id": 2}])

    assert digest == sha256_file(path)
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]
    assert list(tmp_path.glob("*.tmp")) == []


def test_training_guard_accepts_only_explicit_clean_or_synthetic_anchors() -> None:
    assert_training_source({"source_label": "OK", "source_kind": "clean_anchor"})
    assert_training_source({"source_label": "SYNTHETIC", "source_kind": "synthetic_anchor"})
    with pytest.raises(ValueError, match="training source"):
        assert_training_source({"source_label": "NG", "source_kind": "labelme"})
    with pytest.raises(ValueError, match="training source"):
        assert_training_source({"source_label": "OK", "source_kind": "candidate"})


def test_patch_key_requires_native_box_and_sha() -> None:
    key = PatchKey("a" * 64, (128, 256, 384, 512))

    assert key.patch_id == f"{'a' * 64}:128:256:384:512"
    with pytest.raises(ValueError, match="256 x 256"):
        PatchKey("a" * 64, (0, 0, 128, 128))
    with pytest.raises(ValueError, match="SHA-256"):
        PatchKey("not-a-digest", (0, 0, 256, 256))


def test_records_are_immutable_and_convert_to_dicts() -> None:
    source = SourceRow("a" * 64, "/data/a.jpg", "P1", "G1", "C", "train", 8192, 5464)
    candidate = CandidateRow(
        PatchKey("a" * 64, (0, 0, 256, 256)).patch_id,
        "a" * 64,
        "/data/a.jpg",
        "P1",
        "G1",
        "C",
        "train",
        (0, 0, 256, 256),
        1.5,
        0.8,
        0.9,
        "candidate:0",
    )
    clean = AnchorRow(
        candidate.patch_id,
        candidate.source_sha256,
        candidate.product,
        candidate.group_id,
        candidate.condition_id,
        candidate.box,
        0.01,
        0.99,
        0.9,
        "anchor:0",
        "OK",
        "clean_anchor",
    )
    signal = SignalRow(
        candidate.patch_id,
        candidate.source_sha256,
        candidate.product,
        candidate.group_id,
        candidate.condition_id,
        candidate.box,
        candidate.probability,
        "inside",
        1,
        "signal:0",
        32,
        "b" * 64,
        "OK",
    )

    assert source.to_dict()["format_version"] == 1
    assert candidate.to_dict()["box"] == (0, 0, 256, 256)
    assert clean.to_dict()["source_kind"] == "clean_anchor"
    assert signal.to_dict()["schema_digest"] == "b" * 64
    with pytest.raises((AttributeError, TypeError)):
        source.product = "P2"  # type: ignore[misc]


def test_json_writer_and_run_receipt_are_auditable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    write_json_atomic(artifact, {"z": 1, "a": 2})

    assert artifact.read_text(encoding="utf-8") == '{"a":2,"z":1}\n'
    receipt = build_run_receipt(
        "unit-test",
        digests={"source_manifest": "a" * 64},
        counts={"succeeded": 2},
        metrics={"loss": 0.125},
        failures=0,
        temporary_files=0,
        seeds={"python": 17},
    )
    assert receipt["status"] == "completed"
    assert receipt["runtime"]["python"]
    assert "pytorch" in receipt["runtime"]
    assert receipt["failures"] == 0
    assert receipt["temporary_files"] == 0
    with pytest.raises(ValueError, match="finite"):
        build_run_receipt("bad", digests={}, counts={}, metrics={"loss": float("nan")})
