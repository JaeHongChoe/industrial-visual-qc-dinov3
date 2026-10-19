from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.check_repository_safety import check_repository, main


RUNTIME_RECEIPT_NAMES = (
    "receipt.json",
    "receipts.json",
    ".receipt.jsonl",
    ".receipts.jsonl",
    "run_receipt.json",
    "run_receipts.jsonl",
    ".run_receipt.json.tmp",
    "run_receipt.json.tmp",
    "audit_receipts.json.part",
    ".audit_receipt.jsonl.attempt",
    "run-receipt.json",
    "run-receipts.jsonl",
    ".run-receipt.json.tmp",
    "audit-receipts.json.part",
)


def test_clean_source_tree_has_no_findings(tmp_path):
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('safe')\n", encoding="utf-8")

    assert check_repository(tmp_path) == []


def test_rejects_excluded_paths_and_large_files_without_disclosing_contents(tmp_path, capsys):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "head.pt").write_text("private-model-data", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "result.png").write_text("private-output-data", encoding="utf-8")
    (tmp_path / ".env").write_text("API_TOKEN=do-not-disclose", encoding="utf-8")
    blob = tmp_path / "blob.dat"
    with blob.open("wb") as handle:
        handle.truncate(11 * 1024 * 1024)

    findings = check_repository(tmp_path)

    assert {(finding.path, finding.code) for finding in findings} == {
        (".env", "forbidden_secret"),
        ("blob.dat", "file_too_large"),
        ("models/head.pt", "forbidden_runtime_directory"),
        ("outputs/result.png", "forbidden_directory"),
    }

    exit_code = main(["--root", str(tmp_path)])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "private-model-data" not in output
    assert "private-output-data" not in output
    assert "API_TOKEN=do-not-disclose" not in output


def test_rejects_root_runtime_artifacts_but_allows_source_model_modules(tmp_path):
    runtime_files = {
        "data/NG.jpg": "raw image",
        "datasets/labels.json": "dataset labels",
        "models/dinov3.bin": "backbone bytes",
        "checkpoints/head.bin": "checkpoint bytes",
        "freeze_receipt.json": "runtime receipt",
    }
    for relative_path, contents in runtime_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    source_module = tmp_path / "src" / "patch_qc" / "models" / "head.py"
    source_module.parent.mkdir(parents=True)
    source_module.write_text("class Head:\n    pass\n", encoding="utf-8")

    findings = check_repository(tmp_path)

    assert {(finding.path, finding.code) for finding in findings} == {
        ("checkpoints/head.bin", "forbidden_runtime_directory"),
        ("data/NG.jpg", "forbidden_runtime_directory"),
        ("datasets/labels.json", "forbidden_runtime_directory"),
        ("freeze_receipt.json", "forbidden_receipt"),
        ("models/dinov3.bin", "forbidden_runtime_directory"),
    }


def test_rejects_nested_runtime_artifacts_but_allows_canonical_and_snapshot_models(tmp_path):
    runtime_files = {
        "experiments/scratch/data/NG.jpg": "raw image",
        "scratch/models/weights.json": "model metadata",
    }
    for relative_path, contents in runtime_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    source_files = {
        "src/patch_qc/models/head.py": "class Head:\n    pass\n",
        "experiments/scratch/src/patch_qc/models/head.py": "class Head:\n    pass\n",
        "src/patch_qc/models/weights.bin": "binary backbone",
        "experiments/scratch/src/patch_qc/models/example.jpg": "image data",
        "src/patch_qc/models/manifest.jsonl": "{\"source\": \"data\"}\n",
    }
    for relative_path, contents in source_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    findings = check_repository(tmp_path)

    assert {(finding.path, finding.code) for finding in findings} == {
        ("experiments/scratch/data/NG.jpg", "forbidden_runtime_directory"),
        ("scratch/models/weights.json", "forbidden_runtime_directory"),
        ("src/patch_qc/models/manifest.jsonl", "forbidden_extension"),
        ("src/patch_qc/models/weights.bin", "forbidden_extension"),
        (
            "experiments/scratch/src/patch_qc/models/example.jpg",
            "forbidden_extension",
        ),
    }


@pytest.mark.parametrize("name", RUNTIME_RECEIPT_NAMES)
def test_rejects_runtime_receipt_name_variants(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("{}\n", encoding="utf-8")

    findings = check_repository(tmp_path)

    assert [(finding.path, finding.code) for finding in findings] == [
        (name, "forbidden_receipt")
    ]


def test_allows_receipt_schema_names(tmp_path: Path) -> None:
    (tmp_path / "receipt_schema.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".receipt_schema.json").write_text("{}\n", encoding="utf-8")

    assert check_repository(tmp_path) == []


@pytest.mark.parametrize("name", RUNTIME_RECEIPT_NAMES)
def test_gitignore_excludes_runtime_receipt_name_variants(name: str) -> None:
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", name],
        cwd=repository_root,
        check=False,
    )

    assert completed.returncode == 0


@pytest.mark.parametrize("name", ("receipt_schema.json", ".receipt_schema.json"))
def test_gitignore_allows_receipt_schema_name(name: str) -> None:
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", name],
        cwd=repository_root,
        check=False,
    )

    assert completed.returncode == 1
