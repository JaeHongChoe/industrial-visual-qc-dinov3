#!/usr/bin/env python3
"""Fail closed when code-only repository content violates its storage policy."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SKIPPED_DIRECTORIES = frozenset({".git", ".venv", ".superpowers"})
FORBIDDEN_DIRECTORIES = frozenset({"artifacts", "logs", "outputs", "results", "runs"})
FORBIDDEN_RUNTIME_DIRECTORIES = frozenset(
    {"backbones", "checkpoints", "data", "datasets", "models", "weights"}
)
SOURCE_CODE_SUFFIXES = frozenset({".py", ".pyi"})
FORBIDDEN_SUFFIXES = (
    ".bin",
    ".bmp",
    ".ckpt",
    ".csv",
    ".engine",
    ".feather",
    ".gif",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
    ".tsv",
    ".webp",
    ".zip",
    ".bundle",
)
SECRET_NAME_FRAGMENTS = ("credential", "secret", "token")
RUNTIME_RECEIPT_NAME = re.compile(
    r"^\.?(?:(?:.+[_-])?receipts?)\.jsonl?(?:\.(?:tmp|part|attempt))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One policy violation, expressed only with a repository-relative path."""

    path: str
    code: str
    message: str


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _forbidden_extension(name: str) -> bool:
    lower_name = name.lower()
    return (
        lower_name.endswith(FORBIDDEN_SUFFIXES)
        or lower_name.endswith(".tar")
        or ".tar." in lower_name
    )


def _forbidden_secret_name(name: str) -> bool:
    lower_name = name.lower()
    return lower_name.startswith(".env") or any(
        fragment in lower_name for fragment in SECRET_NAME_FRAGMENTS
    )


def _is_runtime_receipt(name: str) -> bool:
    return RUNTIME_RECEIPT_NAME.fullmatch(name) is not None


def _is_source_models_tree(path_parts: tuple[str, ...]) -> bool:
    return "models" in path_parts and "src" in path_parts[: path_parts.index("models")]


def _is_source_model_module(path_parts: tuple[str, ...], name: str) -> bool:
    return _is_source_models_tree(path_parts) and Path(name).suffix.lower() in SOURCE_CODE_SUFFIXES


def _is_forbidden_runtime_path(path_parts: tuple[str, ...], name: str) -> bool:
    for part in path_parts:
        if part.lower() not in FORBIDDEN_RUNTIME_DIRECTORIES:
            continue
        if part.lower() == "models" and _is_source_model_module(path_parts, name):
            continue
        return True
    return False


def check_repository(root: Path, max_bytes: int = 10 * 1024 * 1024) -> list[Finding]:
    """Return every content-policy violation below ``root`` without reading files."""

    root = root.resolve()
    findings: list[Finding] = []

    for current_directory, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current_directory)
        directory_names[:] = sorted(
            name for name in directory_names if name not in SKIPPED_DIRECTORIES
        )
        for name in sorted(file_names):
            path = current_path / name
            relative_path = _relative_path(root, path)
            path_parts = Path(relative_path).parts[:-1]

            if _is_source_models_tree(path_parts) and _forbidden_extension(name):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_extension",
                        message="model, cache, image, dataset, archive, or bundle files are excluded",
                    )
                )
                continue
            if _is_forbidden_runtime_path(path_parts, name):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_runtime_directory",
                        message="data, dataset, model, checkpoint, backbone, or weight paths are excluded",
                    )
                )
                continue
            if any(part.lower() in FORBIDDEN_DIRECTORIES for part in path_parts):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_directory",
                        message="stored output, artifact, log, run, or result paths are excluded",
                    )
                )
                continue
            if _is_runtime_receipt(name):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_receipt",
                        message="actual runtime receipt files are excluded",
                    )
                )
                continue
            if _forbidden_secret_name(name):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_secret",
                        message="environment, secret, credential, and token files are excluded",
                    )
                )
                continue
            if _forbidden_extension(name):
                findings.append(
                    Finding(
                        path=relative_path,
                        code="forbidden_extension",
                        message="model, cache, image, dataset, archive, or bundle files are excluded",
                    )
                )
                continue
            if path.stat().st_size > max_bytes:
                findings.append(
                    Finding(
                        path=relative_path,
                        code="file_too_large",
                        message=f"file exceeds the {max_bytes}-byte repository limit",
                    )
                )

    return findings


def _positive_mib(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that a code-only repository has no prohibited content."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument(
        "--max-mib",
        type=_positive_mib,
        default=10.0,
        help="maximum allowed file size in MiB (default: 10)",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"repository root is not a directory: {args.root}")

    max_bytes = int(args.max_mib * 1024 * 1024)
    findings = check_repository(args.root, max_bytes=max_bytes)
    for finding in findings:
        print(f"{finding.code}: {finding.path} ({finding.message})")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
