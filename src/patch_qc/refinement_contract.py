from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from numbers import Real
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PatchKey:
    source_sha256: str
    box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.source_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        x1, y1, x2, y2 = self.box
        if x1 < 0 or y1 < 0 or x2 - x1 != 256 or y2 - y1 != 256:
            raise ValueError("patch box must be a non-negative native 256 x 256 window")

    @property
    def patch_id(self) -> str:
        return f"{self.source_sha256}:{':'.join(str(value) for value in self.box)}"


class _DictRecord:
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceRow(_DictRecord):
    source_sha256: str
    path: str
    product: str
    group_id: str
    condition_id: str
    split: str
    width: int
    height: int
    source_label: str = "OK"
    format_version: int = 1


@dataclass(frozen=True, slots=True)
class CandidateRow(_DictRecord):
    patch_id: str
    source_sha256: str
    source_path: str
    product: str
    group_id: str
    condition_id: str
    split: str
    box: tuple[int, int, int, int]
    logit: float
    probability: float
    foreground_fraction: float
    feature_key: str
    source_label: str = "OK"
    format_version: int = 1


@dataclass(frozen=True, slots=True)
class AnchorRow(_DictRecord):
    patch_id: str
    source_sha256: str
    product: str
    group_id: str
    condition_id: str
    box: tuple[int, int, int, int]
    probability: float
    tta_stability: float
    foreground_fraction: float
    feature_key: str
    source_label: str
    source_kind: str
    synthetic_family: str = ""
    synthetic_severity: str = ""
    support_fraction: float = 0.0
    roi_relation: str = "unavailable"
    seed: int = 0
    provenance_json: str = "{}"
    model_digest: str = ""
    config_digest: str = ""
    format_version: int = 1


@dataclass(frozen=True, slots=True)
class SignalRow(_DictRecord):
    patch_id: str
    source_sha256: str
    product: str
    group_id: str
    condition_id: str
    box: tuple[int, int, int, int]
    probability: float
    roi_relation: str
    spatial_support: int
    signal_key: str
    feature_dimension: int
    schema_digest: str
    source_label: str
    source_kind: str = "unlabeled_candidate"
    format_version: int = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        temporary.write_text(f"{payload}\n", encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write(f"{payload}\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_training_source(row: Mapping[str, object]) -> None:
    pair = (row.get("source_label"), row.get("source_kind"))
    if pair not in {("OK", "clean_anchor"), ("SYNTHETIC", "synthetic_anchor")}:
        raise ValueError(f"unsupported training source: label={pair[0]!r}, kind={pair[1]!r}")


def _assert_finite(value: object, location: str = "metrics") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        raise ValueError(f"{location} must contain only finite values")


def _runtime_metadata() -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": None,
        "cuda": None,
        "device": {"available": False, "name": None},
    }
    try:
        import torch
    except ImportError:
        return runtime
    runtime["pytorch"] = torch.__version__
    runtime["cuda"] = torch.version.cuda
    available = torch.cuda.is_available()
    runtime["device"] = {
        "available": available,
        "name": torch.cuda.get_device_name(torch.cuda.current_device()) if available else None,
    }
    return runtime


def build_run_receipt(
    stage: str,
    *,
    digests: Mapping[str, str],
    counts: Mapping[str, int],
    metrics: Mapping[str, object] | None = None,
    failures: int = 0,
    temporary_files: int = 0,
    seeds: Mapping[str, int] | None = None,
    artifacts: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    normalized_metrics = dict(metrics or {})
    _assert_finite(normalized_metrics)
    if failures < 0 or temporary_files < 0:
        raise ValueError("failure and temporary file counts must be non-negative")
    receipt: dict[str, object] = {
        "format_version": 1,
        "stage": stage,
        "status": "completed" if failures == 0 and temporary_files == 0 else "failed",
        "digests": dict(digests),
        "counts": dict(counts),
        "metrics": normalized_metrics,
        "failures": failures,
        "temporary_files": temporary_files,
        "seeds": dict(seeds or {}),
        "artifacts": dict(artifacts or {}),
        "runtime": _runtime_metadata(),
    }
    if extra:
        receipt["extra"] = dict(extra)
    return receipt
