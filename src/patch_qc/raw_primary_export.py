from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .actual_multitask_eval import _SAFE_IDENTIFIER
from .actual_multitask_inference import render_mask_overlay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_within(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must remain under {resolved_root}") from error
    return resolved_path


def export_raw_primary_comparisons(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    expected_source_receipt_sha256: str,
) -> dict[str, object]:
    source = Path(source_run).resolve()
    destination = Path(output_dir).resolve()
    receipt_path = source / "run_receipt.json"
    predictions_path = source / "predictions.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    if _sha256(receipt_path) != expected_source_receipt_sha256:
        raise ValueError("source run receipt SHA-256 mismatch")
    source_receipt = _load_json(receipt_path)
    if source_receipt.get("stage") != "actual_ok_ng_multitask_qualitative_eval":
        raise ValueError("source receipt stage must be actual_ok_ng_multitask_qualitative_eval")
    if source_receipt.get("status") != "completed":
        raise ValueError("source run is not completed")
    source_artifacts = source_receipt.get("artifacts")
    fixed_settings = source_receipt.get("fixed_settings")
    if not isinstance(source_artifacts, Mapping) or not isinstance(fixed_settings, Mapping):
        raise ValueError("source receipt is missing artifacts or fixed settings")
    expected_predictions_sha256 = source_artifacts.get("predictions.jsonl")
    if not isinstance(expected_predictions_sha256, str):
        raise ValueError("source receipt does not seal predictions.jsonl")
    if _sha256(predictions_path) != expected_predictions_sha256:
        raise ValueError("source predictions SHA-256 mismatch")
    threshold = float(fixed_settings["mask_threshold"])
    if not 0.0 <= threshold < 1.0:
        raise ValueError("source mask threshold is invalid")

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    identifiers: set[str] = set()
    counts = {"comparisons": 0, "ng": 0, "ok": 0}
    output_artifacts: dict[str, str] = {}
    attempt = destination.with_name(f".{destination.name}.tmp")
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir(parents=True)
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("prediction row must be an object")
            identifier = str(row["id"])
            if not _SAFE_IDENTIFIER.fullmatch(identifier):
                raise ValueError(f"invalid prediction id: {identifier!r}")
            cohort = str(row["cohort"])
            if identifier in identifiers or cohort not in {"ok", "ng"}:
                raise ValueError("prediction IDs must be unique and cohorts must be ok or ng")
            identifiers.add(identifier)
            outputs = row.get("outputs")
            if not isinstance(outputs, Mapping):
                raise ValueError("prediction row is missing outputs")
            raw_relative = str(outputs["raw_heatmap"])
            raw_relative_path = Path(raw_relative)
            if (
                not raw_relative
                or raw_relative_path.is_absolute()
                or ".." in raw_relative_path.parts
            ):
                raise ValueError(f"raw heatmap path must be relative and traversal-free: {raw_relative!r}")
            raw_path = _resolve_within(
                source,
                source / raw_relative_path,
                label="raw heatmap path",
            )
            expected_raw_sha256 = source_artifacts.get(raw_relative)
            if not isinstance(expected_raw_sha256, str) or _sha256(raw_path) != expected_raw_sha256:
                raise ValueError(f"raw heatmap SHA-256 mismatch: {raw_relative}")
            original_path = Path(str(row["image_path"]))
            if _sha256(original_path) != row.get("image_sha256"):
                raise ValueError(f"source image SHA-256 mismatch: {identifier}")
            with Image.open(original_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(raw_path) as image:
                raw_score_map = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            if raw_score_map.shape != rgb.shape[:2]:
                raise ValueError(f"raw heatmap shape mismatch: {identifier}")
            overlay = render_mask_overlay(rgb, raw_score_map, threshold=threshold, alpha=0.55)
            comparison = np.concatenate((rgb, overlay), axis=1)
            relative = Path("comparisons") / cohort / f"{identifier}.jpg"
            output_path = _resolve_within(
                attempt,
                attempt / relative,
                label="comparison output path",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(comparison).save(output_path, quality=95, subsampling=0)
            output_artifacts[relative.as_posix()] = _sha256(output_path)
            counts["comparisons"] += 1
            counts[cohort] += 1

        receipt: dict[str, object] = {
            "status": "completed",
            "stage": "raw_primary_comparison_export",
            "display_map": "raw_score_map_8bit",
            "mask_threshold": threshold,
            "overlay_alpha": 0.55,
            "counts": counts,
            "source_run_receipt_sha256": expected_source_receipt_sha256,
            "source_predictions_sha256": expected_predictions_sha256,
            "artifacts": output_artifacts,
        }
        _write_json(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw-primary original/overlay comparisons")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    export_raw_primary_comparisons(
        args.source_run,
        args.output_dir,
        expected_source_receipt_sha256=args.expected_source_receipt_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
