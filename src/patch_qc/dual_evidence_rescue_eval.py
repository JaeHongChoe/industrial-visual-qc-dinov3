from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .actual_multitask_eval import (
    _utc_now,
    _write_image_atomic,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from .actual_multitask_inference import (
    _sha256_file,
    load_labelme_mask,
    render_mask_overlay,
)
from .dual_evidence_rescue import (
    fuse_primary_weighted_consensus,
    summarize_dual_evidence_results,
    validate_dual_evidence_spec,
)
from .normal_peak_spike import (
    quantize_delivered_score_map,
    score_map_metrics_strict,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required on line {line_number}: {path}")
            rows.append(value)
    return rows


def _sealed_run(
    root: Path,
    *,
    expected_receipt_sha256: str,
    results_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, str]]:
    receipt_path = root / "run_receipt.json"
    if _sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError(f"receipt SHA-256 mismatch: {root}")
    receipt = _load_json(receipt_path)
    artifacts = receipt.get("artifacts")
    if receipt.get("status") != "completed" or not isinstance(artifacts, Mapping):
        raise ValueError(f"input run is not complete and sealed: {root}")
    results_path = root / results_name
    if _sha256_file(results_path) != artifacts.get(results_name):
        raise ValueError(f"results artifact mismatch: {root}")
    return receipt, _load_jsonl(results_path), artifacts


def _row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("input results contain duplicate IDs")
    return indexed


def _runtime_code_digests() -> dict[str, str]:
    package = Path(__file__).parent
    names = (
        "dual_evidence_rescue_eval.py",
        "dual_evidence_rescue.py",
        "actual_multitask_eval.py",
        "actual_multitask_inference.py",
        "normal_peak_spike.py",
    )
    return {name: _sha256_file(package / name) for name in names}


def run(
    *,
    candidate_spec_path: str | Path,
    expected_candidate_spec_sha256: str,
    source_run: str | Path,
    expected_source_receipt_sha256: str,
    normal_peak_run: str | Path,
    expected_normal_peak_receipt_sha256: str,
    auxiliary_max_run: str | Path,
    expected_auxiliary_max_receipt_sha256: str,
    output_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.with_name(f".{destination.name}.attempt-{os.getpid()}")
    if attempt.exists():
        raise FileExistsError(attempt)

    spec_path = Path(candidate_spec_path)
    if _sha256_file(spec_path) != expected_candidate_spec_sha256:
        raise ValueError("candidate spec SHA-256 mismatch")
    spec = _load_json(spec_path)
    settings = validate_dual_evidence_spec(spec)
    expected_inputs = {
        "source_receipt_sha256": expected_source_receipt_sha256,
        "normal_peak_receipt_sha256": expected_normal_peak_receipt_sha256,
        "auxiliary_max_receipt_sha256": expected_auxiliary_max_receipt_sha256,
    }
    if any(settings[key] != value for key, value in expected_inputs.items()):
        raise ValueError("input receipt CLI value differs from candidate spec")
    initial_code = _runtime_code_digests()
    expected_code = {
        str(key): str(value)
        for key, value in dict(spec.get("runtime_code_sha256", {})).items()
    }
    if initial_code != expected_code:
        raise ValueError("runtime code differs from the frozen candidate spec")

    source_root = Path(source_run)
    source_receipt, source_rows, source_artifacts = _sealed_run(
        source_root,
        expected_receipt_sha256=expected_source_receipt_sha256,
        results_name="predictions.jsonl",
    )
    normal_root = Path(normal_peak_run)
    _, normal_rows, normal_artifacts = _sealed_run(
        normal_root,
        expected_receipt_sha256=expected_normal_peak_receipt_sha256,
        results_name="results.jsonl",
    )
    auxiliary_root = Path(auxiliary_max_run)
    _, auxiliary_rows, auxiliary_artifacts = _sealed_run(
        auxiliary_root,
        expected_receipt_sha256=expected_auxiliary_max_receipt_sha256,
        results_name="results.jsonl",
    )
    if len(source_rows) != 115:
        raise ValueError("source evaluation must contain exactly 115 rows")
    if sum(row.get("cohort") == "ok" for row in source_rows) != 27 or sum(
        row.get("cohort") == "ng" for row in source_rows
    ) != 88:
        raise ValueError("source cohort counts differ from the contract")
    source_index = _row_index(source_rows)
    normal_index = _row_index(normal_rows)
    auxiliary_index = _row_index(auxiliary_rows)
    if set(source_index) != set(normal_index) or set(source_index) != set(
        auxiliary_index
    ):
        raise ValueError("input run ID sets differ")

    started_at = _utc_now()
    started = time.perf_counter()
    artifacts: dict[str, str] = {}
    result_rows: list[dict[str, object]] = []
    threshold = float(settings["decision_threshold"])
    try:
        attempt.mkdir(parents=True)
        frozen_spec = attempt / "candidate_spec.json"
        shutil.copyfile(spec_path, frozen_spec)
        artifacts["candidate_spec.json"] = _sha256_file(frozen_spec)

        for index, source_row in enumerate(source_rows, start=1):
            identifier = str(source_row["id"])
            cohort = str(source_row["cohort"])
            normal_row = normal_index[identifier]
            auxiliary_row = auxiliary_index[identifier]
            if normal_row.get("cohort") != cohort or auxiliary_row.get("cohort") != cohort:
                raise ValueError(f"cohort mismatch: {identifier}")

            image_path = Path(str(source_row["image_path"]))
            image_bytes = image_path.read_bytes()
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            if image_sha256 != source_row["image_sha256"]:
                raise RuntimeError(f"source image changed: {identifier}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

            raw_relative = str(dict(source_row["outputs"])["raw_heatmap"])
            p_relative = str(dict(normal_row["outputs"])["p_map"])
            auxiliary_relative = str(
                dict(auxiliary_row["outputs"])["candidate_heatmap"]
            )
            input_paths = (
                (source_root, raw_relative, source_artifacts),
                (normal_root, p_relative, normal_artifacts),
                (auxiliary_root, auxiliary_relative, auxiliary_artifacts),
            )
            input_u8: list[np.ndarray] = []
            for root, relative, sealed_artifacts in input_paths:
                path = root / relative
                if _sha256_file(path) != sealed_artifacts.get(relative):
                    raise ValueError(f"input map artifact mismatch: {identifier}")
                with Image.open(path) as image:
                    input_u8.append(np.asarray(image.convert("L"), dtype=np.uint8))
            raw_u8, p_u8, auxiliary_u8 = input_u8
            if any(array.shape != rgb.shape[:2] for array in input_u8):
                raise ValueError(f"input map shape mismatch: {identifier}")

            candidate_score = fuse_primary_weighted_consensus(
                p_map=p_u8.astype(np.float32) / 255.0,
                auxiliary_map=auxiliary_u8.astype(np.float32) / 255.0,
            )
            candidate_u8, candidate_delivered = quantize_delivered_score_map(
                candidate_score
            )
            raw_delivered = raw_u8.astype(np.float32) / 255.0
            label_path_value = source_row.get("label_path")
            ground_truth = (
                load_labelme_mask(
                    str(label_path_value),
                    image_shape=rgb.shape[:2],
                    expected_sha256=str(source_row["label_sha256"]),
                )
                if label_path_value
                else None
            )
            raw_metrics = score_map_metrics_strict(
                raw_delivered, threshold=threshold, ground_truth=ground_truth
            )
            p_metrics = score_map_metrics_strict(
                p_u8.astype(np.float32) / 255.0,
                threshold=threshold,
                ground_truth=ground_truth,
            )
            candidate_metrics = score_map_metrics_strict(
                candidate_delivered,
                threshold=threshold,
                ground_truth=ground_truth,
            )
            consensus_metrics = {
                "p_ground_truth_positive_pixels": 0,
                "p_candidate_ground_truth_overlap_pixels": 0,
            }
            if ground_truth is not None:
                p_positive = p_u8 >= 128
                candidate_positive = candidate_u8 >= 128
                consensus_metrics = {
                    "p_ground_truth_positive_pixels": int(
                        np.count_nonzero(p_positive & ground_truth)
                    ),
                    "p_candidate_ground_truth_overlap_pixels": int(
                        np.count_nonzero(
                            p_positive & candidate_positive & ground_truth
                        )
                    ),
                }
            comparison = np.concatenate(
                (
                    rgb,
                    render_mask_overlay(
                        rgb,
                        candidate_delivered,
                        threshold=threshold,
                        alpha=0.55,
                    ),
                ),
                axis=1,
            )
            candidate_relative = f"candidate_heatmaps/{cohort}/{identifier}.png"
            comparison_relative = f"comparisons/{cohort}/{identifier}.jpg"
            candidate_path = attempt / candidate_relative
            comparison_path = attempt / comparison_relative
            _write_image_atomic(candidate_path, candidate_u8, format_name="PNG")
            _write_image_atomic(comparison_path, comparison, format_name="JPEG")
            artifacts[candidate_relative] = _sha256_file(candidate_path)
            artifacts[comparison_relative] = _sha256_file(comparison_path)
            result_rows.append(
                {
                    "id": identifier,
                    "cohort": cohort,
                    "image_path": str(image_path),
                    "image_sha256": image_sha256,
                    "label_path": label_path_value,
                    "label_sha256": source_row.get("label_sha256"),
                    "label_status": source_row["label_status"],
                    "metadata": source_row.get("metadata", {}),
                    "prior_usable": bool(normal_row["prior_usable"]),
                    "raw_metrics": raw_metrics,
                    "p_metrics": p_metrics,
                    "candidate_metrics": candidate_metrics,
                    "consensus_metrics": consensus_metrics,
                    "outputs": {
                        "candidate_heatmap": candidate_relative,
                        "comparison": comparison_relative,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(source_rows),
                        "id": identifier,
                        "cohort": cohort,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        results_path = attempt / "results.jsonl"
        _write_jsonl_atomic(results_path, result_rows)
        artifacts["results.jsonl"] = _sha256_file(results_path)
        summary = summarize_dual_evidence_results(result_rows)
        if _runtime_code_digests() != initial_code:
            raise RuntimeError("runtime code changed during evaluation")
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        receipt: dict[str, object] = {
            "status": "completed",
            "stage": "primary_weighted_consensus_v1_spike",
            "interpretation": "development_set_selected_diagnostic_not_sealed_test",
            "candidate_status": (
                "screening_gate_passed"
                if bool(dict(summary["screening_gate"])["passed"])
                else "screening_gate_rejected"
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "counts": {
                "total": len(result_rows),
                "ok": sum(row["cohort"] == "ok" for row in result_rows),
                "ng": sum(row["cohort"] == "ng" for row in result_rows),
                "candidate_heatmaps": len(result_rows),
                "comparisons": len(result_rows),
                "artifacts": len(artifacts),
            },
            "settings": {
                "decision_threshold": threshold,
                "decision_comparator": "strictly_greater_than",
                "formula": spec["formula"],
                "overlay_alpha": 0.55,
            },
            "digests": {
                "candidate_spec_sha256": expected_candidate_spec_sha256,
                **expected_inputs,
                "source_predictions_sha256": source_artifacts["predictions.jsonl"],
                "normal_peak_results_sha256": normal_artifacts["results.jsonl"],
                "auxiliary_max_results_sha256": auxiliary_artifacts["results.jsonl"],
                "code": initial_code,
            },
            "source_receipt_status": source_receipt["status"],
            "summary": summary,
            "code": {"commit": commit, "dirty": True},
            "artifacts": artifacts,
            "execution_failures": 0,
            "failures": 0,
        }
        _write_json_atomic(attempt / "run_receipt.json", receipt)
        attempt.replace(destination)
        return receipt
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen dual-evidence rescue spike")
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--expected-candidate-spec-sha256", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    parser.add_argument("--normal-peak-run", required=True)
    parser.add_argument("--expected-normal-peak-receipt-sha256", required=True)
    parser.add_argument("--auxiliary-max-run", required=True)
    parser.add_argument("--expected-auxiliary-max-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(
        candidate_spec_path=args.candidate_spec,
        expected_candidate_spec_sha256=args.expected_candidate_spec_sha256,
        source_run=args.source_run,
        expected_source_receipt_sha256=args.expected_source_receipt_sha256,
        normal_peak_run=args.normal_peak_run,
        expected_normal_peak_receipt_sha256=args.expected_normal_peak_receipt_sha256,
        auxiliary_max_run=args.auxiliary_max_run,
        expected_auxiliary_max_receipt_sha256=args.expected_auxiliary_max_receipt_sha256,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
