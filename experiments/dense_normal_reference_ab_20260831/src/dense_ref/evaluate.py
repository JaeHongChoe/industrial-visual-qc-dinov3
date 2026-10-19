"""Fail-closed calibration, shard reconciliation, and native A/B evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .cohort import audit_cohort_reference_isolation

METHODS = ("epoch3_patch", "dense_reference")
_REQUIRED_DIGESTS = ("cohort_sha256", "checkpoint_sha256", "model_sha256", "runtime_sha256", "config_sha256")
_MAP_FIELDS = {"baseline_map", "baseline_valid_mask", "dense_map", "dense_valid_mask", "metadata"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _finite(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).ravel()[np.isfinite(np.asarray(values, dtype=np.float64).ravel())]


def _threshold(values: np.ndarray | Sequence[float], target_load: float) -> float:
    if not 0 < target_load <= 1:
        raise ValueError("target_load must be in (0, 1]")
    finite = _finite(values)
    if not finite.size:
        raise ValueError("threshold calibration requires at least one finite value")
    ordered = np.sort(finite)
    allowed = math.floor(target_load * ordered.size)
    # Strict greater-than is immutable.  Selecting a value from the observed
    # distribution makes ties deterministic and can only reduce alert load.
    return float(ordered[-1] if allowed == 0 else ordered[ordered.size - allowed - 1])


def calibrate_alert_threshold(scores: np.ndarray | Sequence[float], target_load: float = 0.05) -> float:
    """Freeze an image alert threshold with a finite-only strict-``>`` load."""
    return _threshold(scores, target_load)


def calibrate_pixel_threshold(values: np.ndarray | Sequence[float], target_load: float = 0.0001) -> float:
    """Freeze a method-specific pixel threshold from valid calibration cells."""
    return _threshold(values, target_load)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_scores(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, text in enumerate(raw.decode("utf-8").splitlines(), 1):
        if text.strip():
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid score JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"score row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def _source_sha(row: dict[str, Any]) -> str:
    value = row.get("source_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("score/cohort row is missing a 64-character source SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("source SHA-256 is not hexadecimal") from error
    return value


def _audit_score_reference_isolation(rows: Sequence[dict[str, Any]]) -> None:
    """Enforce direct and whole-cohort query/reference disjointness on scores."""
    query_shas: set[str] = set()
    query_captures: set[str] = set()
    reference_shas: set[str] = set()
    reference_captures: set[str] = set()
    for row in rows:
        query_sha = _source_sha(row)
        query_capture = row.get("capture_id")
        if not isinstance(query_capture, str) or not query_capture:
            raise ValueError("score row is missing capture identity")
        references = row.get("reference_membership")
        if not isinstance(references, list):
            raise ValueError("score row is missing reference membership")
        if not references:
            raise ValueError("score row requires at least one reference membership")
        query_shas.add(query_sha)
        query_captures.add(query_capture)
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("score reference membership is not an object")
            reference_sha = reference.get("image_sha256")
            reference_capture = reference.get("capture_id")
            if not isinstance(reference_sha, str) or len(reference_sha) != 64:
                raise ValueError("score reference membership is missing a 64-character image SHA-256")
            if not isinstance(reference_capture, str) or not reference_capture:
                raise ValueError("score reference membership is missing capture identity")
            try:
                int(reference_sha, 16)
            except ValueError as error:
                raise ValueError("score reference membership SHA-256 is not hexadecimal") from error
            if reference_sha == query_sha:
                raise ValueError("query/reference SHA leakage")
            if reference_capture == query_capture:
                raise ValueError("query/reference capture leakage")
            reference_shas.add(reference_sha)
            reference_captures.add(reference_capture)
    if query_shas & reference_shas:
        raise ValueError("global query/reference SHA leakage")
    if query_captures & reference_captures:
        raise ValueError("global query/reference capture leakage")


def _receipt_digests(receipt: dict[str, Any]) -> dict[str, str]:
    nested = receipt.get("digests")
    if not isinstance(nested, dict) or set(nested) != set(_REQUIRED_DIGESTS):
        raise ValueError("receipt is missing required digest set")
    result: dict[str, str] = {}
    for name in _REQUIRED_DIGESTS:
        value = nested.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"receipt has invalid {name}")
        if len(value) != 64:
            raise ValueError("receipt has invalid digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("receipt has invalid digest") from error
        if receipt.get(name) != value:
                raise ValueError("receipt digest disagreement")
        result[name] = value
    mode = receipt.get("checkpoint_load_mode")
    if mode not in {"weights_only_true", "trusted_weights_only_false"}:
        raise ValueError("receipt has invalid checkpoint load mode")
    result["checkpoint_load_mode"] = mode
    return result


def _verify_execution_config(receipt: dict[str, Any], digests: dict[str, str]) -> None:
    """Authenticate rank-specific placement without treating it as score config."""
    execution = receipt.get("execution_config")
    if not isinstance(execution, dict) or set(execution) != {"physical_gpu_index"}:
        raise ValueError("receipt execution config is malformed")
    physical_index = execution.get("physical_gpu_index")
    if type(physical_index) is not int or physical_index < 0:
        raise ValueError("receipt execution config physical GPU index is malformed")
    expected = _json_digest({"scoring_config_sha256": digests["config_sha256"], "physical_gpu_index": physical_index})
    if receipt.get("execution_config_sha256") != expected:
        raise ValueError("receipt execution config digest mismatch")
    gpu = receipt.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError("receipt GPU audit metadata is malformed")
    for name in ("requested_physical_index", "physical_index"):
        value = gpu.get(name)
        if value is not None and value != physical_index:
            raise ValueError("receipt physical GPU audit metadata disagrees with execution config")


def _load_map(row: dict[str, Any], *, expected_config_sha256: str | None = None) -> dict[str, Any]:
    source_sha = _source_sha(row)
    artifact = row.get("map_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("score row is missing map artifact")
    path_value, declared_sha, declared_bytes = artifact.get("path"), artifact.get("sha256"), artifact.get("bytes")
    if not isinstance(path_value, str) or not isinstance(declared_sha, str) or not isinstance(declared_bytes, int):
        raise ValueError("map artifact has invalid path/digest/byte metadata")
    if row.get("map_archive") is not None and row.get("map_archive") != path_value:
        raise ValueError("map archive path mismatch")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"missing map artifact: {path}")
    if path.stat().st_size != declared_bytes or _sha256(path) != declared_sha:
        raise ValueError("map artifact digest mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _MAP_FIELDS:
                raise ValueError("map metadata schema is malformed")
            metadata = json.loads(str(archive["metadata"].item()))
            if not isinstance(metadata, dict):
                raise ValueError("map metadata is malformed")
            if metadata.get("artifact_version") != 1 or metadata.get("source_sha256") != source_sha:
                raise ValueError("map metadata source/version mismatch")
            if metadata.get("methods") != ["dense_reference", "epoch3_patch"]:
                raise ValueError("map metadata methods mismatch")
            if not isinstance(metadata.get("grid_step"), int) or metadata["grid_step"] <= 0:
                raise ValueError("map metadata grid step is malformed")
            if expected_config_sha256 is not None and metadata.get("config_sha256") != expected_config_sha256:
                raise ValueError("map metadata config digest mismatch")
            arrays = {name: np.array(archive[name], copy=True) for name in _MAP_FIELDS - {"metadata"}}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, ValueError) and str(error).startswith("map"):
            raise
        raise ValueError("map metadata is malformed") from error
    for method, value_name, valid_name in (("epoch3_patch", "baseline_map", "baseline_valid_mask"),
                                           ("dense_reference", "dense_map", "dense_valid_mask")):
        values, valid = arrays[value_name], arrays[valid_name]
        if values.ndim != 2 or valid.ndim != 2 or values.shape != valid.shape or valid.dtype != np.bool_:
            raise ValueError(f"map metadata has malformed {method} scalar map/mask")
        if not valid.any() or not np.isfinite(values[valid]).all():
            raise ValueError(f"map metadata has non-finite {method} valid cells")
    return {"path": path, "metadata": metadata, **arrays}


def merge_shards(shard_dirs: Sequence[Path], cohort_rows: Sequence[dict[str, Any]], *, cohort_sha256: str | None = None) -> dict[str, Any]:
    """Merge precisely two complete shards after provenance and artifact checks."""
    if len(shard_dirs) != 2:
        raise ValueError("merge requires exactly two shards")
    if any("references" in row for row in cohort_rows):
        audit_cohort_reference_isolation(cohort_rows)
    expected = {_source_sha({"source_sha256": row.get("image_sha256", row.get("source_sha256"))}) for row in cohort_rows}
    if len(expected) != len(cohort_rows):
        raise ValueError("requested cohort has duplicate source SHA-256")
    expected_cohort_digest = cohort_sha256 or _json_digest(list(cohort_rows))
    loaded: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]] = []
    for directory in map(Path, shard_dirs):
        receipt_path, scores_path = directory / "shard_receipt.json", directory / "scores.jsonl"
        if not receipt_path.is_file() or not scores_path.is_file():
            raise ValueError("successful shard is missing receipt or scores")
        receipt = _read_json(receipt_path)
        if receipt.get("nonmergeable") is not False or receipt.get("failures", receipt.get("counts", {}).get("failures")) != 0 or receipt.get("counts", {}).get("failures") != 0:
            raise ValueError("merge requires successful shards with zero failures")
        scores, score_digest = _read_scores(scores_path)
        digests = _receipt_digests(receipt)
        _verify_execution_config(receipt, digests)
        loaded.append((receipt, scores, digests))
        receipt["_computed_scores_sha256"] = score_digest
    ranks = {receipt.get("rank") for receipt, _, _ in loaded}
    worlds = {receipt.get("world_size") for receipt, _, _ in loaded}
    if ranks != {0, 1} or worlds != {2}:
        raise ValueError("shard rank/world-size disagreement")
    digest_sets = [digests for _, _, digests in loaded]
    if any(digests != digest_sets[0] for digests in digest_sets[1:]):
        raise ValueError("shard digest disagreement")
    if digest_sets[0]["cohort_sha256"] != expected_cohort_digest:
        raise ValueError("cohort digest mismatch")
    all_scores = [row for _, scores, _ in loaded for row in scores]
    score_sources = [_source_sha(row) for row in all_scores]
    if len(score_sources) != len(set(score_sources)):
        raise ValueError("shard source SHA sets are not disjoint")
    if set(score_sources) != expected:
        raise ValueError("merged source SHA union does not exactly match requested cohort")
    _audit_score_reference_isolation(all_scores)
    for receipt, scores, _ in loaded:
        sources = sorted(_source_sha(row) for row in scores)
        declared_sources = receipt.get("source_sha256s")
        if declared_sources != sources or receipt.get("source_set_sha256") != hashlib.sha256("\n".join(sources).encode()).hexdigest():
            raise ValueError("receipt source membership mismatch")
        if receipt.get("counts", {}).get("owned") != len(scores):
            raise ValueError("receipt owned count mismatch")
        if receipt.get("scores_sha256") != receipt["_computed_scores_sha256"]:
            raise ValueError("receipt scores digest mismatch")
        receipt_maps = receipt.get("maps")
        if not isinstance(receipt_maps, list) or len(receipt_maps) != len(scores):
            raise ValueError("receipt map membership mismatch")
        expected_maps = [{"source_sha256": _source_sha(row), **row.get("map_artifact", {})} for row in scores]
        if sorted(receipt_maps, key=lambda x: x["source_sha256"]) != sorted(expected_maps, key=lambda x: x["source_sha256"]):
            raise ValueError("receipt map artifact mismatch")
        for row in scores:
            _load_map(row, expected_config_sha256=digest_sets[0]["config_sha256"])
            methods = row.get("methods")
            if not isinstance(methods, dict) or set(methods) != set(METHODS):
                raise ValueError("score methods must be exactly the frozen method set")
            if any(not isinstance(methods[name].get("score"), (int, float)) or not math.isfinite(methods[name]["score"]) for name in METHODS):
                raise ValueError("score row contains non-finite method score")
    return {"scores": sorted(all_scores, key=_source_sha), "digests": digest_sets[0], "artifact_count": len(all_scores),
            "source_sha256s": sorted(score_sources)}


def validate_labelme(path: Path, expected_sha256: str, *, native_size: tuple[int, int]) -> list[list[tuple[float, float]]]:
    """Validate LabelMe provenance/native geometry and return non-degenerate polygons."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing label file: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError("label digest mismatch")
    data = _read_json(path)
    if (data.get("imageWidth"), data.get("imageHeight")) != native_size:
        raise ValueError("LabelMe native geometry mismatch")
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("LabelMe shapes are malformed")
    polygons: list[list[tuple[float, float]]] = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("shape_type", "polygon") != "polygon":
            continue
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("LabelMe polygon is malformed")
        polygon: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                raise ValueError("LabelMe polygon point is malformed")
            if not (0 <= point[0] <= native_size[0] and 0 <= point[1] <= native_size[1]):
                raise ValueError("LabelMe polygon point is outside native geometry")
            polygon.append((float(point[0]), float(point[1])))
        polygons.append(polygon)
    if not polygons:
        raise ValueError("labeled image contains no valid LabelMe polygons")
    return polygons


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def _hotspots_strict(values: np.ndarray, valid: np.ndarray, threshold: float, count: int) -> list[tuple[int, int, float]]:
    eligible = valid & np.isfinite(values) & (values > threshold)
    seen = np.zeros_like(eligible, dtype=bool)
    output: list[tuple[int, int, float]] = []
    for row, column in np.argwhere(eligible):
        if seen[row, column]:
            continue
        stack, component = [(int(row), int(column))], []
        seen[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in ((current_row - 1, current_column), (current_row + 1, current_column), (current_row, current_column - 1), (current_row, current_column + 1)):
                if 0 <= next_row < eligible.shape[0] and 0 <= next_column < eligible.shape[1] and eligible[next_row, next_column] and not seen[next_row, next_column]:
                    seen[next_row, next_column] = True; stack.append((next_row, next_column))
        maximum = max(float(values[r, c]) for r, c in component)
        peak_row, peak_column = min((r, c) for r, c in component if values[r, c] == maximum)
        output.append((peak_column, peak_row, maximum))
    return sorted(output, key=lambda x: (-x[2], x[1], x[0]))[:count]


def _map_for_method(row: dict[str, Any], method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    archive = _load_map(row)
    return (archive["baseline_map"], archive["baseline_valid_mask"], archive) if method == "epoch3_patch" else (archive["dense_map"], archive["dense_valid_mask"], archive)


def evaluate_scores(calibration_rows: Sequence[dict[str, Any]], evaluation_rows: Sequence[dict[str, Any]], *, target_load: float = .05,
                    pixel_target_load: float = .0001, native_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Evaluate full-development rows using thresholds frozen only from calibration OK."""
    calibration = list(calibration_rows); evaluation = list(evaluation_rows)
    if not calibration:
        raise ValueError("calibration requires calibration_ok rows")
    calibration_shas = [_source_sha(row) for row in calibration]
    evaluation_shas = [_source_sha(row) for row in evaluation]
    if len(calibration_shas) != len(set(calibration_shas)) or len(evaluation_shas) != len(set(evaluation_shas)):
        raise ValueError("duplicate source SHA-256 in calibration/evaluation")
    if set(calibration_shas) & set(evaluation_shas):
        raise ValueError("calibration/evaluation source SHA overlap")
    for row in calibration:
        if row.get("cohort") != "calibration_ok":
            raise ValueError("calibration accepts only calibration_ok rows")
        if row.get("labeled") or str(row.get("source_label", "")).upper() == "NG":
            raise ValueError("calibration rejects NG or labeled rows")
    if any(row.get("cohort") not in {"evaluation_ok", "evaluation_ng"} for row in evaluation):
        raise ValueError("evaluation accepts only evaluation_ok/evaluation_ng rows")
    output: dict[str, Any] = {"mode": "full", "calibration_count": len(calibration), "evaluation_count": len(evaluation), "methods": {}}
    for method in METHODS:
        calibration_scores = np.asarray([row["methods"][method]["score"] for row in calibration], dtype=float)
        image_threshold = calibrate_alert_threshold(calibration_scores, target_load)
        calibration_cells: list[np.ndarray] = []
        for row in calibration:
            values, valid, _ = _map_for_method(row, method)
            calibration_cells.append(values[valid & np.isfinite(values)])
        pixels = np.concatenate(calibration_cells) if calibration_cells else np.empty(0)
        pixel_threshold = calibrate_pixel_threshold(pixels, pixel_target_load)
        alerts = int(np.sum(np.isfinite(calibration_scores) & (calibration_scores > image_threshold)))
        localization = {"labeled_images": 0, "polygons": 0}
        for top_k in (1, 3, 5): localization[f"top_{top_k}"] = {"image_hits": 0, "polygon_hits": 0}
        evaluation_ok = [row for row in evaluation if row["cohort"] == "evaluation_ok"]
        evaluation_ng = [row for row in evaluation if row["cohort"] == "evaluation_ng"]
        ng_alerts = sum(row["methods"][method]["score"] > image_threshold for row in evaluation_ng)
        for row in evaluation_ng:
            if not row.get("labeled"):
                continue
            values, valid, archive = _map_for_method(row, method)
            row_native = tuple(archive["metadata"].get("native_size", ()))
            desired_native = native_size or row_native
            if row_native != desired_native:
                raise ValueError("map/native LabelMe geometry disagreement")
            polygons = validate_labelme(Path(str(row.get("label_path", ""))), str(row.get("label_sha256", "")), native_size=desired_native)
            localization["labeled_images"] += 1; localization["polygons"] += len(polygons)
            hotspots = _hotspots_strict(values, valid, pixel_threshold, 5)
            step = archive["metadata"]["grid_step"]
            for top_k in (1, 3, 5):
                points = [((column + .5) * step, (grid_row + .5) * step) for column, grid_row, _ in hotspots[:top_k]]
                polygon_hits = [any(_point_in_polygon(point, polygon) for point in points) for polygon in polygons]
                localization[f"top_{top_k}"]["polygon_hits"] += sum(polygon_hits)
                localization[f"top_{top_k}"]["image_hits"] += int(any(polygon_hits))
        output["methods"][method] = {
            "image_threshold": {"eligible_count": len(calibration_scores), "alert_count": alerts, "threshold": image_threshold, "realized_load": alerts / len(calibration_scores)},
            "pixel_threshold": {"eligible_count": len(pixels), "alert_count": int(np.sum(pixels > pixel_threshold)), "threshold": pixel_threshold, "realized_load": float(np.mean(pixels > pixel_threshold))},
            "evaluation": {"ok_count": len(evaluation_ok), "ng_count": len(evaluation_ng), "ng_alert_count": int(ng_alerts), "ng_sensitivity": (ng_alerts / len(evaluation_ng) if evaluation_ng else None)},
            "localization": localization,
        }
    return output


def _overlay(original: np.ndarray, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = original.shape[:2]
    finite = values[valid & np.isfinite(values)]
    if not finite.size:
        raise ValueError("cannot render a map without finite valid cells")
    low, high = float(finite.min()), float(finite.max())
    normalized = np.zeros(values.shape, dtype=np.float32) if high == low else np.clip((values - low) / (high - low), 0, 1)
    # Maps normally use the frozen 8-pixel grid.  Resizing by nearest cell also
    # keeps synthetic/test and edge-aligned grids deterministic without ever
    # resizing the original panel.
    expanded = np.asarray(Image.fromarray(normalized).resize((width, height), Image.Resampling.NEAREST))
    expanded_valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)).astype(bool)
    color = np.zeros_like(original, dtype=np.float32)
    color[..., 0] = expanded * 255
    color[..., 1] = (1 - np.abs(expanded * 2 - 1)) * 255
    color[..., 2] = (1 - expanded) * 255
    result = original.astype(np.float32).copy()
    result[expanded_valid] = result[expanded_valid] * .55 + color[expanded_valid] * .45
    return np.rint(result).astype(np.uint8)


def _atomic_png(destination: Path, pixels: np.ndarray) -> None:
    """Publish an already-rendered PNG only after file and directory sync."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        Image.fromarray(pixels).save(temporary_path, format="PNG", optimize=False, compress_level=1)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def render_comparison(source: Path, map_artifact: dict[str, Any], destination: Path, *, source_sha256: str) -> dict[str, Any]:
    """Write deterministic native-height ``original | epoch3 | dense`` PNG."""
    synthetic = {"source_sha256": source_sha256, "map_artifact": map_artifact}
    archive = _load_map(synthetic)
    with Image.open(source) as image:
        original = np.asarray(image.convert("RGB"))
    native_size = tuple(archive["metadata"].get("native_size", ()))
    if (original.shape[1], original.shape[0]) != native_size:
        raise ValueError("source geometry does not match map native geometry")
    baseline = _overlay(original, archive["baseline_map"], archive["baseline_valid_mask"])
    dense = _overlay(original, archive["dense_map"], archive["dense_valid_mask"])
    combined = np.concatenate((original, baseline, dense), axis=1)
    destination = Path(destination)
    _atomic_png(destination, combined)
    artifact = {"path": str(destination), "sha256": _sha256(destination), "bytes": destination.stat().st_size}
    return {"source_sha256": source_sha256, "artifact": artifact, "native_size": list(native_size),
            "panel_labels": ["original", "epoch3_patch", "dense_reference"], "colormap": "deterministic_blue_green_red_v1", "alpha": .45}


def write_metrics(destination: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Publish evaluated metrics atomically, never via a partial JSON file."""
    _atomic_json(Path(destination), metrics)
    return metrics


def _contains_smoke_quality_claim(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and any(term in key.lower() for term in ("calibration", "threshold", "sensitivity", "localization")):
                return True
            if _contains_smoke_quality_claim(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_smoke_quality_claim(child) for child in value)
    return False


def write_run_receipt(destination: Path, *, mode: str, merged: dict[str, Any], metrics: dict[str, Any], visualizations: Sequence[dict[str, Any]],
                      gpu_processes: Sequence[dict[str, Any]], failures: int) -> dict[str, Any]:
    """Atomically receipt reconciled artifacts; smoke deliberately has no quality claims."""
    if mode not in {"smoke", "full"} or failures != 0:
        raise ValueError("receipt requires smoke/full mode and zero failures")
    if mode == "smoke" and _contains_smoke_quality_claim(metrics):
        raise ValueError("smoke receipt forbids threshold-based quality claims")
    for item in visualizations:
        artifact = item.get("artifact", {})
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError("visualization artifact digest mismatch")
    if not gpu_processes or any(not isinstance(item.get("pid"), int) or not item.get("uuid") or not isinstance(item.get("peak_allocated_bytes"), int) for item in gpu_processes):
        raise ValueError("receipt requires GPU/PID/peak-memory evidence")
    elapsed = [float(row.get("telemetry", {}).get("elapsed_seconds", 0)) for row in merged.get("scores", [])]
    if any(not math.isfinite(value) or value < 0 for value in elapsed):
        raise ValueError("receipt has invalid score timing")
    receipt: dict[str, Any] = {"mode": mode, "digests": merged.get("digests"), "source_sha256s": merged.get("source_sha256s", []),
                               "counts": {"scores": len(merged.get("scores", [])), "artifacts": merged.get("artifact_count"), "failures": failures},
                               "metrics": metrics, "visualizations": list(visualizations), "gpu_processes": list(gpu_processes),
                               "timing": {"total_seconds": sum(elapsed), "max_seconds": max(elapsed, default=0)}, "temporary_files": 0}
    if mode == "smoke":
        receipt["quality_claims"] = False
    _atomic_json(Path(destination), receipt)
    return receipt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if text.strip():
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid cohort JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"cohort row is not an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"cohort is empty: {path}")
    if any("references" in row for row in rows):
        audit_cohort_reference_isolation(rows)
    return rows


def _controller_gpu_processes(controller_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    controller = _read_json(controller_path)
    children = controller.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("controller receipt must contain exactly two children")
    processes: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for child in children:
        if not isinstance(child, dict) or not isinstance(child.get("rank"), int) or not isinstance(child.get("pid"), int):
            raise ValueError("controller child is malformed")
        rank = child["rank"]
        if rank not in {0, 1} or rank in seen_ranks:
            raise ValueError("controller ranks must be exactly 0 and 1")
        seen_ranks.add(rank)
        receipt = _read_json(output_dir / f"shard_{rank}" / "shard_receipt.json")
        gpu = receipt.get("gpu")
        if not isinstance(gpu, dict) or not gpu.get("uuid") or not isinstance(gpu.get("peak_allocated_bytes"), int):
            raise ValueError("shard receipt lacks GPU peak-memory evidence")
        if child.get("physical_gpu_index") != gpu.get("requested_physical_index", child.get("physical_gpu_index")):
            raise ValueError("controller/shard physical GPU disagreement")
        processes.append({"pid": child["pid"], "uuid": gpu["uuid"], "peak_allocated_bytes": gpu["peak_allocated_bytes"],
                          "peak_reserved_bytes": gpu.get("peak_reserved_bytes"), "physical_gpu_index": child.get("physical_gpu_index")})
    if seen_ranks != {0, 1}:
        raise ValueError("controller ranks must be exactly 0 and 1")
    return processes


def _scores_for_exact_cohort(merged: dict[str, Any], rows: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    expected = {_source_sha({"source_sha256": row.get("image_sha256", row.get("source_sha256"))}) for row in rows}
    scores = [row for row in merged["scores"] if _source_sha(row) in expected]
    if {_source_sha(row) for row in scores} != expected:
        raise ValueError(f"{label} score membership does not exactly match cohort")
    return scores


def validate_full_partition(merged: dict[str, Any], calibration: Sequence[dict[str, Any]], evaluation_ok: Sequence[dict[str, Any]],
                            evaluation_ng: Sequence[dict[str, Any]]) -> None:
    """Require the three frozen full-evaluation splits to cover each score once."""
    merged_values = merged.get("source_sha256s")
    if not isinstance(merged_values, list):
        raise ValueError("merged cohort is missing source SHA membership")
    merged_sources = [_source_sha({"source_sha256": value}) for value in merged_values]
    if len(merged_sources) != len(set(merged_sources)):
        raise ValueError("merged cohort has duplicate source SHA membership")
    groups = [
        {_source_sha({"source_sha256": row.get("image_sha256", row.get("source_sha256"))}) for row in rows}
        for rows in (calibration, evaluation_ok, evaluation_ng)
    ]
    raw_counts = [len(rows) for rows in (calibration, evaluation_ok, evaluation_ng)]
    if any(len(group) != count for group, count in zip(groups, raw_counts, strict=True)):
        raise ValueError("full split cohorts must be mutually disjoint with no duplicate source SHA")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("full split cohorts must be mutually disjoint")
    if set().union(*groups) != set(merged_sources):
        raise ValueError("full split cohorts must exactly partition merged cohort membership")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dense-ref-evaluate")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--calibration-cohort", type=Path)
    parser.add_argument("--evaluation-ok-cohort", type=Path)
    parser.add_argument("--evaluation-ng-cohort", type=Path)
    args = parser.parse_args(argv)
    cohort_rows = _load_jsonl(args.cohort)
    merged = merge_shards((args.output_dir / "shard_0", args.output_dir / "shard_1"), cohort_rows, cohort_sha256=_sha256(args.cohort))
    gpu_processes = _controller_gpu_processes(args.controller_receipt, args.output_dir)
    visualizations: list[dict[str, Any]] = []
    if args.mode == "smoke":
        for score in merged["scores"]:
            source = score.get("source_path")
            if not isinstance(source, str):
                raise ValueError("smoke score is missing source path")
            visualizations.append(render_comparison(Path(source), score["map_artifact"], args.output_dir / "visualizations" / f"{_source_sha(score)}.png", source_sha256=_source_sha(score)))
        metrics: dict[str, Any] = {"execution": {"reconciled_scores": len(merged["scores"]), "finite_maps": True, "visualizations": len(visualizations)}}
    else:
        paths = (args.calibration_cohort, args.evaluation_ok_cohort, args.evaluation_ng_cohort)
        if any(path is None for path in paths):
            raise ValueError("full evaluation requires separate calibration/evaluation cohort files")
        calibration_cohort = _load_jsonl(args.calibration_cohort)
        evaluation_ok_cohort = _load_jsonl(args.evaluation_ok_cohort)
        evaluation_ng_cohort = _load_jsonl(args.evaluation_ng_cohort)
        validate_full_partition(merged, calibration_cohort, evaluation_ok_cohort, evaluation_ng_cohort)
        calibration_rows = _scores_for_exact_cohort(merged, calibration_cohort, "calibration")
        evaluation_ok = _scores_for_exact_cohort(merged, evaluation_ok_cohort, "evaluation OK")
        evaluation_ng = _scores_for_exact_cohort(merged, evaluation_ng_cohort, "evaluation NG")
        metrics = evaluate_scores(calibration_rows, evaluation_ok + evaluation_ng)
    write_metrics(args.output_dir / "metrics.json", metrics)
    write_run_receipt(args.output_dir / "run_receipt.json", mode=args.mode, merged=merged, metrics=metrics,
                      visualizations=visualizations, gpu_processes=gpu_processes, failures=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
