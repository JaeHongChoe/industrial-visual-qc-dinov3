from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .inference import Box
from .refinement_contract import (
    build_run_receipt,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .refinement_ranking import ReviewCandidate


@dataclass(frozen=True, slots=True)
class LabelShape:
    case_id: str
    shape_id: str
    label: str
    points: tuple[tuple[float, float], ...]
    source_sha256: str
    shape_type: str = "polygon"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelCase:
    case_id: str
    product: str
    lot: str
    image_path: str
    source_sha256: str
    json_sha256: str
    width: int
    height: int
    shapes: tuple[LabelShape, ...]

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["shapes"] = [shape.to_dict() for shape in self.shapes]
        return row


def _shape_from_dict(case_id: str, source_sha256: str, index: int, value: Mapping[str, object]) -> LabelShape:
    shape_type = str(value.get("shape_type") or "polygon").casefold()
    if shape_type not in {"polygon", "rectangle", "circle"}:
        raise ValueError(f"unsupported LabelMe shape_type: {shape_type}")
    raw_points = value.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("LabelMe shape points must be a list")
    points = tuple((float(point[0]), float(point[1])) for point in raw_points)
    minimum = 3 if shape_type == "polygon" else 2
    if len(points) < minimum or any(not math.isfinite(coordinate) for point in points for coordinate in point):
        raise ValueError(f"invalid {shape_type} points in {case_id}")
    return LabelShape(
        case_id=case_id,
        shape_id=f"shape-{index:04d}",
        label=str(value.get("label") or "unlabeled"),
        points=points,
        source_sha256=source_sha256,
        shape_type=shape_type,
    )


def seal_labelme_manifest(labelme_root: str | Path, output_path: str | Path) -> str:
    root = Path(labelme_root).resolve()
    destination = Path(output_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if destination.exists():
        raise FileExistsError(destination)
    cases: list[dict[str, object]] = []
    for json_path in sorted(root.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_reference = payload.get("imagePath")
        if not image_reference:
            raise ValueError(f"LabelMe JSON has no imagePath: {json_path}")
        image_path = (json_path.parent / str(image_reference)).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        source_sha256 = sha256_file(image_path)
        relative = json_path.relative_to(root)
        case_id = relative.with_suffix("").as_posix()
        parts = relative.parts
        product = parts[0] if len(parts) >= 2 else "unknown"
        lot = parts[1] if len(parts) >= 3 else "unknown"
        with Image.open(image_path) as image:
            width, height = image.size
        declared_width = int(payload.get("imageWidth") or width)
        declared_height = int(payload.get("imageHeight") or height)
        if (declared_width, declared_height) != (width, height):
            raise ValueError(f"LabelMe dimensions do not match image bytes: {json_path}")
        shapes = tuple(
            _shape_from_dict(case_id, source_sha256, index, value)
            for index, value in enumerate(payload.get("shapes") or [])
        )
        case = LabelCase(
            case_id=case_id,
            product=product,
            lot=lot,
            image_path=str(image_path),
            source_sha256=source_sha256,
            json_sha256=sha256_file(json_path),
            width=width,
            height=height,
            shapes=shapes,
        )
        cases.append(case.to_dict())
    if not cases:
        raise ValueError("LabelMe root contains no JSON cases")
    write_json_atomic(
        destination,
        {
            "format_version": 1,
            "labelme_root": str(root),
            "cases": cases,
        },
    )
    return sha256_file(destination)


def _shape_bounds(shape: LabelShape) -> tuple[float, float, float, float]:
    points = np.asarray(shape.points, dtype=np.float64)
    if shape.shape_type == "circle":
        center = points[0]
        radius = float(np.linalg.norm(points[1] - center))
        return center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def candidate_intersects_shape(
    box: Box,
    shape: LabelShape,
    image_shape: tuple[int, int],
) -> bool:
    height, width = image_shape
    x0, y0 = max(0, box.x0), max(0, box.y0)
    x1, y1 = min(width, box.x1), min(height, box.y1)
    if x0 >= x1 or y0 >= y1:
        return False
    shape_x0, shape_y0, shape_x1, shape_y1 = _shape_bounds(shape)
    if min(x1, shape_x1) <= max(x0, shape_x0) or min(y1, shape_y1) <= max(y0, shape_y0):
        return False
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    points = np.rint(np.asarray(shape.points, dtype=np.float64) - np.array([x0, y0])).astype(np.int32)
    if shape.shape_type == "polygon":
        cv2.fillPoly(mask, [points], 1)
    elif shape.shape_type == "rectangle":
        left = int(min(points[0, 0], points[1, 0]))
        top = int(min(points[0, 1], points[1, 1]))
        right = int(max(points[0, 0], points[1, 0]))
        bottom = int(max(points[0, 1], points[1, 1]))
        cv2.rectangle(mask, (left, top), (right, bottom), 1, -1)
    elif shape.shape_type == "circle":
        radius = max(1, int(round(np.linalg.norm(points[1] - points[0]))))
        cv2.circle(mask, tuple(points[0]), radius, 1, -1)
    else:
        raise ValueError(f"unsupported LabelMe shape_type: {shape.shape_type}")
    return bool(mask.any())


def _candidate_box(candidate: ReviewCandidate | Box) -> Box:
    return candidate if isinstance(candidate, Box) else candidate.box


def evaluate_topk(
    cases: Sequence[LabelCase],
    ranked_candidates: Mapping[str, Sequence[ReviewCandidate | Box]],
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, object]:
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Top-K values must be positive")
    labeled_cases = [case for case in cases if case.shapes]
    shape_denominator = sum(len(case.shapes) for case in labeled_cases)
    image_denominator = len(labeled_cases)
    if shape_denominator == 0:
        raise ValueError("localization evaluation has no LabelMe shapes")
    product_denominators = Counter(
        case.product for case in labeled_cases for _ in case.shapes
    )
    lot_denominators = Counter(case.lot for case in labeled_cases for _ in case.shapes)
    label_denominators = Counter(shape.label for case in labeled_cases for shape in case.shapes)
    metrics: dict[str, object] = {
        "shape_denominator": shape_denominator,
        "image_denominator": image_denominator,
        "qualitative_unlabeled_images": len(cases) - image_denominator,
        "denominators_by_product": dict(sorted(product_denominators.items())),
        "denominators_by_lot": dict(sorted(lot_denominators.items())),
        "denominators_by_label": dict(sorted(label_denominators.items())),
    }
    case_metrics: list[dict[str, object]] = []
    for k in sorted(set(ks)):
        shape_results: dict[str, bool] = {}
        image_results: dict[str, bool] = {}
        for case in labeled_cases:
            candidates = ranked_candidates.get(case.case_id, ())[:k]
            case_hits: list[bool] = []
            for shape in case.shapes:
                hit = any(
                    candidate_intersects_shape(
                        _candidate_box(candidate), shape, (case.height, case.width)
                    )
                    for candidate in candidates
                )
                shape_results[f"{case.case_id}::{shape.shape_id}"] = hit
                case_hits.append(hit)
            image_results[case.case_id] = any(case_hits)
            case_metrics.append(
                {
                    "case_id": case.case_id,
                    "product": case.product,
                    "lot": case.lot,
                    "k": k,
                    "shape_hits": sum(case_hits),
                    "shape_denominator": len(case_hits),
                    "image_hit": any(case_hits),
                    "candidate_count": len(candidates),
                }
            )
        metrics[f"shape_recall_at_{k}"] = sum(shape_results.values()) / shape_denominator
        metrics[f"image_recall_at_{k}"] = sum(image_results.values()) / image_denominator
        metrics[f"shape_results_at_{k}"] = shape_results
        metrics[f"image_results_at_{k}"] = image_results
    metrics["case_metrics"] = case_metrics
    return metrics


def compare_with_baseline(
    refined: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    refined_shapes = dict(refined.get("shape_results_at_5", {}))
    baseline_shapes = dict(baseline.get("shape_results_at_5", {}))
    refined_images = dict(refined.get("image_results_at_5", {}))
    baseline_images = dict(baseline.get("image_results_at_5", {}))

    def differences(
        current: Mapping[str, object], prior: Mapping[str, object]
    ) -> tuple[list[str], list[str], list[str]]:
        keys = sorted(set(current) | set(prior))
        gained = [key for key in keys if bool(current.get(key)) and not bool(prior.get(key))]
        lost = [key for key in keys if bool(prior.get(key)) and not bool(current.get(key))]
        unchanged = [key for key in keys if bool(prior.get(key)) == bool(current.get(key))]
        return gained, lost, unchanged

    gained_shapes, lost_shapes, unchanged_shapes = differences(refined_shapes, baseline_shapes)
    gained_images, lost_images, unchanged_images = differences(refined_images, baseline_images)
    non_decrease = float(refined["shape_recall_at_5"]) >= float(baseline["shape_recall_at_5"])
    return {
        "gained_shapes_at_5": gained_shapes,
        "lost_shapes_at_5": lost_shapes,
        "unchanged_shapes_at_5": unchanged_shapes,
        "gained_images_at_5": gained_images,
        "lost_images_at_5": lost_images,
        "unchanged_images_at_5": unchanged_images,
        "shape_recall_at_5_delta": float(refined["shape_recall_at_5"])
        - float(baseline["shape_recall_at_5"]),
        "image_recall_at_5_delta": float(refined["image_recall_at_5"])
        - float(baseline["image_recall_at_5"]),
        "acceptance": {"shape_recall_at_5_non_decrease": non_decrease},
    }


def _case_from_dict(value: Mapping[str, object]) -> LabelCase:
    shapes = tuple(LabelShape(**shape) for shape in value.get("shapes", ()))
    fields = {key: item for key, item in value.items() if key != "shapes"}
    return LabelCase(**fields, shapes=shapes)


def run_sealed_evaluation(
    frozen_config: Mapping[str, object],
    frozen_artifacts: Mapping[str, object],
    labelme_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    manifest_path = Path(labelme_manifest)
    manifest_digest = sha256_file(manifest_path)
    config_digests = dict(frozen_config.get("digests", {}))
    artifact_digests = dict(frozen_artifacts.get("digests", {}))
    required = {
        "config",
        "candidate_policy",
        "feature_schema",
        "normal_memory",
        "selected_refiner",
        "labelme_manifest",
    }
    if set(config_digests) < required or config_digests != artifact_digests:
        raise RuntimeError("frozen policy and artifact digests do not match")
    if config_digests["labelme_manifest"] != manifest_digest:
        raise RuntimeError("sealed LabelMe manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = tuple(_case_from_dict(value) for value in manifest.get("cases", ()))
    if not cases:
        raise RuntimeError("sealed LabelMe manifest contains no cases")
    evaluation_hashes = {
        digest for case in cases for digest in (case.source_sha256, case.json_sha256)
    }
    training_hashes = {str(value) for value in frozen_artifacts.get("training_hashes", ())}
    collisions = sorted(evaluation_hashes & training_hashes)
    if collisions:
        raise RuntimeError(
            f"sealed evaluation is not disjoint from training artifacts: {collisions}"
        )
    ranked = frozen_artifacts.get("ranked_candidates")
    baseline = frozen_artifacts.get("baseline_candidates")
    if not isinstance(ranked, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("sealed evaluation requires refined and baseline candidate mappings")
    metrics = evaluate_topk(cases, ranked, ks=(3, 5))
    baseline_metrics = evaluate_topk(cases, baseline, ks=(3, 5))
    comparison = compare_with_baseline(metrics, baseline_metrics)

    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "evaluation_manifest.json",
        {"manifest_sha256": manifest_digest, **manifest},
    )
    case_metrics = metrics.pop("case_metrics")
    baseline_metrics.pop("case_metrics", None)
    write_jsonl_atomic(output / "case_metrics.jsonl", case_metrics)
    write_json_atomic(output / "metrics.json", metrics)
    write_json_atomic(output / "baseline_metrics.json", baseline_metrics)
    write_json_atomic(output / "baseline_comparison.json", comparison)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    receipt = build_run_receipt(
        "sealed-labelme-evaluation",
        digests={**config_digests, **artifact_hashes},
        counts={
            "cases": len(cases),
            "labeled_images": int(metrics["image_denominator"]),
            "shapes": int(metrics["shape_denominator"]),
        },
        metrics={
            "shape_recall_at_3": metrics["shape_recall_at_3"],
            "shape_recall_at_5": metrics["shape_recall_at_5"],
            "image_recall_at_3": metrics["image_recall_at_3"],
            "image_recall_at_5": metrics["image_recall_at_5"],
        },
        failures=0,
        temporary_files=len(list(output.rglob("*.tmp"))),
    )
    receipt["acceptance"] = comparison["acceptance"]
    write_json_atomic(output / "run_receipt.json", receipt)
    return receipt
