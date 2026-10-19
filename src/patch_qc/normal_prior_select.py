from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .normal_prior_postprocess import register_reference_thumbnail


def _thumbnail(path: str | Path, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("thumbnail_width must be positive")
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.draft("L", (width, width))
        gray = image.convert("L")
        height = max(1, int(round(gray.height * min(width, gray.width) / gray.width)))
        gray = gray.resize((min(width, gray.width), height), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32) / 255.0


def select_visual_references(
    plan: Mapping[str, Any],
    *,
    required_references: int = 3,
    thumbnail_width: int = 1024,
) -> dict[str, Any]:
    """Choose the best registerable exact-condition OK references per query."""
    if plan.get("schema_version") != "normal_prior_reference_plan_v1":
        raise ValueError("unsupported reference plan schema")
    if required_references <= 0:
        raise ValueError("required_references must be positive")
    source_queries = plan.get("queries")
    source_references = plan.get("references")
    if not isinstance(source_queries, list) or not isinstance(source_references, list):
        raise ValueError("reference plan must contain query and reference lists")
    reference_by_id: dict[str, dict[str, Any]] = {}
    for value in source_references:
        if not isinstance(value, dict):
            raise ValueError("reference rows must be objects")
        reference_id = value.get("reference_id")
        if not isinstance(reference_id, str) or reference_id in reference_by_id:
            raise ValueError("reference IDs must be unique strings")
        reference_by_id[reference_id] = dict(value)

    thumbnail_cache: dict[str, np.ndarray] = {}

    def load(path: str) -> np.ndarray:
        if path not in thumbnail_cache:
            thumbnail_cache[path] = _thumbnail(path, thumbnail_width)
        return thumbnail_cache[path]

    selected_queries: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    ready_count = 0
    for source_query in source_queries:
        if not isinstance(source_query, dict):
            raise ValueError("query rows must be objects")
        query = dict(source_query)
        selected: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        if query.get("status") == "ready_exact_product_condition":
            image_path = query.get("image_path")
            candidates = query.get("candidate_reference_ids")
            if not isinstance(image_path, str) or not isinstance(candidates, list):
                raise ValueError("ready query is missing its image path or candidates")
            query_thumbnail = load(image_path)
            ranked: list[tuple[float, str]] = []
            for reference_id in candidates:
                if not isinstance(reference_id, str) or reference_id not in reference_by_id:
                    raise ValueError("query names an unknown reference")
                reference = reference_by_id[reference_id]
                thumbnail = load(str(reference["image_path"]))
                if thumbnail.shape != query_thumbnail.shape:
                    diagnostics.append(
                        {"reference_id": reference_id, "accepted": False, "reason": "shape_mismatch"}
                    )
                    continue
                registration = register_reference_thumbnail(
                    thumbnail, query_thumbnail, allow_feature_fallback=True
                )
                diagnostics.append(
                    {
                        "reference_id": reference_id,
                        "accepted": registration.accepted,
                        "reason": registration.reason,
                        "response": registration.response,
                        "shift_xy": list(registration.shift_xy),
                        "overlap_fraction": registration.overlap_fraction,
                        "method": registration.method,
                        "feature_inliers": registration.feature_inliers,
                        "feature_inlier_ratio": registration.feature_inlier_ratio,
                        "scale": registration.scale,
                    }
                )
                if registration.accepted:
                    ranked.append((registration.response, reference_id))
            ranked.sort(key=lambda value: (-value[0], value[1]))
            selected = [reference_id for _, reference_id in ranked[:required_references]]
            if len(selected) == required_references:
                query["status"] = f"ready_visual_nearest_{required_references}"
                ready_count += 1
                selected_ids.update(selected)
            else:
                query["status"] = (
                    f"fail_open_registration_{len(selected)}_of_{required_references}"
                )
                selected = []
        query["selected_reference_ids"] = selected
        query["selection_diagnostics"] = diagnostics
        selected_queries.append(query)

    references = [
        reference_by_id[reference_id]
        for reference_id in sorted(
            selected_ids,
            key=lambda value: (
                str(reference_by_id[value].get("product", "")),
                str(reference_by_id[value].get("condition_id", "")),
                value,
            ),
        )
    ]
    total_queries = len(selected_queries)
    return {
        "schema_version": "normal_prior_selected_v1",
        "required_references": required_references,
        "queries": selected_queries,
        "references": references,
        "source_plan_schema_version": plan.get("schema_version"),
        "audit": {
            "total_queries": total_queries,
            "ready_queries": ready_count,
            "fail_open_queries": total_queries - ready_count,
            "selected_unique_references": len(references),
        },
    }
