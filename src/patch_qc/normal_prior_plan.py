from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .capture import parse_standard_capture


STRICT_EVALUATION_GROUP_IDS = frozenset(
    {
        "PRODUCT_F::G019",
        "PRODUCT_D::G285",
        "PRODUCT_E::G526",
        "PRODUCT_C::G395",
        "PRODUCT_C::G404",
        "PRODUCT_A::G2867",
        "PRODUCT_A::G2899",
        "PRODUCT_A::G3753",
        "PRODUCT_A::G3801",
        "PRODUCT_A::G3803",
        "PRODUCT_A::G3807",
        "PRODUCT_A::G3816",
        "PRODUCT_A::G3834",
        "PRODUCT_A::G3871",
        "PRODUCT_A::G3898",
        "PRODUCT_B::PRODUCT_G-G397",
    }
)

_STANDARD_CONDITION = re.compile(r"^[ABC]_L1-\d{2}$")
_ELIGIBLE_SPLITS = frozenset({"train", "calibration"})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standard_condition(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if _STANDARD_CONDITION.fullmatch(normalized) else None


def _product_from_ok_path(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parts = PurePosixPath(value).parts
    for marker in ("OK", "ok"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) > index + 2:
                product = parts[index + 2].strip()
                return product or None
    return None


def _evaluation_identity(row: dict[str, Any]) -> tuple[str | None, str | None]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    product_value = metadata.get("product")
    product = product_value.strip() if isinstance(product_value, str) else None
    product = product or None
    condition_id = _standard_condition(metadata.get("condition_id"))

    relative_value = metadata.get("relative_source_path")
    relative_path = relative_value if isinstance(relative_value, str) else None
    if relative_path:
        parts = PurePosixPath(relative_path).parts
        if product is None and len(parts) >= 2:
            candidate_product = parts[0].strip()
            product = candidate_product or None
        if condition_id is None:
            condition = parse_standard_capture(PurePosixPath(relative_path).name)
            condition_id = condition.condition_id if condition is not None else None

    image_value = row.get("image_path")
    image_path = image_value if isinstance(image_value, str) else None
    if image_path:
        if product is None:
            product = _product_from_ok_path(image_path)
        if condition_id is None:
            condition = parse_standard_capture(PurePosixPath(image_path).name)
            condition_id = condition.condition_id if condition is not None else None

    return product, condition_id


def _reference_id(path: str) -> str:
    return f"ref_{hashlib.sha256(path.encode('utf-8')).hexdigest()[:20]}"


def build_reference_plan(
    ok_manifest_path: str | Path,
    predictions_path: str | Path,
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Build a sealed, exact-product/condition OK-reference plan.

    Every known evaluation unit is excluded as a whole group across all splits.
    Only train/calibration standard captures may become references. A query that
    cannot be parsed or matched is recorded as fail-open so downstream code can
    preserve its unmodified raw score map.
    """
    if not 1 <= max_candidates <= 64:
        raise ValueError("max_candidates must be between 1 and 64")

    manifest_rows = _read_jsonl(ok_manifest_path)
    seen_group_ids: set[str] = set()
    safe_group_ids: set[str] = set()
    eligible_group_ids: set[str] = set()
    safe_images = 0
    eligible_images = 0
    blocked_captures = 0
    blocked_images = 0
    holdout_captures = 0
    holdout_images = 0
    by_exact_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for line_number, row in enumerate(manifest_rows, start=1):
        if row.get("source_label") != "OK":
            raise ValueError(f"manifest line {line_number} is not an OK source record")
        group_id = row.get("group_id")
        product = row.get("product")
        images = row.get("images")
        split = row.get("split")
        if not isinstance(group_id, str) or not isinstance(product, str) or not isinstance(images, list):
            raise ValueError(f"manifest line {line_number} is missing group/product/images")

        seen_group_ids.add(group_id)
        if group_id in STRICT_EVALUATION_GROUP_IDS:
            blocked_captures += 1
            blocked_images += len(images)
            continue

        safe_group_ids.add(group_id)
        safe_images += len(images)
        if split not in _ELIGIBLE_SPLITS:
            if split == "holdout":
                holdout_captures += 1
                holdout_images += len(images)
            continue

        eligible_group_ids.add(group_id)
        eligible_images += len(images)
        for image in images:
            if not isinstance(image, dict):
                continue
            condition = image.get("condition")
            if not isinstance(condition, dict) or condition.get("is_standard") is not True:
                continue
            condition_id = _standard_condition(condition.get("condition_id"))
            image_path = image.get("path")
            if condition_id is None or not isinstance(image_path, str):
                continue
            candidate = {
                "capture_id": row.get("capture_id"),
                "condition_id": condition_id,
                "date": row.get("date"),
                "group_id": group_id,
                "image_path": image_path,
                "product": product,
                "split": split,
            }
            group_candidates = by_exact_key.setdefault((product, condition_id), {})
            previous = group_candidates.get(group_id)
            previous_rank = (
                str(previous.get("date", "")),
                str(previous.get("capture_id", "")),
                str(previous.get("image_path", "")),
            ) if previous is not None else None
            candidate_rank = (
                str(candidate.get("date", "")),
                str(candidate.get("capture_id", "")),
                image_path,
            )
            if previous_rank is None or candidate_rank > previous_rank:
                group_candidates[group_id] = candidate

    missing_blocked = STRICT_EVALUATION_GROUP_IDS - seen_group_ids
    if missing_blocked:
        raise ValueError(
            "manifest is missing strict evaluation group_ids: "
            + ", ".join(sorted(missing_blocked))
        )

    selected_by_path: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    for line_number, row in enumerate(_read_jsonl(predictions_path), start=1):
        evaluation_id = row.get("id")
        image_path = row.get("image_path")
        if not isinstance(evaluation_id, str) or not isinstance(image_path, str):
            raise ValueError(f"predictions line {line_number} is missing id/image_path")
        product, condition_id = _evaluation_identity(row)
        if product is None or condition_id is None:
            status = "fail_open_nonstandard"
            candidates: list[dict[str, Any]] = []
        else:
            candidates = sorted(
                by_exact_key.get((product, condition_id), {}).values(),
                key=lambda value: (
                    str(value["group_id"]),
                    str(value["capture_id"]),
                    str(value["image_path"]),
                ),
            )[:max_candidates]
            status = (
                "ready_exact_product_condition"
                if candidates
                else "fail_open_no_exact_references"
            )

        candidate_reference_ids: list[str] = []
        for candidate in candidates:
            candidate_path = str(candidate["image_path"])
            reference_id = _reference_id(candidate_path)
            candidate_reference_ids.append(reference_id)
            if candidate_path not in selected_by_path:
                selected_by_path[candidate_path] = {
                    **candidate,
                    "image_sha256": _sha256(candidate_path),
                    "reference_id": reference_id,
                }
        queries.append(
            {
                "candidate_reference_ids": candidate_reference_ids,
                "condition_id": condition_id,
                "id": evaluation_id,
                "image_path": image_path,
                "image_sha256": row.get("image_sha256"),
                "product": product,
                "status": status,
                "visual_nearest_limit": 4,
            }
        )

    references = sorted(
        selected_by_path.values(),
        key=lambda value: (
            str(value["product"]),
            str(value["condition_id"]),
            str(value["group_id"]),
            str(value["image_path"]),
        ),
    )
    return {
        "audit": {
            "blocked_capture_rows": blocked_captures,
            "blocked_images": blocked_images,
            "eligible_capture_rows": sum(
                1
                for row in manifest_rows
                if row.get("group_id") not in STRICT_EVALUATION_GROUP_IDS
                and row.get("split") in _ELIGIBLE_SPLITS
            ),
            "eligible_images": eligible_images,
            "eligible_unique_group_ids": len(eligible_group_ids),
            "holdout_capture_rows_excluded": holdout_captures,
            "holdout_images_excluded": holdout_images,
            "manifest_capture_rows": len(manifest_rows),
            "manifest_unique_group_ids": len(seen_group_ids),
            "safe_all_split_capture_rows": len(manifest_rows) - blocked_captures,
            "safe_all_split_images": safe_images,
            "safe_all_split_unique_group_ids": len(safe_group_ids),
        },
        "blocklist": sorted(STRICT_EVALUATION_GROUP_IDS),
        "max_candidates": max_candidates,
        "queries": queries,
        "references": references,
        "schema_version": "normal_prior_reference_plan_v1",
    }


def write_reference_plan(plan: dict[str, Any], destination: str | Path) -> None:
    """Atomically write a plan in canonical, human-auditable JSON form."""
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
