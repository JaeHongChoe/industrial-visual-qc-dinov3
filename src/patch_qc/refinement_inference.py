from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .inference import tile_boxes
from .refinement_contract import CandidateRow, PatchKey
from .refinement_encoding import CandidatePolicy, select_candidate_indices
from .refinement_memory import NormalMemoryIndex
from .refinement_ranking import (
    EmpiricalTailCalibrator,
    RankingReceipt,
    ReviewCandidate,
    WeakRankerReceipt,
    rank_with_fallback,
)
from .refinement_signals import (
    SignalBundle,
    augment_candidate_evidence,
    build_signal_bundle,
)
from .roi import estimate_placement_mask


@dataclass(frozen=True, slots=True)
class InferenceCase:
    case_id: str
    image_path: str
    source_sha256: str
    product: str
    group_id: str
    condition_id: str


@dataclass(frozen=True, slots=True)
class CaseRanking:
    case_id: str
    refined: tuple[ReviewCandidate, ...]
    baseline: tuple[ReviewCandidate, ...]
    candidate_count: int
    total_patches: int
    refined_refiner: str
    baseline_refiner: str


def select_native_candidates(
    rgb: np.ndarray,
    case: InferenceCase,
    encoder: Any,
    policy: CandidatePolicy,
    *,
    batch_size: int,
    patch_size: int = 256,
    stride: int = 128,
) -> tuple[list[CandidateRow], int]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    boxes = tile_boxes(rgb.shape[1], rgb.shape[0], patch_size, stride)
    logits: list[np.ndarray] = []
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start : start + batch_size]
        patches = np.stack(
            [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in batch_boxes]
        )
        batch_logits, _ = encoder.encode(patches)
        values = np.asarray(batch_logits, dtype=np.float32)
        if values.shape != (len(batch_boxes),) or not np.isfinite(values).all():
            raise ValueError("encoder logits must be finite and match native patches")
        logits.append(values)
    all_logits = np.concatenate(logits)
    probabilities = (
        1.0 / (1.0 + np.exp(-np.clip(all_logits, -80.0, 80.0)))
    ).astype(np.float32)
    selected_indices = select_candidate_indices(probabilities, policy)
    rows: list[CandidateRow] = []
    for index in selected_indices.tolist():
        box = boxes[index]
        box_values = (box.x0, box.y0, box.x1, box.y1)
        patch = rgb[box.y0 : box.y1, box.x0 : box.x1]
        patch_id = PatchKey(case.source_sha256, box_values).patch_id
        rows.append(
            CandidateRow(
                patch_id=patch_id,
                source_sha256=case.source_sha256,
                source_path=case.image_path,
                product=case.product,
                group_id=case.group_id,
                condition_id=case.condition_id,
                split="sealed_evaluation",
                box=box_values,
                logit=float(all_logits[index]),
                probability=float(probabilities[index]),
                foreground_fraction=float((patch.mean(axis=2) > 8).mean()),
                feature_key=patch_id,
                source_label="EVALUATION",
            )
        )
    return rows, len(boxes)


def rank_image_case(
    rgb: np.ndarray,
    case: InferenceCase,
    encoder: Any,
    policy: CandidatePolicy,
    normal_memory: NormalMemoryIndex,
    refiner: EmpiricalTailCalibrator | WeakRankerReceipt,
    *,
    batch_size: int,
    query_k: int,
    top_k: int,
    stride: int,
) -> CaseRanking:
    candidates, total_patches = select_native_candidates(
        rgb,
        case,
        encoder,
        policy,
        batch_size=batch_size,
        patch_size=256,
        stride=stride,
    )
    evidence_rows, feature_blocks = augment_candidate_evidence(
        candidates,
        image_loader=lambda _path: rgb,
        encoder=encoder,
        roi_provider=lambda _path, image: estimate_placement_mask(image, margin=8),
        batch_size=batch_size,
    )
    evidence_by_id = {row.patch_id: row for row in evidence_rows}
    block_by_id = {
        row.patch_id: block
        for row, block in zip(evidence_rows, feature_blocks, strict=True)
    }
    bundles: list[SignalBundle] = []
    for candidate in candidates:
        block = block_by_id[candidate.patch_id]
        evidence = evidence_by_id[candidate.patch_id]
        memory_query = normal_memory.query(
            block[0], candidate.product, candidate.condition_id, k=query_k
        )
        bundles.append(
            build_signal_bundle(candidate, block[0], [block[2]], memory_query, evidence)
        )
    refined: RankingReceipt = rank_with_fallback(candidates, bundles, refiner, top_k)
    baseline: RankingReceipt = rank_with_fallback(candidates, None, None, top_k)
    return CaseRanking(
        case_id=case.case_id,
        refined=refined.candidates,
        baseline=baseline.candidates,
        candidate_count=len(candidates),
        total_patches=total_patches,
        refined_refiner=refined.refiner,
        baseline_refiner=baseline.refiner,
    )


def review_candidates_to_json(
    rows: Sequence[ReviewCandidate],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if row.review_state != "REVIEW":
            raise ValueError("ranked outputs must remain REVIEW-only")
        output.append(
            {
                "component_id": row.component_id,
                "patch_ids": list(row.patch_ids),
                "box": {
                    "x0": row.box.x0,
                    "y0": row.box.y0,
                    "x1": row.box.x1,
                    "y1": row.box.y1,
                },
                "score": row.score,
                "probability": row.probability,
                "support": row.support,
                "roi_relation": row.roi_relation,
                "review_state": row.review_state,
            }
        )
    return output
