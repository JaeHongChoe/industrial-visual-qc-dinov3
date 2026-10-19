from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .inference import Box
from .refinement_contract import CandidateRow, canonical_digest
from .refinement_signals import SignalBundle, SignalSchema


_SCALAR_OFFSETS = {
    "probability": -18,
    "memory_distance": -17,
    "tta_probability_stability": -16,
    "tta_feature_stability": -15,
    "context_contrast": -14,
    "spatial_support": -13,
    "foreground_fraction": -12,
}


def _validate_bundle(bundle: SignalBundle) -> int:
    values = np.asarray(bundle.values, dtype=np.float32)
    if values.ndim != 1 or values.size != bundle.row.feature_dimension:
        raise ValueError("signal values do not match the declared feature dimension")
    if not np.isfinite(values).all():
        raise ValueError("signal values must be finite")
    if values.size < 21 or (values.size - 18) % 3:
        raise ValueError("signal values do not match the fixed schema")
    pooled_dimension = (values.size - 18) // 3
    expected = SignalSchema.default(pooled_dimension)
    if expected.digest != bundle.row.schema_digest:
        raise ValueError("signal schema digest mismatch")
    return pooled_dimension


def _scalar(bundle: SignalBundle, name: str) -> float:
    _validate_bundle(bundle)
    return float(bundle.values[_SCALAR_OFFSETS[name]])


@dataclass(frozen=True, slots=True)
class EmpiricalTailCalibrator:
    distributions: dict[str, dict[str, tuple[float, ...]]]
    minimum_bank_size: int = 5

    @classmethod
    def fit(
        cls,
        clean_signal_rows: Sequence[SignalBundle],
        minimum_bank_size: int = 5,
    ) -> "EmpiricalTailCalibrator":
        if minimum_bank_size <= 0 or not clean_signal_rows:
            raise ValueError("clean-tail calibration requires non-empty clean signals")
        names = ("probability", "memory_distance", "context_contrast")
        mutable: dict[str, dict[str, list[float]]] = {
            name: {} for name in names
        }
        schema_digests: set[str] = set()
        for bundle in clean_signal_rows:
            _validate_bundle(bundle)
            if bundle.row.source_label != "OK" or bundle.row.source_kind != "clean_anchor":
                raise ValueError("clean-tail calibration accepts only OK clean anchors")
            schema_digests.add(bundle.row.schema_digest)
            keys = (
                f"condition:{bundle.row.product}:{bundle.row.condition_id}",
                f"product:{bundle.row.product}",
                "global",
            )
            for name in names:
                value = _scalar(bundle, name)
                for key in keys:
                    mutable[name].setdefault(key, []).append(value)
        if len(schema_digests) != 1:
            raise ValueError("clean-tail signal schemas must match")
        distributions = {
            name: {key: tuple(sorted(values)) for key, values in banks.items()}
            for name, banks in mutable.items()
        }
        return cls(distributions=distributions, minimum_bank_size=minimum_bank_size)

    def tail(self, bundle: SignalBundle, name: str) -> float:
        if name not in self.distributions:
            raise KeyError(name)
        candidates = (
            f"condition:{bundle.row.product}:{bundle.row.condition_id}",
            f"product:{bundle.row.product}",
            "global",
        )
        bank: tuple[float, ...] | None = None
        for key in candidates:
            values = self.distributions[name].get(key, ())
            if len(values) >= self.minimum_bank_size or key == "global":
                bank = values
                break
        if not bank:
            raise RuntimeError(f"no clean-tail bank is available for {name}")
        value = _scalar(bundle, name)
        return float(np.searchsorted(np.asarray(bank), value, side="right") / len(bank))


def deterministic_relevance(row: SignalBundle, calibrator: EmpiricalTailCalibrator) -> float:
    score = (
        0.40 * calibrator.tail(row, "probability")
        + 0.25 * calibrator.tail(row, "memory_distance")
        + 0.15 * calibrator.tail(row, "context_contrast")
        + 0.15 * _scalar(row, "tta_feature_stability")
        + 0.05 * min(_scalar(row, "spatial_support") / 3.0, 1.0)
    )
    if row.row.roi_relation == "outside":
        score -= 0.05
    return float(np.clip(score, 0.0, 1.0))


class WeakRanker(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 128) -> None:
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("ranker dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass(frozen=True, slots=True)
class WeakRankerReceipt:
    model: WeakRanker
    state_digest: str
    schema_digest: str
    data_digest: str
    config_digest: str
    epochs: int
    best_calibration_loss: float


def _ranker_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _training_matrix(
    rows: Sequence[SignalBundle],
    *,
    expected_label: str | None = None,
    expected_kind: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[float] = []
    for bundle in rows:
        _validate_bundle(bundle)
        if bundle.row.source_kind == "unlabeled_candidate":
            raise ValueError("unlabeled OK candidates may not enter weak-ranker training")
        if expected_label is not None and (
            bundle.row.source_label != expected_label or bundle.row.source_kind != expected_kind
        ):
            raise ValueError(
                f"unexpected ranker training source {bundle.row.source_label}/{bundle.row.source_kind}"
            )
        if bundle.row.source_label == "OK" and bundle.row.source_kind == "clean_anchor":
            label = 0.0
        elif bundle.row.source_label == "SYNTHETIC" and bundle.row.source_kind == "synthetic_anchor":
            label = 1.0
        else:
            raise ValueError("ranker accepts only clean or synthetic anchors")
        values.append(np.asarray(bundle.values, dtype=np.float32))
        labels.append(label)
    if not values:
        raise ValueError("ranker source rows must not be empty")
    matrix = np.stack(values)
    if len({row.shape for row in values}) != 1:
        raise ValueError("ranker signal dimensions must match")
    return matrix, np.asarray(labels, dtype=np.float32)


def fit_weak_ranker(
    clean_rows: Sequence[SignalBundle],
    synthetic_rows: Sequence[SignalBundle],
    calibration_rows: Sequence[SignalBundle],
    seed: int,
) -> WeakRankerReceipt:
    clean_values, clean_labels = _training_matrix(
        clean_rows, expected_label="OK", expected_kind="clean_anchor"
    )
    synthetic_values, synthetic_labels = _training_matrix(
        synthetic_rows, expected_label="SYNTHETIC", expected_kind="synthetic_anchor"
    )
    calibration_values, calibration_labels = _training_matrix(calibration_rows)
    if clean_values.shape[1] != synthetic_values.shape[1] or clean_values.shape[1] != calibration_values.shape[1]:
        raise ValueError("ranker signal dimensions must match")
    schema_digests = {
        bundle.row.schema_digest for bundle in [*clean_rows, *synthetic_rows, *calibration_rows]
    }
    if len(schema_digests) != 1:
        raise ValueError("ranker signal schemas must match")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = WeakRanker(clean_values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    training_values = torch.from_numpy(np.concatenate((clean_values, synthetic_values)))
    training_labels = torch.from_numpy(np.concatenate((clean_labels, synthetic_labels)))
    calibration_tensor = torch.from_numpy(calibration_values)
    calibration_targets = torch.from_numpy(calibration_labels)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    epochs = 0
    for epoch in range(30):
        model.train()
        permutation = torch.randperm(len(training_values), generator=generator)
        for start in range(0, len(permutation), 512):
            indices = permutation[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(training_values[indices]), training_labels[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_loss = float(loss_fn(model(calibration_tensor), calibration_targets))
        epochs = epoch + 1
        if calibration_loss < best_loss - 1e-9:
            best_loss = calibration_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is None:
        raise RuntimeError("weak-ranker training did not produce a finite state")
    model.load_state_dict(best_state)
    model.eval()
    data_digest = hashlib.sha256(
        np.ascontiguousarray(
            np.concatenate((clean_values, synthetic_values, calibration_values)), dtype=np.float32
        ).tobytes()
    ).hexdigest()
    config_digest = canonical_digest(
        {
            "architecture": [clean_values.shape[1], 128, 1],
            "batch_size": 512,
            "epochs": 30,
            "learning_rate": 1e-3,
            "patience": 5,
            "seed": seed,
            "weight_decay": 1e-4,
        }
    )
    return WeakRankerReceipt(
        model=model,
        state_digest=_ranker_state_digest(model),
        schema_digest=next(iter(schema_digests)),
        data_digest=data_digest,
        config_digest=config_digest,
        epochs=epochs,
        best_calibration_loss=best_loss,
    )


def select_refiner(
    fusion_metrics: Mapping[str, float], weak_metrics: Mapping[str, float]
) -> str:
    weak_wins = (
        weak_metrics["synthetic_local_recall_at_5"]
        >= fusion_metrics["synthetic_local_recall_at_5"] + 0.01
        and weak_metrics["clean_mean_candidates_per_image"]
        <= fusion_metrics["clean_mean_candidates_per_image"] + 0.05
        and weak_metrics["tta_top5_agreement"] >= fusion_metrics["tta_top5_agreement"]
    )
    return "weak_ranker" if weak_wins else "deterministic_fusion"


@dataclass(frozen=True, slots=True)
class RankedPatch:
    patch_id: str
    box: Box
    relevance: float
    probability: float
    feature: np.ndarray
    roi_relation: str


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    component_id: str
    patch_ids: tuple[str, ...]
    box: Box
    score: float
    probability: float
    support: int
    roi_relation: str
    review_state: str = "REVIEW"


@dataclass(frozen=True, slots=True)
class _Component:
    review: ReviewCandidate
    feature: np.ndarray


def _overlap(first: Box, second: Box) -> bool:
    return min(first.x1, second.x1) > max(first.x0, second.x0) and min(
        first.y1, second.y1
    ) > max(first.y0, second.y0)


def _adjacent(first: RankedPatch, second: RankedPatch, stride: int) -> bool:
    return (
        abs(first.box.x0 - second.box.x0) <= stride
        and abs(first.box.y0 - second.box.y0) <= stride
        and _overlap(first.box, second.box)
    )


def _components(rows: Sequence[RankedPatch], stride: int) -> list[_Component]:
    remaining = set(range(len(rows)))
    output: list[_Component] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        members = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            joined = [index for index in sorted(remaining) if _adjacent(rows[current], rows[index], stride)]
            for index in joined:
                remaining.remove(index)
                frontier.append(index)
                members.append(index)
        member_rows = [rows[index] for index in members]
        best = min(member_rows, key=lambda row: (-row.relevance, row.patch_id))
        patch_ids = tuple(sorted(row.patch_id for row in member_rows))
        support = len(member_rows)
        score = float(np.clip(best.relevance + min(0.01 * (support - 1), 0.03), 0.0, 1.0))
        feature = np.stack([np.asarray(row.feature, dtype=np.float32) for row in member_rows]).mean(axis=0)
        if not np.isfinite(feature).all():
            raise ValueError("ranked patch features must be finite")
        feature /= max(float(np.linalg.norm(feature)), 1e-12)
        component_id = canonical_digest({"patch_ids": patch_ids})[:16]
        output.append(
            _Component(
                ReviewCandidate(
                    component_id=component_id,
                    patch_ids=patch_ids,
                    box=Box(
                        min(row.box.x0 for row in member_rows),
                        min(row.box.y0 for row in member_rows),
                        max(row.box.x1 for row in member_rows),
                        max(row.box.y1 for row in member_rows),
                    ),
                    score=score,
                    probability=max(row.probability for row in member_rows),
                    support=support,
                    roi_relation=best.roi_relation,
                ),
                feature,
            )
        )
    return sorted(output, key=lambda item: (-item.review.score, item.review.component_id))


def _centers_close(first: Box, second: Box, stride: int) -> bool:
    first_center = ((first.x0 + first.x1) / 2.0, (first.y0 + first.y1) / 2.0)
    second_center = ((second.x0 + second.x1) / 2.0, (second.y0 + second.y1) / 2.0)
    return abs(first_center[0] - second_center[0]) <= stride and abs(
        first_center[1] - second_center[1]
    ) <= stride


def merge_and_rank_candidates(
    rows: Sequence[RankedPatch],
    top_k: int,
    stride: int = 128,
    diversity_cosine: float = 0.98,
) -> list[ReviewCandidate]:
    if top_k <= 0 or stride <= 0:
        raise ValueError("top_k and stride must be positive")
    if not -1.0 <= diversity_cosine <= 1.0:
        raise ValueError("diversity_cosine must be between -1 and 1")
    components = _components(rows, stride)
    selected: list[_Component] = []
    deferred: list[_Component] = []
    for component in components:
        duplicate = any(
            float(np.dot(component.feature, prior.feature)) >= diversity_cosine
            and _centers_close(component.review.box, prior.review.box, stride)
            for prior in selected
        )
        (deferred if duplicate else selected).append(component)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        selected.extend(deferred[: top_k - len(selected)])
    if len(selected) < top_k:
        chosen = {item.review.component_id for item in selected}
        selected.extend(
            item for item in components if item.review.component_id not in chosen
        )
    return [item.review for item in selected[:top_k]]


@dataclass(frozen=True, slots=True)
class RankingReceipt:
    candidates: tuple[ReviewCandidate, ...]
    fallback_level: str
    refiner: str


def rank_with_fallback(
    candidates: Sequence[CandidateRow],
    signal_bundles: Sequence[SignalBundle] | None,
    refiner: EmpiricalTailCalibrator | WeakRankerReceipt | None,
    top_k: int,
) -> RankingReceipt:
    if signal_bundles is None or refiner is None:
        ranked_rows = [
            RankedPatch(
                patch_id=row.patch_id,
                box=Box(*row.box),
                relevance=row.probability,
                probability=row.probability,
                feature=np.asarray([row.probability, row.box[0], row.box[1]], dtype=np.float32),
                roi_relation="unavailable",
            )
            for row in candidates
        ]
        ranked = merge_and_rank_candidates(ranked_rows, top_k)
        return RankingReceipt(tuple(ranked), "epoch3_probability", "epoch3_probability")

    if len(candidates) != len(signal_bundles):
        raise ValueError("candidate and signal counts must match")
    bundles_by_id = {bundle.row.patch_id: bundle for bundle in signal_bundles}
    if len(bundles_by_id) != len(signal_bundles):
        raise ValueError("signal patch IDs must be unique")
    ranked_rows: list[RankedPatch] = []
    for candidate in candidates:
        bundle = bundles_by_id.get(candidate.patch_id)
        if bundle is None:
            raise ValueError(f"signal is missing for candidate {candidate.patch_id}")
        pooled_dimension = _validate_bundle(bundle)
        if isinstance(refiner, EmpiricalTailCalibrator):
            relevance = deterministic_relevance(bundle, refiner)
            refiner_name = "deterministic_fusion"
        else:
            if bundle.row.schema_digest != refiner.schema_digest:
                raise ValueError("weak-ranker schema digest mismatch")
            with torch.no_grad():
                relevance = float(
                    torch.sigmoid(
                        refiner.model(torch.from_numpy(bundle.values.astype(np.float32))[None, :])
                    )[0]
                )
            refiner_name = "weak_ranker"
        ranked_rows.append(
            RankedPatch(
                patch_id=candidate.patch_id,
                box=Box(*candidate.box),
                relevance=relevance,
                probability=candidate.probability,
                feature=bundle.values[:pooled_dimension],
                roi_relation=bundle.row.roi_relation,
            )
        )
    ranked = merge_and_rank_candidates(ranked_rows, top_k)
    return RankingReceipt(tuple(ranked), "none", refiner_name)
