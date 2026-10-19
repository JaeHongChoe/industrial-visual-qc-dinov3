from __future__ import annotations

from collections import defaultdict
import hashlib

from .manifest import CaptureRecord


def _stable_group_key(seed: int, product: str, group_id: str) -> str:
    value = f"{seed}|{product}|{group_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _partition_counts(group_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count <= 0:
        return 0, 0, 0
    if group_count == 1:
        return 1, 0, 0
    if group_count == 2:
        return 1, 0, 1
    calibration = max(1, round(group_count * ratios[1]))
    holdout = max(1, round(group_count * ratios[2]))
    if calibration + holdout >= group_count:
        calibration = 1
        holdout = 1
    train = group_count - calibration - holdout
    return train, calibration, holdout


def split_by_group(
    records: list[CaptureRecord],
    seed: int = 17,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[CaptureRecord]]:
    """Split complete physical groups, stratified by product."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be three non-negative values summing to 1")

    product_groups: dict[str, dict[str, list[CaptureRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        product_groups[record.product][record.group_id].append(record)

    result: dict[str, list[CaptureRecord]] = {"train": [], "calibration": [], "holdout": []}
    for product, groups in sorted(product_groups.items()):
        group_ids = sorted(groups, key=lambda group_id: _stable_group_key(seed, product, group_id))
        train_count, calibration_count, _ = _partition_counts(len(group_ids), ratios)
        train_ids = set(group_ids[:train_count])
        calibration_ids = set(group_ids[train_count : train_count + calibration_count])
        for group_id in group_ids:
            split_name = (
                "train"
                if group_id in train_ids
                else "calibration"
                if group_id in calibration_ids
                else "holdout"
            )
            result[split_name].extend(groups[group_id])

    for split_records in result.values():
        split_records.sort(key=lambda record: record.capture_id)
    return result
