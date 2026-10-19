"""Dense normal-reference scoring helpers."""

from .core import (
    GridAccumulator,
    axis_positions,
    canonical_condition,
    dense_reference_distance,
    tail_mean_score,
    top_hotspots,
)

__all__ = [
    "GridAccumulator",
    "axis_positions",
    "canonical_condition",
    "dense_reference_distance",
    "tail_mean_score",
    "top_hotspots",
]
