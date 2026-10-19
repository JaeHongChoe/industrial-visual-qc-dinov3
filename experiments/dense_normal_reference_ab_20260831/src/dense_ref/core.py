"""Pure geometry and dense normal-reference scoring primitives."""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor


_CONDITION = re.compile(r"^([ABC])_(?:Photo-)?(L\d+-\d+)$")


def axis_positions(length: int, patch_size: int, stride: int) -> tuple[int, ...]:
    """Return native tile origins, including the final edge-aligned tile."""
    if length <= 0 or patch_size <= 0 or stride <= 0:
        raise ValueError("length, patch_size, and stride must be positive")
    if length <= patch_size:
        return (0,)
    last = length - patch_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


def canonical_condition(value: str) -> str | None:
    """Normalize supported A/B/C imaging-condition names for reference lookup."""
    match = _CONDITION.fullmatch(value.strip())
    if match is None:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def _as_channel_first(query: Tensor, references: Tensor) -> tuple[Tensor, Tensor, bool]:
    if query.ndim == 3 and references.ndim == 4:
        # [H, W, C] and [reference, H, W, C]
        return query.permute(2, 0, 1).unsqueeze(0), references.permute(0, 3, 1, 2).unsqueeze(1), True
    if query.ndim == 4 and references.ndim == 5:
        if query.shape[-1] > query.shape[1] and references.shape[1] == query.shape[0]:
            # [batch, H, W, C] and [reference, batch, H, W, C]
            return query.permute(0, 3, 1, 2), references.permute(0, 1, 4, 2, 3), False
        # [batch, channel, H, W] and [reference, batch, channel, H, W]
        return query, references, False
    raise ValueError(
        "expected [H,W,C]/[R,H,W,C] or [B,C,H,W]/[R,B,C,H,W] token tensors"
    )


def dense_reference_distance(query: Tensor, references: Tensor, radius: int = 1) -> Tensor:
    """Return ``1 - best cosine`` over each reference's local token neighborhood."""
    query_cf, references_cf, channel_last = _as_channel_first(query, references)
    query_normalized = functional.normalize(query_cf.float(), dim=1).to(torch.float16)
    references_normalized = functional.normalize(references_cf.float(), dim=2).to(torch.float16)
    return dense_reference_distance_normalized(
        query_normalized,
        references_normalized,
        radius=radius,
        channel_last=channel_last,
    )


def dense_reference_distance_normalized(
    query: Tensor,
    references: Tensor,
    radius: int = 1,
    *,
    channel_last: bool = False,
) -> Tensor:
    """Score already-normalized FP16 token tensors without a second normalization pass.

    Inputs are channel-first ``[B,C,H,W]`` and ``[R,B,C,H,W]``.  The optional
    ``channel_last`` flag restores a ``[B,H,W]`` result for callers that adapted
    channel-last encoder output before invoking this bounded fast path.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if query.ndim != 4 or references.ndim != 5:
        raise ValueError("normalized tokens must be [B,C,H,W] and [R,B,C,H,W]")
    if query.dtype != torch.float16 or references.dtype != torch.float16:
        raise ValueError("normalized dense scoring requires FP16 operands")
    reference_count, batch, channels, height, width = references.shape
    if reference_count == 0 or query.shape != (batch, channels, height, width):
        raise ValueError("query and references must have matching non-empty token geometry")

    # FP16 operands are deliberately multiplied directly; only the reduction is
    # promoted to FP32 so the stored/compared representation remains frozen.
    padded = functional.pad(
        references.reshape(reference_count * batch, channels, height, width),
        (radius, radius, radius, radius),
    ).reshape(reference_count, batch, channels, height + 2 * radius, width + 2 * radius)
    valid_positions = functional.pad(
        torch.ones((height, width), dtype=torch.bool, device=query.device),
        (radius, radius, radius, radius),
        value=False,
    )

    best_similarity: Tensor | None = None
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            candidate = padded[:, :, :, offset_y : offset_y + height, offset_x : offset_x + width]
            similarity = (candidate * query.unsqueeze(0)).sum(dim=2, dtype=torch.float32)
            valid = valid_positions[offset_y : offset_y + height, offset_x : offset_x + width]
            similarity = similarity.masked_fill(~valid, -torch.inf)
            local_best = similarity.max(dim=0).values
            best_similarity = local_best if best_similarity is None else torch.maximum(best_similarity, local_best)

    assert best_similarity is not None
    distance = 1.0 - best_similarity.clamp(max=1.0)
    return distance.squeeze(0) if channel_last else distance


class GridAccumulator:
    """Accumulate scalar evidence on a global pixel-center grid by finite mean."""

    def __init__(self, width: int, height: int, step: int = 8) -> None:
        if width <= 0 or height <= 0 or step <= 0:
            raise ValueError("width, height, and step must be positive")
        self.width = width
        self.height = height
        self.step = step
        shape = (math.ceil(height / step), math.ceil(width / step))
        self._sums = np.zeros(shape, dtype=np.float64)
        self._counts = np.zeros(shape, dtype=np.int64)

    def _add_cell(self, x: float, y: float, value: float) -> None:
        if not np.isfinite(value) or not (0 <= x < self.width and 0 <= y < self.height):
            return
        column = int(x // self.step)
        row = int(y // self.step)
        self._sums[row, column] += value
        self._counts[row, column] += 1

    def add_token_tile(
        self, x: int, y: int, values: np.ndarray | Tensor, token_step: int = 16
    ) -> None:
        """Add token scores at their token centers within a tile at ``(x, y)``."""
        if token_step <= 0:
            raise ValueError("token_step must be positive")
        scores = np.asarray(values.detach().cpu() if isinstance(values, Tensor) else values)
        if scores.ndim != 2:
            raise ValueError("token scores must be a two-dimensional array")
        offset = token_step / 2
        for row, column in np.ndindex(scores.shape):
            self._add_cell(x + offset + column * token_step, y + offset + row * token_step, float(scores[row, column]))

    def add_patch(self, x: int, y: int, width: int, height: int, value: float) -> None:
        """Add one scalar patch score to every grid center it covers."""
        if width <= 0 or height <= 0:
            raise ValueError("patch width and height must be positive")
        if not np.isfinite(value):
            return
        first_column = max(0, math.ceil((x - self.step / 2) / self.step))
        last_column = min(self._sums.shape[1] - 1, math.ceil((x + width - self.step / 2) / self.step) - 1)
        first_row = max(0, math.ceil((y - self.step / 2) / self.step))
        last_row = min(self._sums.shape[0] - 1, math.ceil((y + height - self.step / 2) / self.step) - 1)
        if first_column > last_column or first_row > last_row:
            return
        self._sums[first_row : last_row + 1, first_column : last_column + 1] += value
        self._counts[first_row : last_row + 1, first_column : last_column + 1] += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a NaN-filled scalar map and its valid-cell mask."""
        valid = self._counts > 0
        values = np.full(self._sums.shape, np.nan, dtype=np.float32)
        values[valid] = (self._sums[valid] / self._counts[valid]).astype(np.float32)
        return values, valid


def tail_mean_score(values: np.ndarray, fraction: float = 0.0001) -> float:
    """Mean the highest finite fraction of a map, retaining at least one cell."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    count = max(1, math.ceil(finite.size * fraction))
    return float(np.partition(finite, finite.size - count)[-count:].mean())


def top_hotspots(values: np.ndarray, count: int, threshold: float) -> list[tuple[int, int, float]]:
    """Return maxima of 4-connected finite components at or above ``threshold``."""
    if count < 0:
        raise ValueError("count must be non-negative")
    array = np.asarray(values, dtype=np.float64)
    eligible = np.isfinite(array) & (array >= threshold)
    visited = np.zeros_like(eligible)
    hotspots: list[tuple[int, int, float]] = []

    for start_row, start_column in np.argwhere(eligible):
        if visited[start_row, start_column]:
            continue
        component: list[tuple[int, int]] = []
        frontier = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        while frontier:
            row, column = frontier.pop()
            component.append((row, column))
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < array.shape[0]
                    and 0 <= next_column < array.shape[1]
                    and eligible[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    frontier.append((next_row, next_column))
        peak_value = max(array[row, column] for row, column in component)
        peak_row, peak_column = min(
            (row, column) for row, column in component if array[row, column] == peak_value
        )
        hotspots.append((peak_column, peak_row, float(peak_value)))

    hotspots.sort(key=lambda item: (-item[2], item[1], item[0]))
    return hotspots[:count]
