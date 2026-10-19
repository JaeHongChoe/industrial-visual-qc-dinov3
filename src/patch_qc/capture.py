from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_STANDARD_PATTERN = re.compile(
    r"(?:^|[_-])(?P<family>[abc])_Photo-L1-(?P<index>\d{1,2})(?:_\d+)?(?:\.[^.]+)?$",
    re.IGNORECASE,
)
_NATURAL_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True, slots=True)
class CaptureCondition:
    family: str
    view_index: int | None
    condition_id: str
    is_standard: bool


def parse_standard_capture(name: str | Path) -> CaptureCondition | None:
    """Parse an A/B/C L1 capture without guessing unresolved extra semantics."""
    filename = Path(name).name
    match = _STANDARD_PATTERN.search(filename)
    if match is None:
        return None
    family = match.group("family").upper()
    view_index = int(match.group("index"))
    return CaptureCondition(
        family=family,
        view_index=view_index,
        condition_id=f"{family}_L1-{view_index:02d}",
        is_standard=True,
    )


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = _NATURAL_NUMBER.split(path.name.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def rank_extra_captures(paths: list[Path]) -> dict[Path, CaptureCondition]:
    """Assign deterministic ranks while leaving physical meanings configurable."""
    ranked: dict[Path, CaptureCondition] = {}
    for index, path in enumerate(sorted(paths, key=_natural_key), start=1):
        ranked[path] = CaptureCondition(
            family="extra",
            view_index=index,
            condition_id=f"extra_{index}",
            is_standard=False,
        )
    return ranked
