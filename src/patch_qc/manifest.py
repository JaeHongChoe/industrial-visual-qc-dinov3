from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
import re

from .capture import CaptureCondition, parse_standard_capture, rank_extra_captures


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_OBJECT_ID = re.compile(r"^(G\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: str
    condition: CaptureCondition

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "condition": asdict(self.condition)}


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    capture_id: str
    group_id: str
    product: str
    object_id: str
    date: str
    source_dir: str
    images: tuple[ImageRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "group_id": self.group_id,
            "product": self.product,
            "object_id": self.object_id,
            "date": self.date,
            "source_dir": self.source_dir,
            "source_label": "OK",
            "images": [image.to_dict() for image in self.images],
        }


def _parse_date_dir(name: str) -> date | None:
    if re.fullmatch(r"\d{8}", name):
        normalized = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    else:
        normalized = name.replace(".", "-").replace("_", "-")
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _canonical_object_id(name: str) -> str:
    match = _OBJECT_ID.match(name)
    return match.group(1).upper() if match else name


def scan_ok_tree(root: str | Path, start_date: str = "2026-05-01") -> list[CaptureRecord]:
    """Read an OK tree shaped as date/product/object and return capture records."""
    root_path = Path(root)
    lower_bound = date.fromisoformat(start_date)
    records: list[CaptureRecord] = []
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    for date_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        capture_date = _parse_date_dir(date_dir.name)
        if capture_date is None or capture_date < lower_bound:
            continue
        iso_date = capture_date.isoformat()
        for product_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
            for object_dir in sorted(path for path in product_dir.iterdir() if path.is_dir()):
                image_paths = sorted(
                    path
                    for path in object_dir.rglob("*")
                    if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                )
                if not image_paths:
                    continue
                standard: dict[Path, CaptureCondition] = {}
                extras: list[Path] = []
                for image_path in image_paths:
                    condition = parse_standard_capture(image_path.name)
                    if condition is None:
                        extras.append(image_path)
                    else:
                        standard[image_path] = condition
                conditions = standard | rank_extra_captures(extras)
                family_order = {"A": 0, "B": 1, "C": 2, "extra": 3}
                image_paths.sort(
                    key=lambda path: (
                        family_order.get(conditions[path].family, 9),
                        conditions[path].view_index or 0,
                        path.name.casefold(),
                    )
                )
                images = tuple(ImageRecord(path=str(path), condition=conditions[path]) for path in image_paths)
                object_id = _canonical_object_id(object_dir.name)
                records.append(
                    CaptureRecord(
                        capture_id=f"{iso_date}/{product_dir.name}/{object_dir.name}",
                        group_id=f"{product_dir.name}::{object_id}",
                        product=product_dir.name,
                        object_id=object_id,
                        date=iso_date,
                        source_dir=str(object_dir),
                        images=images,
                    )
                )
    return records


def load_manifest_records(path: str | Path, split: str | None = None) -> list[CaptureRecord]:
    """Load manifests produced by this project and enforce their OK-only contract."""
    records: list[CaptureRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_label") != "OK":
                raise ValueError(f"manifest line {line_number} is not an OK source record")
            if split is not None and row.get("split") != split:
                continue
            images = tuple(
                ImageRecord(
                    path=image["path"],
                    condition=CaptureCondition(**image["condition"]),
                )
                for image in row["images"]
            )
            records.append(
                CaptureRecord(
                    capture_id=row["capture_id"],
                    group_id=row["group_id"],
                    product=row["product"],
                    object_id=row["object_id"],
                    date=row["date"],
                    source_dir=row["source_dir"],
                    images=images,
                )
            )
    return records
