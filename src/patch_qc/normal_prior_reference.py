from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch

from .actual_multitask_inference import predict_multitask_image


@dataclass(frozen=True, slots=True)
class ReferenceImage:
    identifier: str
    image_path: Path
    image_sha256: str
    product: str
    condition_id: str
    capture_id: str
    group_id: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_size(height: int, width: int, map_width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0 or map_width <= 0:
        raise ValueError("image dimensions and map_width must be positive")
    compact_width = min(width, map_width)
    compact_height = max(1, int(round(height * compact_width / width)))
    return compact_height, compact_width


def evaluate_reference_rows(
    rows: Sequence[ReferenceImage],
    model: torch.nn.Module,
    *,
    output_dir: str | Path,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device | str,
    map_width: int = 1024,
    use_amp: bool = True,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Infer compact raw maps for a pre-sealed, leakage-safe OK reference bank."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    if not rows:
        raise ValueError("reference rows must not be empty")
    if len({row.identifier for row in rows}) != len(rows):
        raise ValueError("reference identifiers must be unique")

    sealed_bytes: dict[str, bytes] = {}
    for row in rows:
        raw = row.image_path.read_bytes()
        if _sha256_bytes(raw) != row.image_sha256:
            raise RuntimeError(f"source changed after reference planning: {row.image_path}")
        sealed_bytes[row.identifier] = raw

    raw_dir = destination / "raw_maps"
    preview_dir = destination / "previews"
    raw_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}
    for row in rows:
        with Image.open(io.BytesIO(sealed_bytes[row.identifier])) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prediction = predict_multitask_image(
            model,
            rgb,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            patch_gate_threshold=0.5,
            use_amp=use_amp,
        )
        compact_height, compact_width = _compact_size(*rgb.shape[:2], map_width)
        compact_raw = cv2.resize(
            prediction.raw_score_map,
            (compact_width, compact_height),
            interpolation=cv2.INTER_AREA,
        )
        compact_rgb = cv2.resize(
            rgb,
            (compact_width, compact_height),
            interpolation=cv2.INTER_AREA,
        )
        raw_path = raw_dir / f"{row.identifier}.png"
        preview_path = preview_dir / f"{row.identifier}.jpg"
        Image.fromarray(np.rint(np.clip(compact_raw, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
            raw_path, compress_level=4
        )
        Image.fromarray(compact_rgb).save(preview_path, quality=92, subsampling=0)
        raw_relative = raw_path.relative_to(destination).as_posix()
        preview_relative = preview_path.relative_to(destination).as_posix()
        artifacts[preview_relative] = _sha256_file(preview_path)
        artifacts[raw_relative] = _sha256_file(raw_path)
        records.append(
            {
                "id": row.identifier,
                "product": row.product,
                "condition_id": row.condition_id,
                "capture_id": row.capture_id,
                "group_id": row.group_id,
                "image_path": str(row.image_path),
                "image_sha256": row.image_sha256,
                "native_shape": [int(value) for value in rgb.shape],
                "map_shape": [compact_height, compact_width],
                "patch_count": prediction.patch_count,
                "baseline_patches_above_0_7": int(
                    (prediction.baseline_probabilities >= 0.7).sum()
                ),
                "outputs": {"raw_map": raw_relative, "preview": preview_relative},
            }
        )
    return records, artifacts

