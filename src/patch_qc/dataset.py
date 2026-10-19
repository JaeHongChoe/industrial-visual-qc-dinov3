from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .manifest import CaptureRecord, ImageRecord
from .roi import estimate_placement_mask
from .synthetic import SyntheticDefectGenerator


@dataclass(frozen=True, slots=True)
class _Source:
    record: CaptureRecord
    image: ImageRecord


@dataclass(frozen=True, slots=True)
class SamplePlan:
    product: str
    condition_id: str
    capture_id: str
    group_id: str
    image_path: str
    label: int
    synthetic_family: str | None


class BalancedPatchDataset(Dataset):
    """Hierarchically sample product and condition before selecting an image."""

    def __init__(
        self,
        records: list[CaptureRecord],
        patch_size: int = 256,
        samples_per_epoch: int = 20_000,
        positive_probability: float = 0.5,
        seed: int = 17,
        source_reuse: int = 8,
        return_mask: bool = False,
        generator: SyntheticDefectGenerator | None = None,
        placement_mask_fn: Callable[[np.ndarray, int], np.ndarray] = estimate_placement_mask,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if patch_size <= 0 or samples_per_epoch <= 0 or source_reuse <= 0:
            raise ValueError("patch_size, samples_per_epoch, and source_reuse must be positive")
        if not 0.0 <= positive_probability <= 1.0:
            raise ValueError("positive_probability must be between 0 and 1")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.positive_probability = positive_probability
        self.seed = seed
        self.source_reuse = source_reuse
        self.return_mask = return_mask
        self.generator = generator or SyntheticDefectGenerator()
        self.placement_mask_fn = placement_mask_fn

        nested: dict[str, dict[str, list[_Source]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            for image in record.images:
                nested[record.product][image.condition.condition_id].append(_Source(record, image))
        self._sources = {
            product: {condition: tuple(sources) for condition, sources in sorted(conditions.items())}
            for product, conditions in sorted(nested.items())
        }
        self._products = tuple(self._sources)
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._placement_mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= len(self):
            raise IndexError(index)

    def _select_source(self, rng: np.random.Generator) -> tuple[str, str, _Source]:
        product = self._products[int(rng.integers(0, len(self._products)))]
        conditions = tuple(self._sources[product])
        condition_id = conditions[int(rng.integers(0, len(conditions)))]
        sources = self._sources[product][condition_id]
        source = sources[int(rng.integers(0, len(sources)))]
        return product, condition_id, source

    def _plan_and_content_rng(self, index: int) -> tuple[SamplePlan, np.random.Generator]:
        self._validate_index(index)
        source_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, index // self.source_reuse, 0])
        )
        content_rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 1]))
        product, condition_id, source = self._select_source(source_rng)
        label = int(content_rng.random() < self.positive_probability)
        family = (
            self.generator.families[int(content_rng.integers(0, len(self.generator.families)))]
            if label
            else None
        )
        plan = SamplePlan(
            product=product,
            condition_id=condition_id,
            capture_id=source.record.capture_id,
            group_id=source.record.group_id,
            image_path=source.image.path,
            label=label,
            synthetic_family=family,
        )
        return plan, content_rng

    def sample_plan(self, index: int) -> SamplePlan:
        plan, _ = self._plan_and_content_rng(index)
        return plan

    def _crop(self, rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = rgb.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"source image {width}x{height} is smaller than native patch size {self.patch_size}"
            )
        x0 = int(rng.integers(0, width - self.patch_size + 1))
        y0 = int(rng.integers(0, height - self.patch_size + 1))
        box = (x0, y0, x0 + self.patch_size, y0 + self.patch_size)
        return rgb[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size].copy(), box

    def _crop_near_mask(
        self,
        rgb: np.ndarray,
        valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop around a sampled ROI pixel without materializing all ROI coordinates."""
        height, width = rgb.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"source image {width}x{height} is smaller than native patch size {self.patch_size}"
            )
        active_rows = np.flatnonzero(valid_mask.any(axis=1))
        active_columns = np.flatnonzero(valid_mask.any(axis=0))
        if not active_rows.size or not active_columns.size:
            raise RuntimeError("full-image placement ROI is empty")
        y_low, y_high = int(active_rows[0]), int(active_rows[-1])
        x_low, x_high = int(active_columns[0]), int(active_columns[-1])
        for _ in range(256):
            y = int(rng.integers(y_low, y_high + 1))
            x = int(rng.integers(x_low, x_high + 1))
            if not valid_mask[y, x]:
                continue
            x_offset = int(rng.integers(0, self.patch_size))
            y_offset = int(rng.integers(0, self.patch_size))
            x0 = int(np.clip(x - x_offset, 0, width - self.patch_size))
            y0 = int(np.clip(y - y_offset, 0, height - self.patch_size))
            box = (x0, y0, x0 + self.patch_size, y0 + self.patch_size)
            return rgb[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size].copy(), box
        raise RuntimeError("could not sample a pixel from the full-image placement ROI")

    @staticmethod
    def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[:, None, None]
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, object]:
        plan, rng = self._plan_and_content_rng(index)
        source_path = Path(plan.image_path)
        rgb = self._load_rgb(source_path)
        full_valid_mask = self._load_placement_mask(source_path, rgb) if plan.label else None

        selected_patch: np.ndarray | None = None
        selected_mask: np.ndarray | None = None
        selected_box: tuple[int, int, int, int] | None = None
        synthetic_parameters: dict[str, float | int | str] | None = None
        attempts = 24 if plan.label else 1
        for _ in range(attempts):
            if plan.label:
                patch, box = self._crop_near_mask(rgb, full_valid_mask, rng)
            else:
                patch, box = self._crop(rgb, rng)
                selected_patch, selected_box = patch, box
                if self.return_mask:
                    selected_mask = np.zeros((self.patch_size, self.patch_size), dtype=bool)
                break
            x0, y0, x1, y1 = box
            valid_mask = full_valid_mask[y0:y1, x0:x1]
            try:
                result = self.generator.apply(
                    patch,
                    valid_mask,
                    family=str(plan.synthetic_family),
                    rng=rng,
                )
            except (ValueError, RuntimeError):
                continue
            if result.mask.shape != valid_mask.shape or result.mask.dtype != np.bool_:
                raise ValueError("synthetic mask must be boolean and match the patch shape")
            if np.any(result.mask & ~valid_mask):
                raise ValueError("synthetic mask must stay inside the placement ROI")
            selected_patch, selected_box = result.image, box
            if self.return_mask:
                selected_mask = result.mask
            synthetic_parameters = result.parameters
            break

        if selected_patch is None or selected_box is None or (self.return_mask and selected_mask is None):
            raise RuntimeError(f"could not sample a valid positive patch from {source_path}")

        metadata: dict[str, object] = {
            "source_label": "OK",
            "product": plan.product,
            "capture_id": plan.capture_id,
            "group_id": plan.group_id,
            "condition_id": plan.condition_id,
            "image_path": plan.image_path,
            "patch_box": selected_box,
            "synthetic_family": plan.synthetic_family,
            "synthetic_parameters": synthetic_parameters,
        }
        item: dict[str, object] = {
            "image": self._to_tensor(selected_patch),
            "label": torch.tensor(float(plan.label), dtype=torch.float32),
            "metadata": metadata,
        }
        if self.return_mask:
            item["mask"] = torch.from_numpy(np.ascontiguousarray(selected_mask)).unsqueeze(0).float()
        return item

    def _load_rgb(self, source_path: Path) -> np.ndarray:
        key = str(source_path)
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        with Image.open(source_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        self._image_cache[key] = rgb
        self._image_cache.move_to_end(key)
        while len(self._image_cache) > 2:
            evicted_key, _ = self._image_cache.popitem(last=False)
            self._placement_mask_cache.pop(evicted_key, None)
        return rgb

    def _load_placement_mask(self, source_path: Path, rgb: np.ndarray) -> np.ndarray:
        key = str(source_path)
        cached = self._placement_mask_cache.get(key)
        if cached is not None:
            self._placement_mask_cache.move_to_end(key)
            return cached
        mask = self.placement_mask_fn(rgb, 8)
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("placement_mask_fn must return a boolean mask matching the full image")
        self._placement_mask_cache[key] = mask
        self._placement_mask_cache.move_to_end(key)
        while len(self._placement_mask_cache) > 2:
            self._placement_mask_cache.popitem(last=False)
        return mask


def collate_patch_batch(items: list[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    mask_presence = ["mask" in item for item in items]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError("either every item or no item must contain a mask")
    batch: dict[str, object] = {
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "metadata": [item["metadata"] for item in items],
    }
    if all(mask_presence):
        batch["mask"] = torch.stack([item["mask"] for item in items])
    return batch
