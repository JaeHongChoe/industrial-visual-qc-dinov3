from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .inference import tile_boxes
from .manifest import scan_ok_tree
from .roi import estimate_placement_mask
from .split import split_by_group
from .synthetic import SyntheticDefectGenerator
from .visualize import save_synthetic_panel


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_manifest_command(args: argparse.Namespace) -> int:
    records = scan_ok_tree(args.root, start_date=args.start_date)
    if not records:
        raise RuntimeError("no eligible OK captures were found")
    splits = split_by_group(records, seed=args.seed, ratios=tuple(args.ratios))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    split_capture_counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    for split_name in ("train", "calibration", "holdout"):
        for record in splits[split_name]:
            row = record.to_dict()
            row["split"] = split_name
            rows.append(row)
            split_capture_counts[split_name] += 1
            split_image_counts[split_name] += len(record.images)
            product_counts[record.product] += len(record.images)
            condition_counts.update(image.condition.condition_id for image in record.images)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    dates = sorted(record.date for record in records)
    summary = {
        "source_label": "OK",
        "root": str(Path(args.root)),
        "start_date": args.start_date,
        "date_min": dates[0],
        "date_max": dates[-1],
        "captures": len(records),
        "groups": len({record.group_id for record in records}),
        "products": len({record.product for record in records}),
        "images": sum(len(record.images) for record in records),
        "split_captures": dict(split_capture_counts),
        "split_images": dict(split_image_counts),
        "product_images": dict(sorted(product_counts.items())),
        "condition_images": dict(sorted(condition_counts.items())),
        "seed": args.seed,
        "ratios": list(args.ratios),
    }
    _write_json(output_dir / "summary.json", summary)
    return 0


def synthetic_preview_command(args: argparse.Namespace) -> int:
    source_path = Path(args.image)
    with Image.open(source_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    full_valid_mask = estimate_placement_mask(rgb, margin=8)
    boxes = tile_boxes(rgb.shape[1], rgb.shape[0], args.patch_size, max(1, args.patch_size // 2))
    candidates = []
    for box in boxes:
        patch = rgb[box.y0 : box.y1, box.x0 : box.x1].copy()
        valid_mask = full_valid_mask[box.y0 : box.y1, box.x0 : box.x1]
        candidates.append((int(valid_mask.sum()), box, patch, valid_mask))
    valid_pixels, box, patch, valid_mask = max(candidates, key=lambda item: item[0])
    if valid_pixels == 0:
        raise RuntimeError("no foreground placement region was found in the source image")

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDefectGenerator()
    parameters: dict[str, dict[str, object]] = {}
    for family_index, family in enumerate(generator.families):
        parameters[family] = {}
        for severity in generator.severities:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, family_index]))
            result = generator.apply(
                patch,
                valid_mask,
                family=family,
                severity=severity,
                rng=rng,
            )
            stem = f"{family}_{severity}"
            save_synthetic_panel(
                patch,
                result.image,
                result.mask,
                destination / f"{stem}.png",
                draw_outline=False,
            )
            save_synthetic_panel(
                patch,
                result.image,
                result.mask,
                destination / f"{stem}_location.png",
                draw_outline=True,
            )
            parameters[family][severity] = result.parameters
    _write_json(
        destination / "summary.json",
        {
            "source_label": "OK",
            "source_image": str(source_path),
            "patch_box": [box.x0, box.y0, box.x1, box.y1],
            "families": list(generator.families),
            "severities": list(generator.severities),
            "parameters": parameters,
            "seed": args.seed,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patch-qc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("build-manifest", help="scan OK images and seal group-safe splits")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output-dir", required=True)
    manifest.add_argument("--start-date", default="2026-05-01")
    manifest.add_argument("--seed", type=int, default=17)
    manifest.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    manifest.set_defaults(handler=build_manifest_command)
    preview = subparsers.add_parser(
        "make-synthetic-preview",
        help="create one native side-by-side OK/synthetic panel per procedural family",
    )
    preview.add_argument("--image", required=True)
    preview.add_argument("--output-dir", required=True)
    preview.add_argument("--patch-size", type=int, default=256)
    preview.add_argument("--seed", type=int, default=17)
    preview.set_defaults(handler=synthetic_preview_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
