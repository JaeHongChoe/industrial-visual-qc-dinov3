from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from patch_qc.inference import Box
from patch_qc.refinement_contract import sha256_file
from patch_qc.refinement_evaluation import (
    LabelCase,
    LabelShape,
    candidate_intersects_shape,
    compare_with_baseline,
    evaluate_topk,
    run_sealed_evaluation,
    seal_labelme_manifest,
)
from patch_qc.refinement_ranking import ReviewCandidate


def _label_shape(shape_type: str, center: tuple[int, int], case_id: str = "case-1") -> LabelShape:
    x, y = center
    if shape_type == "polygon":
        points = ((x - 10.0, y - 10.0), (x + 12.0, y - 8.0), (x, y + 14.0))
    elif shape_type == "rectangle":
        points = ((x - 10.0, y - 10.0), (x + 10.0, y + 10.0))
    else:
        points = ((float(x), float(y)), (float(x + 10), float(y)))
    return LabelShape(case_id, f"{case_id}-shape", "chip", points, "a" * 64, shape_type)


@pytest.mark.parametrize("shape_type", ["polygon", "rectangle", "circle"])
def test_candidate_intersection_uses_shape_pixels_not_only_image_label(shape_type: str) -> None:
    shape = _label_shape(shape_type, center=(100, 100))

    assert candidate_intersects_shape(Box(64, 64, 128, 128), shape, (256, 256))
    assert not candidate_intersects_shape(Box(160, 160, 224, 224), shape, (256, 256))


def _review(component_id: str, box: Box, score: float = 0.8) -> ReviewCandidate:
    return ReviewCandidate(component_id, (component_id,), box, score, score, 1, "inside")


def _two_cases() -> tuple[LabelCase, LabelCase]:
    return (
        LabelCase(
            case_id="case-1",
            product="P1",
            lot="L1",
            image_path="/unused/one.png",
            source_sha256="1" * 64,
            json_sha256="a" * 64,
            width=256,
            height=256,
            shapes=(_label_shape("polygon", (80, 80), "case-1"),),
        ),
        LabelCase(
            case_id="case-2",
            product="P2",
            lot="L2",
            image_path="/unused/two.png",
            source_sha256="2" * 64,
            json_sha256="b" * 64,
            width=256,
            height=256,
            shapes=(_label_shape("rectangle", (200, 200), "case-2"),),
        ),
    )


def _ranked_candidates() -> dict[str, list[ReviewCandidate]]:
    return {
        "case-1": [_review("hit-1", Box(64, 64, 128, 128))],
        "case-2": [
            _review("miss-1", Box(0, 0, 32, 32), 0.9),
            _review("miss-2", Box(40, 0, 72, 32), 0.8),
            _review("miss-3", Box(80, 0, 112, 32), 0.7),
            _review("miss-4", Box(120, 0, 152, 32), 0.6),
            _review("hit-2", Box(180, 180, 224, 224), 0.5),
        ],
    }


def test_topk_metrics_keep_shape_and_image_denominators_explicit() -> None:
    metrics = evaluate_topk(_two_cases(), _ranked_candidates(), ks=(3, 5))

    assert metrics["shape_recall_at_3"] == 0.5
    assert metrics["shape_recall_at_5"] == 1.0
    assert metrics["shape_denominator"] == 2
    assert metrics["image_recall_at_3"] == 0.5
    assert metrics["image_recall_at_5"] == 1.0
    assert metrics["image_denominator"] == 2
    assert metrics["denominators_by_product"] == {"P1": 1, "P2": 1}


def test_baseline_comparison_names_gained_lost_and_acceptance() -> None:
    refined = evaluate_topk(_two_cases(), _ranked_candidates(), ks=(3, 5))
    baseline_candidates = {
        "case-1": [_review("baseline-miss", Box(0, 0, 32, 32))],
        "case-2": [_review("baseline-hit", Box(180, 180, 224, 224))],
    }
    baseline = evaluate_topk(_two_cases(), baseline_candidates, ks=(3, 5))

    comparison = compare_with_baseline(refined, baseline)

    assert comparison["gained_shapes_at_5"] == ["case-1::case-1-shape"]
    assert comparison["lost_shapes_at_5"] == []
    assert comparison["acceptance"]["shape_recall_at_5_non_decrease"] is True


def _write_labelme_case(root: Path) -> tuple[Path, Path]:
    case_dir = root / "P1" / "LOT1"
    case_dir.mkdir(parents=True)
    image_path = case_dir / "case.png"
    Image.fromarray(np.full((256, 256, 3), 120, dtype=np.uint8)).save(image_path)
    json_path = case_dir / "case.json"
    json_path.write_text(
        json.dumps(
            {
                "imagePath": "case.png",
                "imageHeight": 256,
                "imageWidth": 256,
                "shapes": [
                    {
                        "label": "chip",
                        "shape_type": "rectangle",
                        "points": [[80, 80], [110, 110]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return image_path, json_path


def test_labelme_manifest_is_hash_sealed_and_evaluation_is_one_shot(tmp_path: Path) -> None:
    labelme_root = tmp_path / "labelme"
    image_path, _ = _write_labelme_case(labelme_root)
    manifest_path = tmp_path / "sealed_manifest.json"
    manifest_digest = seal_labelme_manifest(labelme_root, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_id = manifest["cases"][0]["case_id"]
    digests = {
        "config": "1" * 64,
        "candidate_policy": "2" * 64,
        "feature_schema": "3" * 64,
        "normal_memory": "4" * 64,
        "selected_refiner": "5" * 64,
        "labelme_manifest": manifest_digest,
    }
    artifacts = {
        "digests": digests.copy(),
        "training_hashes": [],
        "ranked_candidates": {case_id: [_review("refined", Box(64, 64, 128, 128))]},
        "baseline_candidates": {case_id: [_review("baseline", Box(0, 0, 32, 32))]},
    }
    output = tmp_path / "sealed_eval"

    receipt = run_sealed_evaluation(
        {"digests": digests}, artifacts, manifest_path, output
    )

    assert sha256_file(manifest_path) == manifest_digest
    assert receipt["status"] == "completed"
    assert (output / "metrics.json").is_file()
    assert (output / "baseline_comparison.json").is_file()
    with pytest.raises(FileExistsError):
        run_sealed_evaluation({"digests": digests}, artifacts, manifest_path, output)

    collision = {**artifacts, "training_hashes": [sha256_file(image_path)]}
    with pytest.raises(RuntimeError, match="disjoint"):
        run_sealed_evaluation(
            {"digests": digests}, collision, manifest_path, tmp_path / "collision_eval"
        )
