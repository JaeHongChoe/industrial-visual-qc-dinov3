from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from patch_qc.inference import Box
from patch_qc.refinement_ranking import ReviewCandidate
from patch_qc.refinement_review import (
    export_review_case,
    render_display_heatmap,
    stitch_selected_cams,
)


def _review_candidate(box: Box, probability: float = 0.8) -> ReviewCandidate:
    return ReviewCandidate("component-1", ("patch-1",), box, 0.9, probability, 1, "inside")


def _native_rgb(width: int = 500, height: int = 300) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)


def test_stitching_uses_probability_weighted_max_for_selected_only() -> None:
    selected = [_review_candidate(Box(0, 0, 256, 256), probability=0.8)]
    cams = [np.full((256, 256), 0.5, dtype=np.float32)]

    stitched = stitch_selected_cams((300, 500), selected, cams)

    assert np.isclose(stitched[100, 100], 0.4)
    assert stitched[280, 480] == 0.0


def test_display_threshold_changes_png_not_ranked_candidates(tmp_path: Path) -> None:
    ranked = [_review_candidate(Box(0, 0, 256, 256))]
    before = tuple(candidate.patch_ids for candidate in ranked)
    score_map = np.linspace(0.0, 1.0, 300 * 500, dtype=np.float32).reshape(300, 500)

    low = render_display_heatmap(score_map, 0.3)
    high = render_display_heatmap(score_map, 0.7)
    paths = export_review_case(
        _native_rgb(),
        ranked,
        [np.full((256, 256), 0.5, dtype=np.float32)],
        {
            "display_threshold": 0.4,
            "model_digest": "1" * 64,
            "config_digest": "2" * 64,
            "schema_digest": "3" * 64,
            "timings": {"legrad_seconds": 0.1},
        },
        tmp_path,
    )
    after = tuple(candidate.patch_ids for candidate in ranked)

    assert not np.array_equal(low, high)
    assert before == after
    assert Image.open(paths["comparison"]).size == (1000, 300)
    assert np.array_equal(np.asarray(Image.open(paths["original"]).convert("RGB")), _native_rgb())
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["candidates"][0]["review_state"] == "REVIEW"
    assert metadata["output_sha256"]["comparison"]
    assert len(list(tmp_path.glob("*.png"))) == 3
