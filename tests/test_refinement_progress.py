from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from patch_qc.refinement_progress import ProgressReporter


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_progress_reporter_throttles_cache_updates_and_flushes_final_state(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    stream = io.StringIO()
    progress_path = tmp_path / "progress.jsonl"

    with ProgressReporter(
        progress_path,
        interval_seconds=30,
        stream=stream,
        clock=clock,
    ) as reporter:
        reporter.emit("model_load", "started")
        clock.advance(2)
        reporter.emit("model_load", "completed")
        reporter.start_cache(total_sources=10)
        clock.advance(10)
        reporter.source_completed(
            completed_sources=1,
            total_patches=200,
            current_path="/data/source-001.jpg",
        )
        clock.advance(21)
        reporter.source_completed(
            completed_sources=2,
            total_patches=500,
            current_path="/data/source-002.jpg",
        )
        clock.advance(9)
        reporter.source_completed(
            completed_sources=10,
            total_patches=2600,
            current_path="/data/source-010.jpg",
        )
        reporter.finish_cache(total_patches=2600)

    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    cache_events = [event for event in events if event["stage"] == "cache_build"]

    assert [event["status"] for event in cache_events] == [
        "started",
        "running",
        "running",
        "completed",
    ]
    assert cache_events[1]["completed_sources"] == 2
    assert cache_events[1]["total_sources"] == 10
    assert cache_events[1]["total_patches"] == 500
    assert cache_events[1]["patches_per_second"] == pytest.approx(500 / 31)
    assert cache_events[1]["eta_seconds"] == pytest.approx(124)
    assert cache_events[1]["current_path"] == "/data/source-002.jpg"
    assert cache_events[2]["completed_sources"] == 10
    assert cache_events[2]["eta_seconds"] == 0.0
    assert cache_events[3]["total_patches"] == 2600
    assert reporter.event_count == len(events)
    assert "[cache_build] 2/10 sources" in stream.getvalue()
    assert "[cache_build] 10/10 sources" in stream.getvalue()


def test_progress_reporter_records_failure_before_propagating_exception(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "progress.jsonl"

    with pytest.raises(RuntimeError, match="decode failed"):
        with ProgressReporter(progress_path, interval_seconds=30) as reporter:
            reporter.emit("calibration", "started")
            raise RuntimeError("decode failed")

    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert events[-1]["stage"] == "calibration"
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "RuntimeError"
    assert events[-1]["error_message"] == "decode failed"
