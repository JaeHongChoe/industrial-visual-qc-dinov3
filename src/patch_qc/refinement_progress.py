from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


class ProgressReporter:
    """Append-only stage and throttled cache progress reporting."""

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float,
        stream: IO[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        self.stream = sys.stdout if stream is None else stream
        self.clock = clock
        self.event_count = 0
        self._handle: IO[str] | None = None
        self._started_at = 0.0
        self._current_stage = "initialization"
        self._cache_started_at: float | None = None
        self._last_cache_emit_at: float | None = None
        self._total_sources: int | None = None
        self._completed_sources = 0
        self._total_patches = 0

    def __enter__(self) -> ProgressReporter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._started_at = self.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if exc_value is not None:
                self.emit(
                    self._current_stage,
                    "failed",
                    error_type=type(exc_value).__name__,
                    error_message=str(exc_value),
                )
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        return False

    def emit(self, stage: str, status: str, **fields: Any) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("ProgressReporter must be used as a context manager")
        now = self.clock()
        self._current_stage = stage
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "elapsed_seconds": max(0.0, now - self._started_at),
            **fields,
        }
        self._handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()
        self.event_count += 1
        self.stream.write(self._format_line(event) + "\n")
        self.stream.flush()
        return event

    def start_cache(self, *, total_sources: int) -> None:
        if total_sources < 0:
            raise ValueError("total_sources must be non-negative")
        now = self.clock()
        self._cache_started_at = now
        self._last_cache_emit_at = now
        self._total_sources = total_sources
        self._completed_sources = 0
        self._total_patches = 0
        self.emit(
            "cache_build",
            "started",
            completed_sources=0,
            total_sources=total_sources,
            total_patches=0,
        )

    def source_completed(
        self,
        *,
        completed_sources: int,
        total_patches: int,
        current_path: str,
    ) -> None:
        if self._cache_started_at is None or self._total_sources is None:
            raise RuntimeError("start_cache must be called before source_completed")
        if not 0 <= completed_sources <= self._total_sources:
            raise ValueError("completed_sources is outside the cache source boundary")
        now = self.clock()
        is_final_source = completed_sources == self._total_sources
        if (
            not is_final_source
            and self._last_cache_emit_at is not None
            and now - self._last_cache_emit_at < self.interval_seconds
        ):
            return
        elapsed = max(0.0, now - self._cache_started_at)
        rate = total_patches / elapsed if elapsed > 0 else 0.0
        eta = (
            elapsed * (self._total_sources - completed_sources) / completed_sources
            if completed_sources > 0
            else None
        )
        self._completed_sources = completed_sources
        self._total_patches = total_patches
        self._last_cache_emit_at = now
        self.emit(
            "cache_build",
            "running",
            completed_sources=completed_sources,
            total_sources=self._total_sources,
            total_patches=total_patches,
            patches_per_second=rate,
            eta_seconds=eta,
            current_path=current_path,
        )

    def finish_cache(self, *, total_patches: int) -> None:
        if self._total_sources is None:
            raise RuntimeError("start_cache must be called before finish_cache")
        self.emit(
            "cache_build",
            "completed",
            completed_sources=self._total_sources,
            total_sources=self._total_sources,
            total_patches=total_patches,
        )

    @staticmethod
    def _format_line(event: dict[str, Any]) -> str:
        if event["stage"] == "cache_build" and "completed_sources" in event:
            line = (
                f"[cache_build] {event['completed_sources']}/{event['total_sources']} sources"
                f" patches={event['total_patches']} status={event['status']}"
            )
            if "patches_per_second" in event:
                line += f" pps={event['patches_per_second']:.2f}"
            if event.get("eta_seconds") is not None:
                line += f" eta={event['eta_seconds']:.0f}s"
            if event.get("current_path"):
                line += f" current={event['current_path']}"
            return line
        line = f"[{event['stage']}] status={event['status']}"
        if event.get("error_message"):
            line += f" error={event['error_message']}"
        return line
