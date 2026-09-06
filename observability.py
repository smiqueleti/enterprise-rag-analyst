"""Structured local traces and aggregate runtime metrics."""

from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import TRACE_DIR


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a finite runtime sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def safe_json(value: Any) -> Any:
    """Convert provider metadata to JSON without exposing object internals."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return str(value)


class TraceRecorder:
    """Append sanitized request traces as JSON Lines."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or TRACE_DIR / "requests.jsonl"
        self._lock = threading.Lock()

    def write(self, trace: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(safe_json(trace), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")


@dataclass
class MetricsRegistry:
    """Thread-safe in-process metrics for the current service worker."""

    records: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
        stages = ["planning", "embedding", "retrieval", "reranking", "generation", "total"]
        latency = {}
        for stage in stages:
            values = [float(row["latency_ms"].get(stage, 0.0)) for row in records]
            latency[stage] = {
                "mean_ms": statistics.fmean(values) if values else 0.0,
                "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95),
            }
        count = len(records)
        return {
            "requests": count,
            "latency": latency,
            "average_embedding_calls": (
                statistics.fmean(row.get("embedding_calls", 0) for row in records)
                if records else 0.0
            ),
            "average_generation_calls": (
                statistics.fmean(row.get("generation_calls", 0) for row in records)
                if records else 0.0
            ),
            "retry_rate": (
                sum(bool(row.get("retry_count")) for row in records) / count if count else 0.0
            ),
            "refusal_rate": (
                sum(bool(row.get("refused")) for row in records) / count if count else 0.0
            ),
            "failure_rate": (
                sum(bool(row.get("failed")) for row in records) / count if count else 0.0
            ),
        }


runtime_metrics = MetricsRegistry()
