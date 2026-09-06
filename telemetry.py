"""Durable SQLite runtime telemetry without secrets or full credentials."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from config import TELEMETRY_DB


class SQLiteTelemetryStore:
    def __init__(self, path: Path = TELEMETRY_DB) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_traces (
                    trace_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    http_status INTEGER NOT NULL,
                    retrieval_strategy TEXT NOT NULL,
                    model_used TEXT,
                    fallback_used INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    queue_wait_ms REAL NOT NULL,
                    planning_latency_ms REAL NOT NULL,
                    embedding_latency_ms REAL NOT NULL,
                    retrieval_latency_ms REAL NOT NULL,
                    reranking_latency_ms REAL NOT NULL,
                    generation_latency_ms REAL NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    correction_used INTEGER NOT NULL,
                    refused INTEGER NOT NULL,
                    error_category TEXT,
                    admitted INTEGER NOT NULL,
                    rejection_reason TEXT,
                    detail_json TEXT NOT NULL
                )
                """
            )

    def record(self, trace: dict[str, Any], http_status: int) -> None:
        latency = trace.get("latency_ms", {})
        usage = trace.get("token_usage", {})
        model_used = trace.get("models", {}).get("generation_used")
        fallback = model_used == trace.get("models", {}).get("generation_fallback")
        queue_wait = sum(
            float(item.get("queue_wait_ms", 0.0))
            for item in trace.get("provider_calls", [])
        )
        values = (
            trace["trace_id"],
            trace.get("request_id") or trace["trace_id"],
            trace["timestamp"],
            http_status,
            trace.get("retrieval_strategy", "hybrid_v1"),
            model_used,
            int(fallback),
            len(trace.get("retries", [])),
            queue_wait,
            latency.get("planning", 0.0),
            latency.get("embedding", 0.0),
            latency.get("retrieval", 0.0),
            latency.get("reranking", 0.0),
            latency.get("generation", 0.0),
            latency.get("total", 0.0),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
            int(trace.get("correction_used", False)),
            int(trace.get("refused", False)),
            trace.get("error_category"),
            int(trace.get("admitted", True)),
            trace.get("rejection_reason"),
            json.dumps(trace, ensure_ascii=False),
        )
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO request_traces VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
            connection.commit()

    def rows(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            return list(connection.execute("SELECT * FROM request_traces ORDER BY timestamp"))


runtime_telemetry = SQLiteTelemetryStore()
