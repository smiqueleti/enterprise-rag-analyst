"""Checkpoint 9 reliability, authorization, telemetry, and streaming tests."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import create_app
from reliability import (
    AdmissionController,
    AdmissionRejected,
    CircuitBreaker,
    CircuitOpen,
    ProviderGate,
    QueueTimedOut,
    RequestDeadline,
    RequestDeadlineExceeded,
)
from observability import MetricsRegistry, TraceRecorder
from rag import GeneratedAnswer
from retrieval import (
    AuthorizationFilter,
    Candidate,
    HybridRetrievalResult,
    QueryPlan,
    vector_search_by_embedding,
)
from service import RAGService
from telemetry import SQLiteTelemetryStore


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search_by_vector_with_relevance_scores(self, embedding, k, filter=None):
        return [(document, 0.2 + index / 10) for index, document in enumerate(self.documents[:k])]


def make_streaming_service(directory: str) -> RAGService:
    evidence = Candidate(
        Document(
            id="stable-policy-chunk",
            page_content="Potential penalties require Regional Operations Manager approval.",
            metadata={
                "source": "financial_approval_policy.md", "document_id": "ATL-FIN",
                "department": "Finance", "document_type": "Policy", "version": "1",
                "effective_date": "2026-01-01", "chunk_index": 0,
            },
        ),
        0.2,
    )

    def planner(question, history):
        return QueryPlan(standalone_question=question, search_queries=[question, "approval", "penalty", "manager"])

    def retriever(question, plan, history=None, embed_query=None):
        return HybridRetrievalResult(plan, [(query, [evidence]) for query in plan.search_queries], [evidence], [evidence], 0, 0, 0, 0, 4)

    def generator(question, evidence, model=None, correction=None):
        return GeneratedAnswer("Regional Operations Manager approval is required [1].", model, {}, {})

    return RAGService(
        planner=planner, retriever=retriever, generator=generator,
        recorder=TraceRecorder(Path(directory) / "requests.jsonl"),
        metrics=MetricsRegistry(),
        telemetry=SQLiteTelemetryStore(Path(directory) / "telemetry.sqlite3"),
        sleep=lambda _: None,
    )


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_semaphore_enforces_ceiling(self) -> None:
        gate = ProviderGate("generation", maximum=2, queue_timeout=1)
        observations = []
        maximum_active = 0

        async def operation():
            nonlocal maximum_active
            maximum_active = max(maximum_active, gate.active)
            await asyncio.sleep(0.02)

        await asyncio.gather(*[
            gate.run(operation, RequestDeadline(1), 0, observations)
            for _ in range(6)
        ])
        self.assertEqual(maximum_active, 2)
        self.assertTrue(any(item["queue_wait_ms"] > 0 for item in observations))

    async def test_admission_rejects_at_capacity(self) -> None:
        admission = AdmissionController(1)
        await admission.admit()
        with self.assertRaises(AdmissionRejected):
            await admission.admit()
        self.assertEqual(admission.snapshot()["rejected"], 1)
        await admission.release()

    async def test_provider_queue_timeout(self) -> None:
        gate = ProviderGate("planner", maximum=1, queue_timeout=0.01)
        release = asyncio.Event()
        observations = []

        async def holding_call():
            await release.wait()

        first = asyncio.create_task(
            gate.run(holding_call, RequestDeadline(1), 0, observations)
        )
        await asyncio.sleep(0.005)
        with self.assertRaises(QueueTimedOut):
            await gate.run(lambda: asyncio.sleep(0), RequestDeadline(1), 0, observations)
        release.set()
        await first

    async def test_request_deadline_is_not_reset(self) -> None:
        gate = ProviderGate("embedding", maximum=1, queue_timeout=1)
        with self.assertRaises(RequestDeadlineExceeded):
            await gate.run(
                lambda: asyncio.sleep(0.05),
                RequestDeadline(0.01),
                0,
                [],
            )
        self.assertEqual(gate.active, 1)
        await asyncio.sleep(0.06)
        self.assertEqual(gate.active, 0)

    async def test_circuit_opens_half_opens_and_closes(self) -> None:
        circuit = CircuitBreaker("planner", 2, 0.01, 1)
        await circuit.failure(True)
        await circuit.failure(True)
        with self.assertRaises(CircuitOpen):
            await circuit.before_call()
        await asyncio.sleep(0.02)
        await circuit.before_call()
        self.assertEqual(circuit.state.value, "HALF_OPEN")
        await circuit.success()
        self.assertEqual(circuit.state.value, "CLOSED")

    async def test_nontransient_failure_does_not_open_circuit(self) -> None:
        circuit = CircuitBreaker("planner", 1, 1, 1)
        await circuit.failure(False)
        self.assertEqual(circuit.state.value, "CLOSED")

    async def test_authorization_filter_blocks_unauthorized_chunks(self) -> None:
        authorized = Document(
            id="allowed",
            page_content="Allowed",
            metadata={"tenant_id": "atlas-logistics", "access_scope": "employees"},
        )
        wrong_tenant = Document(
            id="wrong-tenant",
            page_content="Secret",
            metadata={"tenant_id": "other", "access_scope": "employees"},
        )
        wrong_scope = Document(
            id="wrong-scope",
            page_content="Executive",
            metadata={"tenant_id": "atlas-logistics", "access_scope": "executives"},
        )
        results = vector_search_by_embedding(
            [0.1],
            5,
            AuthorizationFilter("atlas-logistics", ("employees",)),
            FakeVectorStore([wrong_tenant, authorized, wrong_scope]),
        )
        self.assertEqual([item.chunk_id for item in results], ["allowed"])

    async def test_telemetry_persists_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTelemetryStore(Path(directory) / "telemetry.sqlite3")
            trace = {
                "trace_id": "trace-1", "request_id": "request-1",
                "timestamp": "2026-09-05T00:00:00+00:00",
                "retrieval_strategy": "hybrid_v1", "models": {"generation_used": "model", "generation_fallback": "fallback"},
                "retries": [], "provider_calls": [],
                "latency_ms": {"planning": 1, "embedding": 2, "retrieval": 3, "reranking": 4, "generation": 5, "total": 15},
                "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "correction_used": False, "refused": False, "admitted": True,
                "error_category": None, "rejection_reason": None,
            }
            store.record(trace, 200)
            row = store.rows()[0]
            self.assertEqual(row["trace_id"], "trace-1")
            self.assertEqual(row["total_tokens"], 15)
            self.assertEqual(row["http_status"], 200)


class StreamingTests(unittest.TestCase):
    def test_api_admission_rejection_has_retry_after(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = make_streaming_service(directory)
            service.admission = AdmissionController(0)
            response = TestClient(create_app(service)).post(
                "/query", json={"question": "What is the SLA?"}
            )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["retry-after"], "5")
            row = service.telemetry.rows()[0]
            self.assertEqual(row["rejection_reason"], "The service request queue is at capacity.")

    def test_streaming_endpoint_is_separate_and_reports_ttft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = make_streaming_service(directory)
            response = TestClient(create_app(service)).post(
                "/query/stream", json={"question": "Who approves the penalty?"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("time_to_first_token_ms", response.text)
            self.assertIn("validated_buffered", response.text)


if __name__ == "__main__":
    unittest.main()
