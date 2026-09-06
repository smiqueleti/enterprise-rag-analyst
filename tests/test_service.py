"""Service, API, fallback, citation, and UI contract tests without provider calls."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import create_app
from citation_validation import validate_answer_citations
from observability import MetricsRegistry, TraceRecorder
from rag import GeneratedAnswer, INSUFFICIENT_ANSWER
from retrieval import Candidate, HybridRetrievalResult, QueryPlan
from service import RAGService
from ui_client import build_query_payload


def policy_candidate() -> Candidate:
    """Create representative selected evidence with governance metadata."""
    return Candidate(
        document=Document(
            id="stable-policy-chunk",
            page_content=(
                "Potential contractual penalties above EUR 1,000 require approval "
                "from the Regional Operations Manager."
            ),
            metadata={
                "source": "financial_approval_policy.md",
                "document_id": "ATL-POL-FIN-009",
                "department": "Finance",
                "document_type": "Policy",
                "version": "1.8",
                "effective_date": "2026-03-01",
                "chunk_index": 1,
            },
        ),
        distance=0.4,
        reciprocal_rank_score=0.06,
        bm25_score=4.2,
        hybrid_score=0.9,
        rerank_position=1,
    )


def fake_plan(question: str, history: list[dict[str, str]] | None) -> QueryPlan:
    """Return the frozen four-query planner contract."""
    return QueryPlan(
        standalone_question=question,
        search_queries=[question, "financial threshold", "contractual penalty", "approver role"],
    )


def fake_retrieval(
    question: str,
    plan: QueryPlan,
    history: list[dict[str, str]] | None = None,
    embed_query=None,
) -> HybridRetrievalResult:
    """Return one selected chunk without external embeddings."""
    evidence = [policy_candidate()]
    return HybridRetrievalResult(
        plan=plan,
        candidates_by_query=[(query, evidence) for query in plan.search_queries],
        candidate_pool=evidence,
        evidence=evidence,
        embedding_latency_ms=4.0,
        retrieval_latency_ms=2.0,
        fusion_latency_ms=0.1,
        rerank_latency_ms=0.2,
        embedding_calls=4,
    )


class ServiceTests(unittest.TestCase):
    def make_service(self, generator) -> tuple[RAGService, tempfile.TemporaryDirectory]:
        directory = tempfile.TemporaryDirectory()
        service = RAGService(
            planner=fake_plan,
            retriever=fake_retrieval,
            generator=generator,
            recorder=TraceRecorder(Path(directory.name) / "requests.jsonl"),
            metrics=MetricsRegistry(),
            sleep=lambda _: None,
            primary_model="primary-test-model",
            fallback_model="fallback-test-model",
        )
        return service, directory

    def test_api_contract(self) -> None:
        def generator(question, evidence, model=None, correction=None):
            return GeneratedAnswer(
                "A penalty above EUR 1,000 requires Regional Operations Manager approval [1].",
                model,
                {"input_tokens": 20, "output_tokens": 12},
                {},
            )

        service, directory = self.make_service(generator)
        self.addCleanup(directory.cleanup)
        client = TestClient(create_app(service))
        response = client.post(
            "/query",
            json={
                "question": "Who approves a penalty above EUR 1,000?",
                "conversation_id": "conversation-1",
                "history": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["retrieval_strategy"], "hybrid_v1")
        self.assertEqual(payload["citations"][0]["chunk_id"], "stable-policy-chunk")
        self.assertTrue(payload["citation_validation"]["valid"])
        self.assertIn("trace_id", payload)
        trace_path = Path(directory.name) / "requests.jsonl"
        trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(trace["retrieval_strategy"], "hybrid_v1")
        self.assertEqual(trace["selected_evidence"][0]["chunk_id"], "stable-policy-chunk")
        self.assertIn("bm25_score", trace["candidate_pool"][0])
        self.assertNotIn("api_key", json.dumps(trace).casefold())

    def test_fallback_only_after_transient_primary_failures(self) -> None:
        calls = []

        def generator(question, evidence, model=None, correction=None):
            calls.append(model)
            if model == "primary-test-model":
                raise RuntimeError("503 UNAVAILABLE")
            return GeneratedAnswer(
                "Regional Operations Manager approval is required [1].",
                model,
                {},
                {},
            )

        service, directory = self.make_service(generator)
        self.addCleanup(directory.cleanup)
        result = service.query("Who approves the penalty?")

        self.assertEqual(
            calls,
            ["primary-test-model", "primary-test-model", "fallback-test-model"],
        )
        self.assertEqual(result["model_used"], "fallback-test-model")

    def test_invalid_citation_retries_once(self) -> None:
        calls = []

        def generator(question, evidence, model=None, correction=None):
            calls.append(correction)
            text = (
                "Regional Operations Manager approval is required [9]."
                if correction is None
                else "Regional Operations Manager approval is required [1]."
            )
            return GeneratedAnswer(text, model, {}, {})

        service, directory = self.make_service(generator)
        self.addCleanup(directory.cleanup)
        result = service.query("Who approves the penalty?")

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(correction is None or isinstance(correction, str) for correction in calls))
        self.assertTrue(result["citation_validation"]["valid"])
        self.assertEqual(result["citations"][0]["number"], 1)

    def test_bridge_claim_regression(self) -> None:
        evidence = [policy_candidate()]
        invalid = validate_answer_citations(
            "Approval thresholds apply to these costs:\n"
            "Regional Operations Manager approval is required [1].",
            evidence,
        )
        valid = validate_answer_citations(
            "Regional Operations Manager approval is required for the penalty [1].",
            evidence,
        )

        self.assertFalse(invalid.valid)
        self.assertTrue(any("no citation" in error for error in invalid.errors))
        self.assertTrue(valid.valid)

    def test_unselected_source_is_invalid(self) -> None:
        result = validate_answer_citations(
            "Regional Operations Manager approval is required [1]. "
            "See warehouse_sop.md.",
            [policy_candidate()],
        )

        self.assertFalse(result.valid)
        self.assertTrue(any("unselected sources" in error for error in result.errors))

    def test_invalid_after_one_retry_returns_safe_refusal(self) -> None:
        call_count = 0

        def generator(question, evidence, model=None, correction=None):
            nonlocal call_count
            call_count += 1
            return GeneratedAnswer("Approval thresholds apply to these costs:", model, {}, {})

        service, directory = self.make_service(generator)
        self.addCleanup(directory.cleanup)
        result = service.query("Who approves the penalty?")

        self.assertEqual(call_count, 2)
        self.assertEqual(result["answer"], INSUFFICIENT_ANSWER)
        self.assertTrue(result["refused"])
        self.assertFalse(result["citation_validation"]["valid"])

    def test_pipeline_dependency_failure_returns_api_503(self) -> None:
        def broken_retrieval(*args, **kwargs):
            raise RuntimeError("Chroma collection is corrupt")

        def generator(*args, **kwargs):
            raise AssertionError("Generation must not run after retrieval failure.")

        service, directory = self.make_service(generator)
        self.addCleanup(directory.cleanup)
        service.retriever = broken_retrieval
        client = TestClient(create_app(service))
        response = client.post("/query", json={"question": "What is the SLA?"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("trace_id", response.json()["detail"])

    def test_ui_payload_excludes_current_question_from_history(self) -> None:
        payload = build_query_payload(
            "Who approves it?",
            "conversation-1",
            [
                {"role": "user", "content": "A penalty is expected."},
                {"role": "assistant", "content": "Which amount?"},
            ],
        )

        self.assertEqual(payload["question"], "Who approves it?")
        self.assertEqual(len(payload["history"]), 2)

    def test_empty_retrieval_preserves_exact_refusal(self) -> None:
        def empty_retrieval(question, plan, history=None, embed_query=None):
            return HybridRetrievalResult(
                plan=plan,
                candidates_by_query=[],
                candidate_pool=[],
                evidence=[],
                embedding_latency_ms=0.0,
                retrieval_latency_ms=0.0,
                fusion_latency_ms=0.0,
                rerank_latency_ms=0.0,
                embedding_calls=4,
            )

        def forbidden_generator(*args, **kwargs):
            raise AssertionError("Generation must not run without evidence.")

        service, directory = self.make_service(forbidden_generator)
        self.addCleanup(directory.cleanup)
        service.retriever = empty_retrieval
        result = service.query("How many trucks does Atlas own?")

        self.assertEqual(result["answer"], INSUFFICIENT_ANSWER)
        self.assertTrue(result["refused"])


if __name__ == "__main__":
    unittest.main()
