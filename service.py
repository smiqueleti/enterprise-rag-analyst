"""Reliable orchestration for the frozen Atlas hybrid_v1 RAG pipeline."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from citation_validation import CitationValidation, validate_answer_citations
from config import (
    CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_HALF_OPEN_MAX_CALLS,
    CIRCUIT_RECOVERY_SECONDS, EMBEDDING_MODEL, FALLBACK_CHAT_MODEL, HYBRID_V1,
    MAX_EMBEDDING_CONCURRENCY, MAX_GENERATION_CONCURRENCY,
    MAX_PENDING_REQUESTS, MAX_PLANNER_CONCURRENCY,
    MIN_PROVIDER_CALL_BUDGET_SECONDS, PLANNER_MODEL, PRIMARY_CHAT_MODEL,
    PROVIDER_BACKOFF_SECONDS, PROVIDER_MAX_ATTEMPTS, QUEUE_TIMEOUT_SECONDS,
    REQUEST_DEADLINE_SECONDS, RETRY_AFTER_SECONDS,
)
from observability import MetricsRegistry, TraceRecorder, runtime_metrics
from providers import get_embeddings
from rag import INSUFFICIENT_ANSWER, GeneratedAnswer, generate_answer_with_model
from reliability import (
    AdmissionController, AdmissionRejected, CircuitBreaker, CircuitOpen,
    ProviderGate, QueueTimedOut, RequestDeadline, RequestDeadlineExceeded,
)
from retrieval import (
    AuthorizationFilter, Candidate, HybridRetrievalResult, QueryPlan,
    assemble_hybrid_v1, get_vectorstore, plan_queries,
    retrieve_hybrid_v1_from_plan, vector_search_by_embedding,
)
from telemetry import SQLiteTelemetryStore, runtime_telemetry


class ServiceUnavailable(RuntimeError):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


class ServiceRejected(ServiceUnavailable):
    def __init__(self, message: str, trace_id: str, status_code: int, reason: str) -> None:
        super().__init__(message, trace_id)
        self.status_code = status_code
        self.reason = reason
        self.retry_after = RETRY_AFTER_SECONDS


class MalformedProviderResponse(RuntimeError):
    pass


def is_transient_provider_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".casefold()
    markers = (
        "429", "resource_exhausted", "rate limit", "quota", "503", "unavailable",
        "timeout", "timed out", "temporarily", "connectionerror", "connecterror",
    )
    return isinstance(error, (TimeoutError, CircuitOpen)) or any(marker in text for marker in markers)


class RAGService:
    """Coordinate modular RAG stages under bounded runtime controls."""

    def __init__(
        self,
        planner: Callable[..., QueryPlan] = plan_queries,
        retriever: Callable[..., HybridRetrievalResult] = retrieve_hybrid_v1_from_plan,
        generator: Callable[..., GeneratedAnswer] = generate_answer_with_model,
        recorder: TraceRecorder | None = None,
        metrics: MetricsRegistry | None = None,
        sleep: Callable[[float], None] | None = None,
        primary_model: str = PRIMARY_CHAT_MODEL,
        fallback_model: str = FALLBACK_CHAT_MODEL,
        telemetry: SQLiteTelemetryStore | None = None,
        admission: AdmissionController | None = None,
        deadline_seconds: float = REQUEST_DEADLINE_SECONDS,
        planner_gate: ProviderGate | None = None,
        embedding_gate: ProviderGate | None = None,
        generation_gate: ProviderGate | None = None,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.generator = generator
        self.recorder = recorder or TraceRecorder()
        self.metrics = metrics or runtime_metrics
        self.sleep = sleep
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.telemetry = telemetry or runtime_telemetry
        self.admission = admission or AdmissionController(MAX_PENDING_REQUESTS)
        self.deadline_seconds = deadline_seconds
        self.gates = {
            "planning": planner_gate or ProviderGate("planning", MAX_PLANNER_CONCURRENCY, QUEUE_TIMEOUT_SECONDS),
            "embedding": embedding_gate or ProviderGate("embedding", MAX_EMBEDDING_CONCURRENCY, QUEUE_TIMEOUT_SECONDS),
            "generation": generation_gate or ProviderGate("generation", MAX_GENERATION_CONCURRENCY, QUEUE_TIMEOUT_SECONDS),
        }
        self.circuits = {
            name: CircuitBreaker(name, CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS, CIRCUIT_HALF_OPEN_MAX_CALLS)
            for name in ("planning", "embedding", "generation_primary", "generation_fallback")
        }

    def query(
        self,
        question: str,
        conversation_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        authorization: AuthorizationFilter | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self.query_async(question, conversation_id, history, authorization, request_id))

    async def query_async(
        self,
        question: str,
        conversation_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        authorization: AuthorizationFilter | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        trace_id = str(uuid.uuid4())
        request_id = request_id or trace_id
        timestamp = datetime.now(UTC).isoformat()
        deadline = RequestDeadline(self.deadline_seconds)
        started_total = time.perf_counter()
        provider_calls: list[dict[str, Any]] = []
        retries: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        latency = {key: 0.0 for key in ("planning", "embedding", "retrieval", "reranking", "generation", "total")}
        plan: QueryPlan | None = None
        retrieval: HybridRetrievalResult | None = None
        generated = GeneratedAnswer("", "none", {}, {})
        validation = CitationValidation(False, [], ["Request did not complete."], 0)
        generation_calls = 0
        correction_used = False
        admitted = False

        try:
            await self.admission.admit()
            admitted = True
        except AdmissionRejected as error:
            trace = self._build_trace(
                trace_id, request_id, timestamp, conversation_id, question, history,
                plan, retrieval, generated, validation, latency, retries, errors,
                generation_calls, provider_calls, correction_used, False,
                "admission_capacity", str(error),
            )
            self._persist(trace, 429, True)
            raise ServiceRejected(str(error), trace_id, 429, "admission_capacity") from error

        try:
            started = time.perf_counter()
            plan = await self._provider_retry(
                "planning", lambda: self._validated_plan(self.planner(question, history)), deadline,
                retries, provider_calls, malformed_retry=True,
            )
            latency["planning"] = (time.perf_counter() - started) * 1000

            if self.retriever is retrieve_hybrid_v1_from_plan:
                retrieval = await self._retrieve_default(
                    question, plan, history, authorization, deadline, retries, provider_calls
                )
            else:
                retrieval = await self._local_with_deadline(
                    lambda: self.retriever(
                        question, plan, history=history, embed_query=None
                    ),
                    deadline,
                    "retrieval",
                )
            latency["embedding"] = retrieval.embedding_latency_ms
            latency["retrieval"] = retrieval.retrieval_latency_ms
            latency["reranking"] = retrieval.fusion_latency_ms + retrieval.rerank_latency_ms

            if not retrieval.evidence:
                generated = GeneratedAnswer(INSUFFICIENT_ANSWER, "deterministic_refusal", {}, {})
                validation = CitationValidation(True, [], [], 0)
            else:
                started = time.perf_counter()
                generated, calls = await self._generate_with_fallback(
                    question, retrieval.evidence, deadline, retries, provider_calls
                )
                generation_calls += calls
                validation = validate_answer_citations(generated.text, retrieval.evidence, question=question)
                if not validation.valid:
                    correction_used = True
                    retries.append({"stage": "citation_validation", "attempt": 1, "model": generated.model, "delay_seconds": 0.0, "error_type": "invalid_citations"})
                    deadline.require(MIN_PROVIDER_CALL_BUDGET_SECONDS)
                    corrected, calls = await self._generate_with_fallback(
                        question, retrieval.evidence, deadline, retries, provider_calls,
                        preferred_model=generated.model,
                        correction=validation.correction_prompt(),
                    )
                    generation_calls += calls
                    corrected_validation = validate_answer_citations(corrected.text, retrieval.evidence, question=question)
                    if corrected_validation.valid:
                        generated, validation = corrected, corrected_validation
                    else:
                        errors.append({"stage": "citation_validation", "type": "invalid_after_retry", "message": "; ".join(corrected_validation.errors)})
                        generated = GeneratedAnswer(INSUFFICIENT_ANSWER, corrected.model, corrected.token_usage, corrected.response_metadata)
                        validation = corrected_validation
                latency["generation"] = (time.perf_counter() - started) * 1000

            latency["total"] = (time.perf_counter() - started_total) * 1000
            trace = self._build_trace(
                trace_id, request_id, timestamp, conversation_id, question, history,
                plan, retrieval, generated, validation, latency, retries, errors,
                generation_calls, provider_calls, correction_used, True, None, None,
            )
            self._persist(trace, 200, False)
            return self._response(trace, retrieval)
        except (QueueTimedOut, RequestDeadlineExceeded) as error:
            status = 503 if isinstance(error, QueueTimedOut) else 504
            category = "provider_queue_timeout" if status == 503 else "request_deadline"
            errors.append({"stage": "pipeline", "type": type(error).__name__, "message": str(error)})
            latency["total"] = (time.perf_counter() - started_total) * 1000
            trace = self._build_trace(
                trace_id, request_id, timestamp, conversation_id, question, history,
                plan, retrieval, generated, validation, latency, retries, errors,
                generation_calls, provider_calls, correction_used, admitted, category, str(error),
            )
            self._persist(trace, status, True)
            raise ServiceRejected(str(error), trace_id, status, category) from error
        except Exception as error:
            errors.append({"stage": "pipeline", "type": type(error).__name__, "message": str(error)})
            latency["total"] = (time.perf_counter() - started_total) * 1000
            trace = self._build_trace(
                trace_id, request_id, timestamp, conversation_id, question, history,
                plan, retrieval, generated, validation, latency, retries, errors,
                generation_calls, provider_calls, correction_used, admitted,
                type(error).__name__, str(error),
            )
            self._persist(trace, 503, True)
            raise ServiceUnavailable("The RAG service is temporarily unavailable.", trace_id) from error
        finally:
            if admitted:
                await self.admission.release()

    async def _retrieve_default(
        self,
        question: str,
        plan: QueryPlan,
        history: list[dict[str, str]] | None,
        authorization: AuthorizationFilter | None,
        deadline: RequestDeadline,
        retries: list[dict[str, Any]],
        provider_calls: list[dict[str, Any]],
    ) -> HybridRetrievalResult:
        ranked_results = []
        embedding_ms = 0.0
        retrieval_ms = 0.0
        for query in plan.search_queries:
            started = time.perf_counter()
            embedding = await self._provider_retry(
                "embedding", lambda query=query: get_embeddings().embed_query(query),
                deadline, retries, provider_calls,
            )
            embedding_ms += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            candidates = await self._local_with_deadline(
                lambda embedding=embedding: vector_search_by_embedding(
                    embedding, HYBRID_V1.candidates_per_query, authorization
                ),
                deadline,
                "retrieval",
            )
            retrieval_ms += (time.perf_counter() - started) * 1000
            ranked_results.append((query, candidates))
        return await self._local_with_deadline(
            lambda: assemble_hybrid_v1(
                question, plan, ranked_results, history, embedding_ms, retrieval_ms
            ),
            deadline,
            "reranking",
        )

    async def _local_with_deadline(
        self,
        operation: Callable[[], Any],
        deadline: RequestDeadline,
        stage: str,
    ) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(operation), timeout=deadline.require()
            )
        except TimeoutError as error:
            raise RequestDeadlineExceeded(
                f"The request deadline expired during {stage}."
            ) from error

    def _validated_plan(self, plan: QueryPlan) -> QueryPlan:
        if len(plan.search_queries) != HYBRID_V1.atomic_queries:
            raise MalformedProviderResponse(
                f"Planner returned {len(plan.search_queries)} queries, expected 4."
            )
        return plan

    async def _provider_retry(
        self,
        stage: str,
        operation: Callable[[], Any],
        deadline: RequestDeadline,
        retries: list[dict[str, Any]],
        provider_calls: list[dict[str, Any]],
        malformed_retry: bool = False,
        circuit_name: str | None = None,
    ) -> Any:
        circuit = self.circuits[circuit_name or stage]
        gate = self.gates["generation" if stage == "generation" else stage]
        for attempt in range(1, PROVIDER_MAX_ATTEMPTS + 1):
            deadline.require(MIN_PROVIDER_CALL_BUDGET_SECONDS)
            await circuit.before_call()
            try:
                result = await gate.run(
                    lambda: asyncio.to_thread(operation), deadline,
                    MIN_PROVIDER_CALL_BUDGET_SECONDS, provider_calls,
                )
                if hasattr(result, "text") and not result.text.strip():
                    raise MalformedProviderResponse("Provider returned empty text.")
                await circuit.success()
                return result
            except Exception as error:
                transient = is_transient_provider_error(error)
                malformed = malformed_retry and isinstance(error, MalformedProviderResponse)
                await circuit.failure(transient)
                if not (transient or malformed) or attempt == PROVIDER_MAX_ATTEMPTS:
                    raise
                delay = PROVIDER_BACKOFF_SECONDS * (2 ** (attempt - 1))
                retries.append({"stage": stage, "attempt": attempt, "delay_seconds": delay, "error_type": type(error).__name__})
                await self._backoff(delay, deadline)
        raise RuntimeError("Retry loop ended unexpectedly.")

    async def _backoff(self, delay: float, deadline: RequestDeadline) -> None:
        deadline.require(delay)
        if self.sleep is None:
            await asyncio.sleep(delay)
        else:
            await asyncio.to_thread(self.sleep, delay)

    async def _generate_with_fallback(
        self,
        question: str,
        evidence: list[Candidate],
        deadline: RequestDeadline,
        retries: list[dict[str, Any]],
        provider_calls: list[dict[str, Any]],
        preferred_model: str | None = None,
        correction: str | None = None,
    ) -> tuple[GeneratedAnswer, int]:
        primary = preferred_model or self.primary_model
        before = len([item for item in provider_calls if item["stage"] == "generation"])
        try:
            result = await self._provider_retry(
                "generation",
                lambda: self.generator(question, evidence, model=primary, correction=correction),
                deadline, retries, provider_calls, malformed_retry=True,
                circuit_name="generation_primary",
            )
            calls = len([item for item in provider_calls if item["stage"] == "generation"]) - before
            return result, calls
        except Exception as error:
            if not is_transient_provider_error(error):
                raise
        if not self.fallback_model or self.fallback_model == primary:
            raise ServiceUnavailable("No generation fallback is configured.", "pending")
        deadline.require(MIN_PROVIDER_CALL_BUDGET_SECONDS)
        retries.append({"stage": "generation_fallback", "attempt": 1, "model": self.fallback_model, "delay_seconds": 0.0, "error_type": "primary_transient_failure"})
        result = await self._provider_retry(
            "generation",
            lambda: self.generator(question, evidence, model=self.fallback_model, correction=correction),
            deadline, retries, provider_calls, malformed_retry=True,
            circuit_name="generation_fallback",
        )
        calls = len([item for item in provider_calls if item["stage"] == "generation"]) - before
        return result, calls

    def _build_trace(
        self, trace_id: str, request_id: str, timestamp: str,
        conversation_id: str | None, question: str,
        history: list[dict[str, str]] | None, plan: QueryPlan | None,
        retrieval: HybridRetrievalResult | None, generated: GeneratedAnswer,
        validation: CitationValidation, latency: dict[str, float],
        retries: list[dict[str, Any]], errors: list[dict[str, Any]],
        generation_calls: int, provider_calls: list[dict[str, Any]],
        correction_used: bool, admitted: bool, error_category: str | None,
        rejection_reason: str | None,
    ) -> dict[str, Any]:
        evidence = retrieval.evidence if retrieval else []
        return {
            "trace_id": trace_id, "request_id": request_id, "timestamp": timestamp,
            "conversation_id": conversation_id, "original_query": question,
            "history_turns": len(history or []),
            "rewritten_query": plan.standalone_question if plan else None,
            "atomic_queries": plan.search_queries if plan else [],
            "retrieved_candidates": [
                {"query": query, "chunks": [self._candidate_trace(item, rank) for rank, item in enumerate(items, 1)]}
                for query, items in (retrieval.candidates_by_query if retrieval else [])
            ],
            "candidate_pool": [self._candidate_trace(item, rank) for rank, item in enumerate(retrieval.candidate_pool if retrieval else [], 1)],
            "selected_evidence": [{**self._candidate_trace(item, rank), "text": item.document.page_content} for rank, item in enumerate(evidence, 1)],
            "source_documents": sorted({str(item.document.metadata.get("source")) for item in evidence}),
            "models": {"planner": PLANNER_MODEL, "embedding": EMBEDDING_MODEL, "generation_primary": self.primary_model, "generation_fallback": self.fallback_model, "generation_used": generated.model},
            "token_usage": generated.token_usage, "latency_ms": latency,
            "refused": generated.text == INSUFFICIENT_ANSWER, "answer": generated.text,
            "generated_citations": validation.citations,
            "citation_validation": {"valid": validation.valid, "errors": validation.errors, "claims_checked": validation.claims_checked},
            "embedding_calls": retrieval.embedding_calls if retrieval else 0,
            "generation_calls": generation_calls, "retries": retries, "errors": errors,
            "retrieval_strategy": HYBRID_V1.name, "provider_calls": provider_calls,
            "correction_used": correction_used, "admitted": admitted,
            "error_category": error_category, "rejection_reason": rejection_reason,
            "circuit_transitions": {name: list(circuit.transitions) for name, circuit in self.circuits.items()},
        }

    @staticmethod
    def _candidate_trace(candidate: Candidate, rank: int) -> dict[str, Any]:
        metadata = candidate.document.metadata
        return {
            "rank": rank, "chunk_id": candidate.chunk_id,
            "source": metadata.get("source"), "chunk_index": metadata.get("chunk_index"),
            "tenant_id": metadata.get("tenant_id"), "access_scope": metadata.get("access_scope"),
            "vector_distance_l2": candidate.distance,
            "vector_similarity": candidate.vector_similarity,
            "rrf_score": candidate.reciprocal_rank_score,
            "bm25_score": candidate.bm25_score, "hybrid_score": candidate.hybrid_score,
        }

    def _response(self, trace: dict[str, Any], retrieval: HybridRetrievalResult | None) -> dict[str, Any]:
        evidence = retrieval.evidence if retrieval else []
        citations = [
            {"number": number, "chunk_id": evidence[number - 1].chunk_id, "source": evidence[number - 1].document.metadata["source"]}
            for number in trace["generated_citations"] if 1 <= number <= len(evidence)
        ]
        sources = []
        for rank, candidate in enumerate(evidence, 1):
            metadata = candidate.document.metadata
            sources.append({
                "citation": rank, "chunk_id": candidate.chunk_id,
                "document": metadata["source"], "document_id": metadata["document_id"],
                "department": metadata["department"], "document_type": metadata["document_type"],
                "version": metadata["version"], "effective_date": metadata["effective_date"],
                "tenant_id": metadata.get("tenant_id"), "access_scope": metadata.get("access_scope"),
                "chunk_index": metadata["chunk_index"], "vector_similarity": candidate.vector_similarity,
                "rrf_score": candidate.reciprocal_rank_score, "bm25_score": candidate.bm25_score,
                "hybrid_score": candidate.hybrid_score, "text": candidate.document.page_content,
            })
        return {
            "answer": trace["answer"], "citations": citations, "sources": sources,
            "refused": trace["refused"], "retrieval_strategy": HYBRID_V1.name,
            "latency_ms": trace["latency_ms"], "trace_id": trace["trace_id"],
            "model_used": trace["models"]["generation_used"], "token_usage": trace["token_usage"],
            "citation_validation": trace["citation_validation"],
            "debug": {"rewritten_query": trace["rewritten_query"], "atomic_queries": trace["atomic_queries"], "selected_evidence": trace["selected_evidence"], "provider_calls": trace["provider_calls"], "retries": trace["retries"]},
        }

    def _persist(self, trace: dict[str, Any], status: int, failed: bool) -> None:
        self.recorder.write(trace)
        self.telemetry.record(trace, status)
        self.metrics.observe({"latency_ms": trace["latency_ms"], "embedding_calls": trace["embedding_calls"], "generation_calls": trace["generation_calls"], "retry_count": len(trace["retries"]), "refused": trace["refused"], "failed": failed})

    def reliability_status(self) -> dict[str, Any]:
        return {
            "admission": self.admission.snapshot(),
            "provider_gates": {name: gate.snapshot() for name, gate in self.gates.items()},
            "circuits": {name: circuit.snapshot() for name, circuit in self.circuits.items()},
        }


def health_status() -> dict[str, Any]:
    try:
        chunks = len(get_vectorstore().get(include=[])["ids"])
        status = "ok" if chunks else "degraded"
    except Exception:
        chunks, status = 0, "degraded"
    return {"status": status, "retrieval_strategy": HYBRID_V1.name, "vector_chunks": chunks, "primary_model": PRIMARY_CHAT_MODEL, "fallback_model": FALLBACK_CHAT_MODEL}
