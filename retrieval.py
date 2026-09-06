"""Modular retrieval strategies for the Atlas Logistics knowledge base."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from collections.abc import Callable
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from config import COLLECTION_NAME, HYBRID_V1, PLANNER_MODEL, VECTOR_DIR
from providers import get_chat_model, get_embeddings


RRF_CONSTANT = 60
DEFAULT_EXPANSION_COUNT = 4

RetrievalStrategy = Literal[
    "baseline",
    "rewrite",
    "multi_query",
    "multi_query_rerank",
    "hybrid_v1",
]
RerankMethod = Literal["rrf", "bm25", "hybrid"]


@dataclass(frozen=True)
class AuthorizationFilter:
    """Retrieval boundary supplied by an upstream authorization decision."""

    tenant_id: str
    access_scopes: tuple[str, ...]


class QueryPlan(BaseModel):
    """Standalone rewrite plus diverse vector-search queries."""

    standalone_question: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=3, max_length=4)


@dataclass
class Candidate:
    """One stable Chroma chunk plus its retrieval and reranking signals."""

    document: Document
    distance: float
    reciprocal_rank_score: float = 0.0
    matched_queries: list[str] = field(default_factory=list)
    best_vector_rank: int | None = None
    rrf_rank: int | None = None
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    hybrid_score: float = 0.0
    rerank_position: int | None = None

    @property
    def chunk_id(self) -> str:
        if self.document.id:
            return self.document.id
        metadata_id = self.document.metadata.get("chunk_id")
        if metadata_id:
            return str(metadata_id)
        raise ValueError("Retrieved document has no stable chunk ID.")

    @property
    def vector_similarity(self) -> float:
        """Return a bounded monotonic proxy for the collection's L2 distance."""
        return 1.0 / (1.0 + max(0.0, self.distance))

    def clone(self) -> Candidate:
        """Copy ranking state so controlled reranking experiments stay isolated."""
        return replace(self, matched_queries=list(self.matched_queries))


@dataclass
class RetrievalDiagnostics:
    """Inspectable execution details and logical external-call counts."""

    strategy: RetrievalStrategy
    original_question: str
    standalone_question: str
    search_queries: list[str]
    candidate_count: int
    embedding_calls: int
    llm_calls: int
    rewrite_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    embedding_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.rewrite_latency_ms
            + self.embedding_latency_ms
            + self.retrieval_latency_ms
            + self.fusion_latency_ms
            + self.rerank_latency_ms
        )


@dataclass
class RetrievalOutcome:
    """Final evidence and diagnostics returned by a retrieval strategy."""

    evidence: list[Candidate]
    diagnostics: RetrievalDiagnostics


@dataclass
class HybridRetrievalResult:
    """Full inspectable result for the frozen hybrid_v1 profile."""

    plan: QueryPlan
    candidates_by_query: list[tuple[str, list[Candidate]]]
    candidate_pool: list[Candidate]
    evidence: list[Candidate]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    fusion_latency_ms: float
    rerank_latency_ms: float
    embedding_calls: int


def format_history(history: list[dict[str, str]] | None) -> str:
    """Render recent conversation turns for query rewriting only."""
    if not history:
        return "No previous conversation."

    lines = []
    for message in history:
        role = message.get("role", "user").upper()
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous conversation."


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open the persisted Chroma collection with Gemini embeddings."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=get_embeddings(),
    )
    if not vectorstore.get(include=[])["ids"]:
        raise RuntimeError("The vector store is empty. Run `python ingest.py` first.")
    return vectorstore

def vector_search(query: str, k: int) -> list[Candidate]:
    """Run one vector query while retaining Chroma IDs and distances."""
    results = get_vectorstore().similarity_search_with_score(query, k=k)
    return [
        Candidate(document=document, distance=distance, best_vector_rank=rank)
        for rank, (document, distance) in enumerate(results, start=1)
    ]


def _is_authorized(document: Document, authorization: AuthorizationFilter | None) -> bool:
    if authorization is None:
        return True
    metadata = document.metadata
    return (
        metadata.get("tenant_id") == authorization.tenant_id
        and metadata.get("access_scope") in authorization.access_scopes
    )


def vector_search_by_embedding(
    embedding: list[float],
    k: int,
    authorization: AuthorizationFilter | None = None,
    vectorstore: Chroma | None = None,
) -> list[Candidate]:
    """Search Chroma with a precomputed embedding so stages can be timed separately."""
    store = vectorstore or get_vectorstore()
    filter_expression = None
    if authorization is not None:
        filter_expression = {
            "$and": [
                {"tenant_id": {"$eq": authorization.tenant_id}},
                {"access_scope": {"$in": list(authorization.access_scopes)}},
            ]
        }
    results = store.similarity_search_by_vector_with_relevance_scores(
        embedding, k=k, filter=filter_expression
    )
    return [
        Candidate(document=document, distance=distance, best_vector_rank=rank)
        for rank, (document, distance) in enumerate(results, start=1)
        if _is_authorized(document, authorization)
    ]


def assemble_hybrid_v1(
    question: str,
    plan: QueryPlan,
    ranked_results: list[tuple[str, list[Candidate]]],
    history: list[dict[str, str]] | None = None,
    embedding_latency_ms: float = 0.0,
    retrieval_latency_ms: float = 0.0,
) -> HybridRetrievalResult:
    """Fuse and rerank pre-fetched candidates using the frozen hybrid_v1 profile."""
    started = time.perf_counter()
    candidate_pool = merge_ranked_results(
        ranked_results, candidate_limit=HYBRID_V1.candidate_pool
    )
    fusion_latency_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    evidence = rerank_candidates(
        question,
        candidate_pool,
        top_k=HYBRID_V1.final_evidence,
        history=history,
        search_queries=plan.search_queries,
        method="hybrid",
    )
    rerank_latency_ms = (time.perf_counter() - started) * 1000
    return HybridRetrievalResult(
        plan=plan,
        candidates_by_query=ranked_results,
        candidate_pool=candidate_pool,
        evidence=evidence,
        embedding_latency_ms=embedding_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        fusion_latency_ms=fusion_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        embedding_calls=len(plan.search_queries),
    )


def retrieve_hybrid_v1_from_plan(
    question: str,
    plan: QueryPlan,
    history: list[dict[str, str]] | None = None,
    embed_query: Callable[[str], list[float]] | None = None,
) -> HybridRetrievalResult:
    """Execute the frozen hybrid_v1 profile from an already-created plan."""
    if len(plan.search_queries) != HYBRID_V1.atomic_queries:
        raise ValueError(
            f"{HYBRID_V1.name} requires exactly {HYBRID_V1.atomic_queries} queries."
        )
    embedding_function = embed_query or get_embeddings().embed_query
    ranked_results = []
    embedding_latency_ms = 0.0
    retrieval_latency_ms = 0.0
    for query in plan.search_queries:
        started = time.perf_counter()
        embedding = embedding_function(query)
        embedding_latency_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        candidates = vector_search_by_embedding(
            embedding,
            HYBRID_V1.candidates_per_query,
        )
        retrieval_latency_ms += (time.perf_counter() - started) * 1000
        ranked_results.append((query, candidates))

    return assemble_hybrid_v1(
        question,
        plan,
        ranked_results,
        history=history,
        embedding_latency_ms=embedding_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )


def plan_queries(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> QueryPlan:
    """Rewrite a contextual question and produce diverse retrieval queries."""
    planner = get_chat_model(PLANNER_MODEL).with_structured_output(
        QueryPlan,
        method="json_schema",
    )
    response = planner.invoke(
        build_planner_messages(question, history),
        automatic_function_calling={"disable": True},
    )

    queries = []
    for query in [response.standalone_question, *response.search_queries, question]:
        normalized = query.strip()
        if normalized and normalized.casefold() not in {
            existing.casefold() for existing in queries
        }:
            queries.append(normalized)

    if len(queries) < 3:
        raise ValueError("Query planner returned fewer than three unique search queries.")

    return QueryPlan(
        standalone_question=response.standalone_question.strip(),
        search_queries=queries[:DEFAULT_EXPANSION_COUNT],
    )


def build_planner_messages(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Build the identical provider-neutral query-decomposition prompt."""
    return [
        SystemMessage(
            content=(
                    "Rewrite the current question as a fully standalone Atlas Logistics "
                    "policy question. Then create exactly three semantically different "
                    "search queries. Each search query must target a separate atomic "
                    "information need instead of repeating the whole question. For a "
                    "compound scenario, separate operational escalation, customer "
                    "notification, and financial approval. If a monetary amount appears, "
                    "dedicate one query to that exact amount, the applicable approval "
                    "threshold, and the approver. Cover numeric thresholds, responsible "
                    "roles, timelines, and required actions. Do not answer the question. "
                    "Do not add facts."
            )
        ),
        HumanMessage(
            content=(
                    f"CONVERSATION:\n{format_history(history)}\n\n"
                    f"CURRENT QUESTION:\n{question.strip()}"
            )
        ),
    ]


def merge_ranked_results(
    ranked_results: list[tuple[str, list[Candidate]]],
    candidate_limit: int,
) -> list[Candidate]:
    """Deduplicate by stable chunk ID and fuse ranks with reciprocal rank fusion."""
    merged: dict[str, Candidate] = {}

    for query, candidates in ranked_results:
        for rank, candidate in enumerate(candidates, start=1):
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                existing = Candidate(
                    document=candidate.document,
                    distance=candidate.distance,
                )
                merged[candidate.chunk_id] = existing
            existing.distance = min(existing.distance, candidate.distance)
            if existing.best_vector_rank is None:
                existing.best_vector_rank = rank
            else:
                existing.best_vector_rank = min(existing.best_vector_rank, rank)
            existing.reciprocal_rank_score += 1 / (RRF_CONSTANT + rank)
            if query not in existing.matched_queries:
                existing.matched_queries.append(query)

    fused = sorted(
        merged.values(),
        key=lambda item: (-item.reciprocal_rank_score, item.distance, item.chunk_id),
    )[:candidate_limit]
    for rank, candidate in enumerate(fused, start=1):
        candidate.rrf_rank = rank
    return fused


def multi_query_candidates(
    plan: QueryPlan,
    per_query_k: int = 5,
    candidate_limit: int = 12,
) -> list[Candidate]:
    """Retrieve and fuse candidates for each expanded search query."""
    ranked_results = [
        (query, vector_search(query, per_query_k)) for query in plan.search_queries
    ]
    return merge_ranked_results(ranked_results, candidate_limit=candidate_limit)


def rerank_candidates(
    question: str,
    candidates: list[Candidate],
    top_k: int,
    history: list[dict[str, str]] | None = None,
    search_queries: list[str] | None = None,
    method: RerankMethod = "bm25",
) -> list[Candidate]:
    """Rerank locally with RRF, BM25, or a fixed inspectable hybrid score."""
    if not candidates:
        return []

    working = [candidate.clone() for candidate in candidates]

    contextual_query = " ".join(
        [
            *(message.get("content", "") for message in history or []),
            question,
            *(search_queries or []),
        ]
    )
    tokenized_candidates = [
        _tokenize(candidate.document.page_content) for candidate in working
    ]
    bm25 = BM25Okapi(tokenized_candidates)
    scores = bm25.get_scores(_tokenize(contextual_query))
    bm25_order = sorted(
        zip(working, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].distance, item[0].chunk_id),
    )
    for rank, (candidate, score) in enumerate(bm25_order, start=1):
        candidate.bm25_score = float(score)
        candidate.bm25_rank = rank

    if method == "rrf":
        reranked = sorted(
            working,
            key=lambda item: (
                -item.reciprocal_rank_score,
                item.distance,
                item.chunk_id,
            ),
        )
    elif method == "bm25":
        reranked = sorted(
            working,
            key=lambda item: (
                -item.bm25_score,
                -item.reciprocal_rank_score,
                item.distance,
                item.chunk_id,
            ),
        )
    elif method == "hybrid":
        rrf_values = [candidate.reciprocal_rank_score for candidate in working]
        bm25_values = [candidate.bm25_score for candidate in working]
        for candidate in working:
            normalized_rrf = _min_max(candidate.reciprocal_rank_score, rrf_values)
            normalized_bm25 = _min_max(candidate.bm25_score, bm25_values)
            candidate.hybrid_score = (
                0.50 * normalized_bm25
                + 0.30 * normalized_rrf
                + 0.20 * candidate.vector_similarity
            )
        reranked = sorted(
            working,
            key=lambda item: (-item.hybrid_score, item.distance, item.chunk_id),
        )
    else:
        raise ValueError(f"Unknown reranking method: {method}")

    selected = []
    for position, candidate in enumerate(reranked[:top_k], start=1):
        candidate.rerank_position = position
        selected.append(candidate)
    return selected


def _min_max(value: float, values: list[float]) -> float:
    """Normalize one score within the current candidate pool."""
    lower = min(values)
    upper = max(values)
    if upper == lower:
        return 1.0
    return (value - lower) / (upper - lower)


def _tokenize(text: str) -> list[str]:
    """Tokenize policy language while retaining numbers and currency terms."""
    return re.findall(r"[\w€]+(?:[.,]\d+)?", text.casefold())


def retrieve_evidence(
    question: str,
    strategy: RetrievalStrategy = "baseline",
    top_k: int = 5,
    history: list[dict[str, str]] | None = None,
    per_query_k: int = 5,
    candidate_limit: int = 12,
) -> RetrievalOutcome:
    """Execute one inspectable retrieval strategy end to end."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    valid_strategies = {
        "baseline",
        "rewrite",
        "multi_query",
        "multi_query_rerank",
        "hybrid_v1",
    }
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    rewrite_latency_ms = 0.0
    retrieval_latency_ms = 0.0
    rerank_latency_ms = 0.0
    embedding_latency_ms = 0.0
    fusion_latency_ms = 0.0
    embedding_calls = 0
    llm_calls = 0
    standalone_question = question.strip()
    search_queries = [standalone_question]

    if strategy == "baseline":
        started = time.perf_counter()
        evidence = vector_search(standalone_question, top_k)
        retrieval_latency_ms = (time.perf_counter() - started) * 1000
        embedding_calls = 1
        candidates = evidence
    else:
        started = time.perf_counter()
        plan = plan_queries(question, history=history)
        rewrite_latency_ms = (time.perf_counter() - started) * 1000
        llm_calls = 1
        standalone_question = plan.standalone_question

        if strategy == "hybrid_v1":
            search_queries = plan.search_queries
            hybrid = retrieve_hybrid_v1_from_plan(
                question,
                plan,
                history=history,
            )
            evidence = hybrid.evidence
            candidates = hybrid.candidate_pool
            embedding_latency_ms = hybrid.embedding_latency_ms
            retrieval_latency_ms = hybrid.retrieval_latency_ms
            fusion_latency_ms = hybrid.fusion_latency_ms
            rerank_latency_ms = hybrid.rerank_latency_ms
            embedding_calls = hybrid.embedding_calls
        elif strategy == "rewrite":
            search_queries = [standalone_question]
            started = time.perf_counter()
            evidence = vector_search(standalone_question, top_k)
            retrieval_latency_ms = (time.perf_counter() - started) * 1000
            embedding_calls = 1
            candidates = evidence
        else:
            search_queries = plan.search_queries
            started = time.perf_counter()
            candidates = multi_query_candidates(
                plan,
                per_query_k=per_query_k,
                candidate_limit=candidate_limit,
            )
            retrieval_latency_ms = (time.perf_counter() - started) * 1000
            embedding_calls = len(search_queries)
            evidence = candidates[:top_k]

            if strategy == "multi_query_rerank":
                started = time.perf_counter()
                evidence = rerank_candidates(
                    question,
                    candidates,
                    top_k=top_k,
                    history=history,
                    search_queries=search_queries,
                )
                rerank_latency_ms = (time.perf_counter() - started) * 1000

    diagnostics = RetrievalDiagnostics(
        strategy=strategy,
        original_question=question.strip(),
        standalone_question=standalone_question,
        search_queries=search_queries,
        candidate_count=len(candidates),
        embedding_calls=embedding_calls,
        llm_calls=llm_calls,
        rewrite_latency_ms=rewrite_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        embedding_latency_ms=embedding_latency_ms,
        fusion_latency_ms=fusion_latency_ms,
    )
    return RetrievalOutcome(evidence=evidence, diagnostics=diagnostics)
