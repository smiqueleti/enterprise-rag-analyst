"""Stable FastAPI request and response contract for Atlas RAG."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class AuthorizationContext(BaseModel):
    tenant_id: str = Field(default="atlas-logistics", min_length=1, max_length=200)
    access_scopes: list[str] = Field(default_factory=lambda: ["employees"], min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = Field(default=None, max_length=200)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)
    authorization: AuthorizationContext = Field(default_factory=AuthorizationContext)


class CitationItem(BaseModel):
    number: int
    chunk_id: str
    source: str


class SourceItem(BaseModel):
    citation: int
    chunk_id: str
    document: str
    document_id: str
    department: str
    document_type: str
    version: str
    effective_date: str
    tenant_id: str | None = None
    access_scope: str | None = None
    chunk_index: int
    vector_similarity: float
    rrf_score: float
    bm25_score: float
    hybrid_score: float
    text: str


class LatencyBreakdown(BaseModel):
    planning: float
    embedding: float
    retrieval: float
    reranking: float
    generation: float
    total: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    sources: list[SourceItem]
    refused: bool
    retrieval_strategy: Literal["hybrid_v1"]
    latency_ms: LatencyBreakdown
    trace_id: str
    model_used: str
    token_usage: dict[str, int]
    citation_validation: dict[str, Any]
    debug: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    retrieval_strategy: Literal["hybrid_v1"]
    vector_chunks: int
    primary_model: str
    fallback_model: str
