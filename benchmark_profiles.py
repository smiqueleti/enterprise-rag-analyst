"""Named component configurations for isolated Checkpoint 10 comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from config import (
    EMBEDDING_MODEL,
    NVIDIA_CHAT_MODEL,
    NVIDIA_EMBEDDING_MODEL,
    NVIDIA_PLANNER_MODEL,
    NVIDIA_RERANK_MODEL,
    PLANNER_MODEL,
    PRIMARY_CHAT_MODEL,
)


@dataclass(frozen=True)
class ComponentProfile:
    name: str
    planner: str
    embedding_model: str
    retriever: str
    reranker: str
    generator: str
    model_serving_mechanism: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PROFILES = {
    "gemini_baseline": ComponentProfile(
        "gemini_baseline", PLANNER_MODEL, EMBEDDING_MODEL, "Chroma hybrid_v1",
        "local RRF + BM25 + vector", PRIMARY_CHAT_MODEL, "Google hosted API",
    ),
    "nvidia_embeddings": ComponentProfile(
        "nvidia_embeddings", PLANNER_MODEL, NVIDIA_EMBEDDING_MODEL,
        "separate NVIDIA Chroma collection with hybrid_v1",
        "local RRF + BM25 + vector", PRIMARY_CHAT_MODEL,
        "NVIDIA hosted API or OpenAI-compatible embedding NIM",
    ),
    "nvidia_reranker": ComponentProfile(
        "nvidia_reranker", PLANNER_MODEL, EMBEDDING_MODEL,
        "unchanged Gemini candidate generation", NVIDIA_RERANK_MODEL,
        PRIMARY_CHAT_MODEL, "NVIDIA hosted retrieval API or reranking NIM",
    ),
    "nvidia_generation": ComponentProfile(
        "nvidia_generation", PLANNER_MODEL, EMBEDDING_MODEL, "Chroma hybrid_v1",
        "local RRF + BM25 + vector", NVIDIA_CHAT_MODEL,
        "NVIDIA hosted API or OpenAI-compatible LLM NIM",
    ),
    "nvidia_planner": ComponentProfile(
        "nvidia_planner", NVIDIA_PLANNER_MODEL, EMBEDDING_MODEL,
        "Chroma hybrid_v1", "local RRF + BM25 + vector", PRIMARY_CHAT_MODEL,
        "NVIDIA hosted API or OpenAI-compatible LLM NIM",
    ),
    "nvidia_full": ComponentProfile(
        "nvidia_full", NVIDIA_PLANNER_MODEL, NVIDIA_EMBEDDING_MODEL,
        "separate NVIDIA Chroma collection with hybrid_v1", NVIDIA_RERANK_MODEL,
        NVIDIA_CHAT_MODEL, "NVIDIA hosted APIs or independently deployed NIMs",
    ),
}
