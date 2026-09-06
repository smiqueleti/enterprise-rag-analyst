"""Central configuration for replaceable Atlas RAG components."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = PROJECT_DIR / "data"
VECTOR_DIR = PROJECT_DIR / "vectorstore"
NVIDIA_VECTOR_DIR = PROJECT_DIR / "vectorstore_nvidia"
TRACE_DIR = PROJECT_DIR / "runtime_traces"
TELEMETRY_DB = TRACE_DIR / "telemetry.sqlite3"
COLLECTION_NAME = "atlas_logistics"

EMBEDDING_MODEL = "gemini-embedding-001"
PRIMARY_CHAT_MODEL = os.getenv("PRIMARY_CHAT_MODEL", "gemini-3.5-flash")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemini-3.1-flash-lite")
GENERATION_MODEL = PRIMARY_CHAT_MODEL
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gemini-3.6-flash")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
REQUEST_DEADLINE_SECONDS = float(
    os.getenv("REQUEST_DEADLINE_SECONDS", str(REQUEST_TIMEOUT_SECONDS))
)
MAX_PLANNER_CONCURRENCY = int(os.getenv("MAX_PLANNER_CONCURRENCY", "2"))
MAX_EMBEDDING_CONCURRENCY = int(os.getenv("MAX_EMBEDDING_CONCURRENCY", "4"))
MAX_GENERATION_CONCURRENCY = int(os.getenv("MAX_GENERATION_CONCURRENCY", "2"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "12"))
QUEUE_TIMEOUT_SECONDS = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "10"))
MIN_PROVIDER_CALL_BUDGET_SECONDS = float(
    os.getenv("MIN_PROVIDER_CALL_BUDGET_SECONDS", "1")
)
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
CIRCUIT_RECOVERY_SECONDS = float(os.getenv("CIRCUIT_RECOVERY_SECONDS", "30"))
CIRCUIT_HALF_OPEN_MAX_CALLS = int(os.getenv("CIRCUIT_HALF_OPEN_MAX_CALLS", "1"))
RETRY_AFTER_SECONDS = int(os.getenv("RETRY_AFTER_SECONDS", "5"))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"
)
NVIDIA_RERANK_MODEL = os.getenv(
    "NVIDIA_RERANK_MODEL", "nvidia/llama-nemotron-rerank-vl-1b-v2"
)
NVIDIA_RERANK_URL = os.getenv(
    "NVIDIA_RERANK_URL",
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking",
)
NVIDIA_CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_PLANNER_MODEL = os.getenv("NVIDIA_PLANNER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NVIDIA_TIMEOUT_SECONDS = float(os.getenv("NVIDIA_TIMEOUT_SECONDS", "90"))
PROVIDER_MAX_ATTEMPTS = 2
PROVIDER_BACKOFF_SECONDS = 0.25


@dataclass(frozen=True)
class RetrievalProfile:
    """Named, immutable retrieval configuration."""

    name: str
    atomic_queries: int
    candidates_per_query: int
    candidate_pool: int
    reranker: str
    final_evidence: int
    confidence_threshold: float | None


HYBRID_V1 = RetrievalProfile(
    name="hybrid_v1",
    atomic_queries=4,
    candidates_per_query=5,
    candidate_pool=12,
    reranker="hybrid",
    final_evidence=5,
    confidence_threshold=None,
)
