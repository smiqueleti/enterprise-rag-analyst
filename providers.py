"""Provider factories for embedding and chat models."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from config import EMBEDDING_MODEL, GENERATION_MODEL, PROJECT_DIR, REQUEST_TIMEOUT_SECONDS
from nvidia_providers import NvidiaChat, NvidiaEmbeddings, NvidiaReranker


def load_environment() -> None:
    """Load local credentials without exposing them to callers."""
    load_dotenv(PROJECT_DIR / ".env")


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the replaceable embedding component."""
    load_environment()
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=4)
def get_chat_model(model: str = GENERATION_MODEL) -> ChatGoogleGenerativeAI:
    """Create a replaceable Gemini chat-model component."""
    load_environment()
    return ChatGoogleGenerativeAI(
        model=model,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


@lru_cache(maxsize=2)
def get_nvidia_embeddings() -> NvidiaEmbeddings:
    """Create an NVIDIA hosted-API or NIM embedding adapter."""
    load_environment()
    return NvidiaEmbeddings()


@lru_cache(maxsize=2)
def get_nvidia_reranker() -> NvidiaReranker:
    """Create an NVIDIA hosted-API or NIM neural reranker."""
    load_environment()
    return NvidiaReranker()


@lru_cache(maxsize=4)
def get_nvidia_chat_model(model: str) -> NvidiaChat:
    """Create an OpenAI-compatible NVIDIA hosted-API or LLM NIM client."""
    load_environment()
    return NvidiaChat(model=model)
