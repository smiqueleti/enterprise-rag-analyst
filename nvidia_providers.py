"""Small replaceable clients for NVIDIA hosted APIs and self-hosted NIMs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_CHAT_MODEL,
    NVIDIA_EMBEDDING_MODEL, NVIDIA_RERANK_MODEL, NVIDIA_RERANK_URL,
    NVIDIA_TIMEOUT_SECONDS,
)


class NvidiaConfigurationError(RuntimeError):
    pass


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _hosted_key_required(url: str) -> bool:
    return "nvidia.com" in url.casefold()


@dataclass(frozen=True)
class NvidiaCompletion:
    text: str
    model: str
    usage: dict[str, int]
    latency_ms: float
    ttft_ms: float | None = None


class NvidiaEmbeddings:
    """LangChain-compatible embedding adapter with query/passage intent."""

    def __init__(
        self,
        model: str = NVIDIA_EMBEDDING_MODEL,
        base_url: str = NVIDIA_BASE_URL,
        api_key: str = NVIDIA_API_KEY,
        timeout: float = NVIDIA_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not self.api_key and _hosted_key_required(self.base_url):
            raise NvidiaConfigurationError("NVIDIA_API_KEY is required for the hosted embedding API.")
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=_headers(self.api_key),
            json={
                "model": self.model, "input": texts, "input_type": input_type,
                "encoding_format": "float", "truncate": "END",
            },
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Batch query embeddings without changing retrieval intent."""
        return self._embed(texts, "query")


class NvidiaReranker:
    def __init__(
        self,
        model: str = NVIDIA_RERANK_MODEL,
        url: str = NVIDIA_RERANK_URL,
        api_key: str = NVIDIA_API_KEY,
        timeout: float = NVIDIA_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.url = url
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout)

    def rank(self, query: str, passages: list[str]) -> tuple[list[tuple[int, float]], dict[str, int]]:
        if not self.api_key and _hosted_key_required(self.url):
            raise NvidiaConfigurationError("NVIDIA_API_KEY is required for the hosted reranking API.")
        response = self.client.post(
            self.url,
            headers=_headers(self.api_key),
            json={
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": passage} for passage in passages],
                "truncate": "END",
            },
        )
        response.raise_for_status()
        payload = response.json()
        rankings = [(int(item["index"]), float(item["logit"])) for item in payload["rankings"]]
        usage = {str(key): int(value) for key, value in payload.get("usage", {}).items() if isinstance(value, int)}
        return rankings, usage


class NvidiaChat:
    def __init__(
        self,
        model: str = NVIDIA_CHAT_MODEL,
        base_url: str = NVIDIA_BASE_URL,
        api_key: str = NVIDIA_API_KEY,
        timeout: float = NVIDIA_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0) -> NvidiaCompletion:
        if not self.api_key and _hosted_key_required(self.base_url):
            raise NvidiaConfigurationError("NVIDIA_API_KEY is required for the hosted chat API.")
        started = time.perf_counter()
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=_headers(self.api_key),
            json={"model": self.model, "messages": messages, "temperature": temperature, "stream": False},
        )
        response.raise_for_status()
        payload = response.json()
        usage = {str(key): int(value) for key, value in payload.get("usage", {}).items() if isinstance(value, int)}
        return NvidiaCompletion(
            payload["choices"][0]["message"]["content"].strip(), self.model,
            usage, (time.perf_counter() - started) * 1000,
        )

    def complete_stream(self, messages: list[dict[str, str]], temperature: float = 0) -> NvidiaCompletion:
        if not self.api_key and _hosted_key_required(self.base_url):
            raise NvidiaConfigurationError("NVIDIA_API_KEY is required for the hosted chat API.")
        started = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage: dict[str, int] = {}
        with self.client.stream(
            "POST", f"{self.base_url}/chat/completions",
            headers={**_headers(self.api_key), "Accept": "text/event-stream"},
            json={"model": self.model, "messages": messages, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ")
                if raw == "[DONE]":
                    break
                event = json.loads(raw)
                if event.get("usage"):
                    usage = {str(key): int(value) for key, value in event["usage"].items() if isinstance(value, int)}
                choices = event.get("choices") or []
                delta = (
                    choices[0].get("delta", {}).get("content")
                    if choices
                    else None
                )
                if delta:
                    if ttft is None:
                        ttft = (time.perf_counter() - started) * 1000
                    parts.append(delta)
        return NvidiaCompletion("".join(parts).strip(), self.model, usage, (time.perf_counter() - started) * 1000, ttft)
