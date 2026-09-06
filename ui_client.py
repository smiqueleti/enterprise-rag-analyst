"""HTTP client and payload helpers for the Streamlit presentation layer."""

from __future__ import annotations

import os
from typing import Any

import httpx


DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def build_query_payload(
    question: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stable API request from prior chat turns only."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-20:]
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    return {
        "question": question.strip(),
        "conversation_id": conversation_id,
        "history": history,
    }


def query_api(
    payload: dict[str, Any],
    api_url: str = DEFAULT_API_URL,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """Call the RAG API and return a validated JSON object."""
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{api_url.rstrip('/')}/query", json=payload)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict) or "answer" not in result:
        raise ValueError("The RAG API returned a malformed response.")
    return result
