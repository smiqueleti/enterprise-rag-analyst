"""NVIDIA provider contracts and component-isolation tests without live calls."""

from __future__ import annotations

import json
import unittest

import httpx

from benchmark_profiles import PROFILES
from evaluation_checkpoint10 import ensure_separate_nvidia_collection
from nvidia_providers import (
    NvidiaChat,
    NvidiaConfigurationError,
    NvidiaEmbeddings,
    NvidiaReranker,
)


class NvidiaProviderTests(unittest.TestCase):
    def test_named_profiles_identify_every_component(self) -> None:
        expected = {
            "gemini_baseline", "nvidia_embeddings", "nvidia_reranker",
            "nvidia_generation", "nvidia_planner", "nvidia_full",
        }
        self.assertEqual(set(PROFILES), expected)
        for profile in PROFILES.values():
            self.assertTrue(profile.planner)
            self.assertTrue(profile.embedding_model)
            self.assertTrue(profile.retriever)
            self.assertTrue(profile.reranker)
            self.assertTrue(profile.generator)
            self.assertTrue(profile.model_serving_mechanism)

    def test_nvidia_collection_is_separate_and_model_bound(self) -> None:
        details = ensure_separate_nvidia_collection()
        self.assertTrue(details["separate_from_gemini"])
        self.assertIn("nvidia", details["collection"])
        self.assertTrue(details["embedding_model"])

    def test_embedding_adapter_preserves_query_and_passage_intent(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(
                200,
                json={"data": [{"index": index, "embedding": [float(index), 1.0]} for index, _ in enumerate(payload["input"])]},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        adapter = NvidiaEmbeddings(base_url="http://nim.local/v1", api_key="", client=client)
        self.assertEqual(len(adapter.embed_documents(["a", "b"])), 2)
        self.assertEqual(adapter.embed_query("q"), [0.0, 1.0])
        self.assertEqual(len(adapter.embed_queries(["q1", "q2"])), 2)
        self.assertEqual(
            [item["input_type"] for item in requests],
            ["passage", "query", "query"],
        )

    def test_reranker_adapter_preserves_provider_order(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(len(payload["passages"]), 2)
            return httpx.Response(
                200,
                json={"rankings": [{"index": 1, "logit": 3.2}, {"index": 0, "logit": 1.1}], "usage": {"prompt_tokens": 12}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        reranker = NvidiaReranker(
            url="http://nim.local/v1/ranking", api_key="",
            client=client,
        )
        ranking, usage = reranker.rank("query", ["first", "second"])
        self.assertEqual(ranking[0][0], 1)
        self.assertEqual(usage["prompt_tokens"], 12)

    def test_chat_adapter_normalizes_content_and_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Grounded [1]."}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        chat = NvidiaChat(
            base_url="http://nim.local/v1", api_key="",
            client=client,
        )
        result = chat.complete([{"role": "user", "content": "question"}])
        self.assertEqual(result.text, "Grounded [1].")
        self.assertEqual(result.usage["total_tokens"], 24)

    def test_hosted_api_requires_key_before_network_call(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
        self.addCleanup(client.close)
        adapter = NvidiaEmbeddings(
            base_url="https://integrate.api.nvidia.com/v1", api_key="", client=client
        )
        with self.assertRaises(NvidiaConfigurationError):
            adapter.embed_query("test")


if __name__ == "__main__":
    unittest.main()
