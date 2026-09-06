"""Checkpoint 11 dataset, isolation, and reporting safety gates."""

from __future__ import annotations

import unittest
from collections import Counter

import httpx

from checkpoint11_dataset import CASES
from config import DATA_DIR, NVIDIA_API_KEY
from evaluation_checkpoint11 import CHUNK_SIZES, split_at_size
from nvidia_providers import NvidiaChat


class Checkpoint11Tests(unittest.TestCase):
    def test_dataset_has_exact_category_coverage(self) -> None:
        self.assertEqual(len(CASES), 50)
        self.assertEqual(
            Counter(item["category"] for item in CASES),
            {
                "direct": 8,
                "paraphrased": 7,
                "cross_document": 7,
                "ambiguous": 7,
                "unsupported": 8,
                "authorization": 6,
                "conversational": 7,
            },
        )

    def test_every_evidence_locator_exists_in_policy_source(self) -> None:
        for case in CASES:
            for locator in case["expected_evidence"]:
                text = (DATA_DIR / locator["source"]).read_text(encoding="utf-8")
                self.assertIn(locator["contains"].casefold(), text.casefold(), case["id"])

    def test_all_chunking_configurations_preserve_governance_metadata(self) -> None:
        for size in CHUNK_SIZES:
            for chunk in split_at_size(size):
                self.assertEqual(chunk.metadata["tenant_id"], "atlas-logistics")
                self.assertEqual(chunk.metadata["access_scope"], "employees")
                self.assertTrue(chunk.metadata["document_id"])
                self.assertTrue(chunk.metadata["chunk_id"])

    def test_nvidia_stream_accepts_usage_only_event(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"Grounded [1]."}}]}\n\n'
            'data: {"choices":[],"usage":{"total_tokens":9}}\n\n'
            'data: [DONE]\n\n'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        result = NvidiaChat(base_url="http://nim.local/v1", api_key="", client=client).complete_stream(
            [{"role": "user", "content": "question"}]
        )
        self.assertEqual(result.text, "Grounded [1].")
        self.assertEqual(result.usage["total_tokens"], 9)

    def test_final_outputs_do_not_contain_nvidia_key(self) -> None:
        if not NVIDIA_API_KEY:
            self.skipTest("No NVIDIA key configured.")
        for name in ("nt11_results.json", "checkpoint11_traces.json"):
            path = DATA_DIR.parent / "eval" / name
            if path.exists():
                self.assertNotIn(NVIDIA_API_KEY, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
