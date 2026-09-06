"""Local unit tests for retrieval invariants that do not call external APIs."""

from __future__ import annotations

import json
import unittest

from langchain_core.documents import Document

from answer_evaluation import evaluate_answer
from confidence import analyze_confidence
from evaluation import DEFAULT_DATASET, validate_dataset
from rag import INSUFFICIENT_ANSWER, generate_answer
from retrieval import Candidate, merge_ranked_results, rerank_candidates


def candidate(chunk_id: str, text: str, distance: float = 0.5) -> Candidate:
    """Create a minimal candidate for local unit tests."""
    return Candidate(
        document=Document(id=chunk_id, page_content=text, metadata={}),
        distance=distance,
    )


class PipelineTests(unittest.TestCase):
    def test_dataset_has_required_cases(self) -> None:
        cases = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        validate_dataset(cases)
        self.assertGreaterEqual(len(cases), 15)

    def test_multi_query_merge_deduplicates_stable_ids(self) -> None:
        shared = candidate("stable-1", "shipment escalation", 0.4)
        duplicate = candidate("stable-1", "shipment escalation", 0.3)
        other = candidate("stable-2", "customer notification", 0.2)

        merged = merge_ranked_results(
            [("query one", [shared, other]), ("query two", [duplicate])],
            candidate_limit=5,
        )

        self.assertEqual([item.chunk_id for item in merged].count("stable-1"), 1)
        self.assertEqual(merged[0].distance, 0.3)
        self.assertEqual(len(merged[0].matched_queries), 2)

    def test_reranker_promotes_financial_threshold(self) -> None:
        candidates = [
            candidate("general", "general shipment operating process"),
            candidate(
                "threshold",
                "contractual penalty approval threshold EUR 1,000 Regional Operations Manager",
            ),
            candidate("customer", "customer communication and notification records"),
            candidate("carrier", "carrier route and dispatch confirmation"),
        ]

        reranked = rerank_candidates(
            "Who approves a contractual penalty above EUR 1,000?",
            candidates,
            top_k=4,
        )

        self.assertEqual(reranked[0].chunk_id, "threshold")

    def test_no_evidence_returns_exact_refusal_without_api_call(self) -> None:
        answer = generate_answer("How many trucks are there?", [])
        self.assertEqual(answer, INSUFFICIENT_ANSWER)

    def test_confidence_captures_source_diversity_and_rank_agreement(self) -> None:
        first = candidate("one", "shipment escalation", 0.2)
        first.document.metadata["source"] = "a.md"
        second = candidate("two", "customer notification", 0.4)
        second.document.metadata["source"] = "b.md"
        first.bm25_score = 2.0
        second.bm25_score = 1.0

        signals = analyze_confidence([first, second], [first, second])

        self.assertEqual(signals.unique_source_documents, 2)
        self.assertGreater(signals.top1_vector_similarity, signals.average_topk_similarity)
        self.assertEqual(signals.vector_reranker_rank_agreement, 1.0)

    def test_citation_entailment_accepts_question_value_and_policy_support(self) -> None:
        evidence = candidate(
            "policy",
            "Potential contractual penalties above EUR 1,000 require approval from the Regional Operations Manager.",
        )
        evidence.document.metadata["source"] = "financial_approval_policy.md"
        result = evaluate_answer(
            "A EUR 1,500 penalty requires Regional Operations Manager approval [1].",
            [evidence],
            {
                "required_terms": [["EUR 1,500"], ["Regional Operations Manager"]],
                "required_sources": ["financial_approval_policy.md"],
            },
            unsupported=False,
            question="What approval is required for a EUR 1,500 penalty?",
        )

        self.assertTrue(result["answer_correct"])
        self.assertEqual(result["citation_entailment"], 1.0)


if __name__ == "__main__":
    unittest.main()
