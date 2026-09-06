"""Evaluate Atlas retrieval strategies against exact chunk-level evidence labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from retrieval import (
    Candidate,
    QueryPlan,
    merge_ranked_results,
    plan_queries,
    rerank_candidates,
    vector_search,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "eval" / "questions.json"
DEFAULT_OUTPUT = PROJECT_DIR / "eval" / "results.json"
STRATEGIES = ["baseline", "rewrite", "multi_query", "multi_query_rerank"]
TOP_K = 5
PER_QUERY_K = 5
CANDIDATE_LIMIT = 12
REGRESSION_ID = "multi_01_financial_threshold_regression"
REGRESSION_TARGET = ("financial_approval_policy.md", 1)


def evidence_key(candidate: Candidate) -> tuple[str, int]:
    """Represent one retrieved chunk using the human-readable evaluation label."""
    metadata = candidate.document.metadata
    return str(metadata["source"]), int(metadata["chunk_index"])


def expected_keys(case: dict[str, Any]) -> list[tuple[str, int]]:
    """Read ordered expected evidence labels from one evaluation case."""
    return [
        (str(item["source"]), int(item["chunk_index"]))
        for item in case["expected_evidence"]
    ]


def validate_dataset(cases: list[dict[str, Any]]) -> None:
    """Reject incomplete or duplicate evaluation cases before API calls begin."""
    required_categories = {
        "single_document",
        "cross_document",
        "multi_policy",
        "ambiguous_followup",
        "unsupported",
    }
    if len(cases) < 15:
        raise ValueError("The evaluation dataset must contain at least 15 questions.")

    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique.")

    categories = {case.get("category") for case in cases}
    missing = required_categories - categories
    if missing:
        raise ValueError(f"Evaluation dataset is missing categories: {sorted(missing)}")

    for case in cases:
        for field in ["id", "category", "question", "history", "expected_evidence"]:
            if field not in case:
                raise ValueError(f"Case {case.get('id')} is missing `{field}`.")
        if case["category"] == "unsupported" and case["expected_evidence"]:
            raise ValueError(f"Unsupported case {case['id']} cannot have expected evidence.")
        if case["category"] != "unsupported" and not case["expected_evidence"]:
            raise ValueError(f"Supported case {case['id']} needs expected evidence.")


def rank_of(target: tuple[str, int], candidates: list[Candidate]) -> int | None:
    """Return the one-based rank of a target evidence chunk."""
    for rank, candidate in enumerate(candidates, start=1):
        if evidence_key(candidate) == target:
            return rank
    return None


def serialize_candidates(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Store only inspectable ranking facts, never API credentials or prompts."""
    return [
        {
            "rank": rank,
            "chunk_id": candidate.chunk_id,
            "source": candidate.document.metadata["source"],
            "chunk_index": candidate.document.metadata["chunk_index"],
            "distance": round(candidate.distance, 6),
            "rrf_score": round(candidate.reciprocal_rank_score, 8),
            "matched_query_count": len(candidate.matched_queries),
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def timed_vector_search(query: str, k: int) -> tuple[list[Candidate], float]:
    """Run one vector search and return wall-clock latency in milliseconds."""
    started = time.perf_counter()
    candidates = vector_search(query, k)
    return candidates, (time.perf_counter() - started) * 1000


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all four strategies while sharing identical intermediate artifacts."""
    question = str(case["question"])
    history = case["history"]

    baseline, baseline_latency = timed_vector_search(question, TOP_K)

    rewrite_started = time.perf_counter()
    plan = plan_queries(question, history=history)
    rewrite_latency = (time.perf_counter() - rewrite_started) * 1000

    rewritten, rewritten_vector_latency = timed_vector_search(
        plan.standalone_question,
        TOP_K,
    )

    ranked_results: list[tuple[str, list[Candidate]]] = []
    multi_vector_latency = 0.0
    for query in plan.search_queries:
        if query == plan.standalone_question:
            candidates = rewritten
            latency = rewritten_vector_latency
        else:
            candidates, latency = timed_vector_search(query, PER_QUERY_K)
        ranked_results.append((query, candidates))
        multi_vector_latency += latency

    candidate_pool = merge_ranked_results(
        ranked_results,
        candidate_limit=CANDIDATE_LIMIT,
    )

    rerank_started = time.perf_counter()
    reranked = rerank_candidates(
        question,
        candidate_pool,
        top_k=TOP_K,
        history=history,
        search_queries=plan.search_queries,
    )
    rerank_latency = (time.perf_counter() - rerank_started) * 1000

    results = {
        "baseline": {
            "evidence": serialize_candidates(baseline),
            "latency_ms": baseline_latency,
            "embedding_calls": 1,
            "llm_calls": 0,
        },
        "rewrite": {
            "evidence": serialize_candidates(rewritten),
            "latency_ms": rewrite_latency + rewritten_vector_latency,
            "embedding_calls": 1,
            "llm_calls": 1,
        },
        "multi_query": {
            "evidence": serialize_candidates(candidate_pool[:TOP_K]),
            "latency_ms": rewrite_latency + multi_vector_latency,
            "embedding_calls": len(plan.search_queries),
            "llm_calls": 1,
        },
        "multi_query_rerank": {
            "evidence": serialize_candidates(reranked),
            "latency_ms": rewrite_latency + multi_vector_latency + rerank_latency,
            "embedding_calls": len(plan.search_queries),
            "llm_calls": 1,
        },
    }

    regression = None
    if case["id"] == REGRESSION_ID:
        deep_baseline, _ = timed_vector_search(question, 16)
        regression = {
            "target": {
                "source": REGRESSION_TARGET[0],
                "chunk_index": REGRESSION_TARGET[1],
            },
            "baseline_rank": rank_of(REGRESSION_TARGET, deep_baseline),
            "multi_query_candidate_rank": rank_of(
                REGRESSION_TARGET,
                candidate_pool,
            ),
            "final_reranked_rank": rank_of(REGRESSION_TARGET, reranked),
            "in_final_evidence": rank_of(REGRESSION_TARGET, reranked) is not None,
        }

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "expected_evidence": case["expected_evidence"],
        "query_plan": {
            "standalone_question": plan.standalone_question,
            "search_queries": plan.search_queries,
        },
        "strategies": results,
        "regression": regression,
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a small evaluation sample."""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_strategy(
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    """Calculate chunk-level recall, full coverage, MRR, latency, and abstention."""
    recall_values = {1: [], 3: [], 5: []}
    full_recall_values = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    latencies = []
    unsupported_abstentions = []
    embedding_calls = []
    llm_calls = []
    category_recall_5: dict[str, list[float]] = {}

    case_map = {case["id"]: case for case in cases}
    for record in records:
        case = case_map[record["id"]]
        strategy_result = record["strategies"][strategy]
        ranked = [
            (item["source"], item["chunk_index"])
            for item in strategy_result["evidence"]
        ]
        expected = set(expected_keys(case))
        latencies.append(float(strategy_result["latency_ms"]))
        embedding_calls.append(int(strategy_result["embedding_calls"]))
        llm_calls.append(int(strategy_result["llm_calls"]))

        if not expected:
            unsupported_abstentions.append(1.0 if not ranked else 0.0)
            continue

        first_relevant_rank = next(
            (rank for rank, item in enumerate(ranked, start=1) if item in expected),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1 / first_relevant_rank)

        for k in [1, 3, 5]:
            hits = len(expected.intersection(ranked[:k]))
            recall = hits / len(expected)
            recall_values[k].append(recall)
            full_recall_values[k].append(1.0 if hits == len(expected) else 0.0)
            if k == 5:
                category_recall_5.setdefault(case["category"], []).append(recall)

    return {
        "recall_at_1": statistics.fmean(recall_values[1]),
        "recall_at_3": statistics.fmean(recall_values[3]),
        "recall_at_5": statistics.fmean(recall_values[5]),
        "full_evidence_at_1": statistics.fmean(full_recall_values[1]),
        "full_evidence_at_3": statistics.fmean(full_recall_values[3]),
        "full_evidence_at_5": statistics.fmean(full_recall_values[5]),
        "mrr": statistics.fmean(reciprocal_ranks),
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "unsupported_abstention_rate": (
            statistics.fmean(unsupported_abstentions)
            if unsupported_abstentions
            else 0.0
        ),
        "mean_embedding_calls": statistics.fmean(embedding_calls),
        "mean_llm_calls": statistics.fmean(llm_calls),
        "recall_at_5_by_category": {
            category: statistics.fmean(values)
            for category, values in sorted(category_recall_5.items())
        },
    }


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    """Print a compact comparison table suitable for terminal review."""
    print("\nRetrieval evaluation summary")
    print(
        "Strategy                 R@1    R@3    R@5    MRR    Full@5 "
        "Abstain  Mean ms  Calls E/L"
    )
    for strategy in STRATEGIES:
        metrics = summary[strategy]
        print(
            f"{strategy:<24} "
            f"{metrics['recall_at_1']:.3f}  "
            f"{metrics['recall_at_3']:.3f}  "
            f"{metrics['recall_at_5']:.3f}  "
            f"{metrics['mrr']:.3f}  "
            f"{metrics['full_evidence_at_5']:.3f}  "
            f"{metrics['unsupported_abstention_rate']:.3f}  "
            f"{metrics['mean_latency_ms']:.0f}  "
            f"{metrics['mean_embedding_calls']:.1f}/"
            f"{metrics['mean_llm_calls']:.1f}"
        )


def recompute_reranking(
    cases: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Recompute local reranking from saved plans without planner LLM calls."""
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    records_by_id = {record["id"]: record for record in saved["cases"]}

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] rerank {case['id']}")
        record = records_by_id[case["id"]]
        queries = record["query_plan"]["search_queries"]
        ranked_results = [
            (query, vector_search(query, PER_QUERY_K)) for query in queries
        ]
        candidate_pool = merge_ranked_results(
            ranked_results,
            candidate_limit=CANDIDATE_LIMIT,
        )
        started = time.perf_counter()
        reranked = rerank_candidates(
            case["question"],
            candidate_pool,
            top_k=TOP_K,
            history=case["history"],
            search_queries=queries,
        )
        rerank_latency = (time.perf_counter() - started) * 1000
        multi_result = record["strategies"]["multi_query"]
        record["strategies"]["multi_query_rerank"] = {
            "evidence": serialize_candidates(reranked),
            "latency_ms": multi_result["latency_ms"] + rerank_latency,
            "embedding_calls": len(queries),
            "llm_calls": 1,
        }

        if case["id"] == REGRESSION_ID:
            record["regression"]["multi_query_candidate_rank"] = rank_of(
                REGRESSION_TARGET,
                candidate_pool,
            )
            final_rank = rank_of(REGRESSION_TARGET, reranked)
            record["regression"]["final_reranked_rank"] = final_rank
            record["regression"]["in_final_evidence"] = final_rank is not None

    records = [records_by_id[case["id"]] for case in cases]
    saved["summary"] = {
        strategy: summarize_strategy(cases, records, strategy)
        for strategy in STRATEGIES
    }
    saved["cases"] = records
    saved["regression"] = next(
        (record["regression"] for record in records if record["regression"]),
        None,
    )
    output_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--rerank-existing",
        action="store_true",
        help="Reuse saved query plans and recompute only multi-query reranking.",
    )
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    validate_dataset(cases)
    if args.limit:
        cases = cases[: args.limit]

    if args.rerank_existing:
        output = recompute_reranking(cases, args.output)
        print_summary(output["summary"])
        print(f"\nFinancial threshold regression: {output['regression']}")
        print(f"\nDetailed results written to {args.output}")
        return

    records = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        records.append(evaluate_case(case))

    summary = {
        strategy: summarize_strategy(cases, records, strategy)
        for strategy in STRATEGIES
    }
    regression = next(
        (record["regression"] for record in records if record["regression"]),
        None,
    )
    output = {
        "configuration": {
            "questions": len(cases),
            "top_k": TOP_K,
            "per_query_k": PER_QUERY_K,
            "candidate_limit": CANDIDATE_LIMIT,
            "strategies": STRATEGIES,
        },
        "summary": summary,
        "regression": regression,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    if regression:
        print(f"\nFinancial threshold regression: {regression}")
    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
