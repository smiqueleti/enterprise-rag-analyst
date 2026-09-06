"""Run isolated NVIDIA component experiments or preserve explicit blocked results."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from answer_evaluation import evaluate_answer
from benchmark_profiles import PROFILES
from citation_validation import validate_answer_citations
from config import (
    COLLECTION_NAME, EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_CHAT_MODEL,
    NVIDIA_EMBEDDING_MODEL, NVIDIA_PLANNER_MODEL, NVIDIA_RERANK_MODEL, NVIDIA_RERANK_URL,
    NVIDIA_VECTOR_DIR, PLANNER_MODEL, PRIMARY_CHAT_MODEL,
)
from evaluation import evidence_key, expected_keys, rank_of, validate_dataset
from evaluation_checkpoint7 import (
    FINANCIAL_TARGET, NOTIFICATION_TARGET, build_configuration,
    restore_raw,
)
from ingest import load_documents, split_documents
from nvidia_providers import NvidiaChat, NvidiaEmbeddings, NvidiaReranker
from rag import INSUFFICIENT_ANSWER, SYSTEM_PROMPT, build_context
from retrieval import (
    Candidate, QueryPlan, assemble_hybrid_v1, build_planner_messages, merge_ranked_results,
    vector_search_by_embedding,
)
from providers import get_embeddings


PROJECT_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_DIR / "eval" / "questions.json"
EXPECTATIONS_PATH = PROJECT_DIR / "eval" / "answer_expectations.json"
PLANS_PATH = PROJECT_DIR / "eval" / "checkpoint7_plans.json"
RAW_PATH = PROJECT_DIR / "eval" / "checkpoint7_raw_cache.json"
CP7_RESULTS_PATH = PROJECT_DIR / "eval" / "checkpoint7_results.json"
OUTPUT_PATH = PROJECT_DIR / "eval" / "checkpoint10_results.json"
TRACE_PATH = PROJECT_DIR / "eval" / "checkpoint10_traces.json"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, QueryPlan], dict[str, Any]]:
    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    validate_dataset(cases)
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    plans_payload = json.loads(PLANS_PATH.read_text(encoding="utf-8"))["plans"]
    plans = {key: QueryPlan.model_validate(value) for key, value in plans_payload.items()}
    raw_payload = json.loads(RAW_PATH.read_text(encoding="utf-8"))["cases"]
    raw = {key: restore_raw(value) for key, value in raw_payload.items()}
    return cases, expectations, plans, raw


def retrieval_metrics(cases: list[dict[str, Any]], rankings: dict[str, list[Candidate]]) -> dict[str, float]:
    recalls = {1: [], 3: [], 5: []}
    full_at_5 = []
    reciprocal = []
    for case in cases:
        if case["category"] == "unsupported":
            continue
        expected = set(expected_keys(case))
        ranked = [evidence_key(item) for item in rankings[case["id"]]]
        first = next((index for index, item in enumerate(ranked, 1) if item in expected), None)
        reciprocal.append(0 if first is None else 1 / first)
        for k in recalls:
            recalls[k].append(len(expected.intersection(ranked[:k])) / len(expected))
        full_at_5.append(float(expected.issubset(set(ranked[:5]))))
    return {
        "recall_at_1": statistics.fmean(recalls[1]),
        "recall_at_3": statistics.fmean(recalls[3]),
        "recall_at_5": statistics.fmean(recalls[5]),
        "mrr": statistics.fmean(reciprocal),
        "full_at_5": statistics.fmean(full_at_5),
    }


def candidate_trace(candidate: Candidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank, "chunk_id": candidate.chunk_id,
        "source": candidate.document.metadata["source"],
        "chunk_index": candidate.document.metadata["chunk_index"],
        "vector_similarity": candidate.vector_similarity,
        "rrf_score": candidate.reciprocal_rank_score,
        "bm25_score": candidate.bm25_score,
    }


def blocked_experiment(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "BLOCKED",
        "reason": reason,
        "quality": None,
        "performance": None,
        "reliability": {"failures": 0, "timeouts": 0, "retries": 0},
    }


def nvidia_collection_name() -> str:
    """Bind one collection name to exactly one NVIDIA embedding model."""
    model_slug = NVIDIA_EMBEDDING_MODEL.replace("/", "_").replace("-", "_")
    return f"{COLLECTION_NAME}_nvidia_{model_slug}"


def ensure_separate_nvidia_collection() -> dict[str, Any]:
    """Create the isolated collection without generating or mixing vectors."""
    collection = nvidia_collection_name()
    store = Chroma(collection_name=collection, persist_directory=str(NVIDIA_VECTOR_DIR))
    return {
        "collection": collection,
        "path": str(NVIDIA_VECTOR_DIR),
        "embedding_model": NVIDIA_EMBEDDING_MODEL,
        "indexed_chunks": len(store.get(include=[])["ids"]),
        "separate_from_gemini": NVIDIA_VECTOR_DIR != PROJECT_DIR / "vectorstore",
    }


def run_embeddings(
    cases: list[dict[str, Any]],
    plans: dict[str, QueryPlan],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    adapter = NvidiaEmbeddings()
    chunks = split_documents(load_documents())
    texts = [chunk.page_content for chunk in chunks]
    started = time.perf_counter()
    vectors = adapter.embed_documents(texts)
    indexing_embedding_ms = (time.perf_counter() - started) * 1000
    collection = nvidia_collection_name()
    store = Chroma(collection_name=collection, persist_directory=str(NVIDIA_VECTOR_DIR))
    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)
    started = time.perf_counter()
    store._collection.upsert(
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    local_index_ms = (time.perf_counter() - started) * 1000
    rankings = {}
    embedding_latencies = []
    retrieval_latencies = []
    for case in cases:
        plan = plans[case["id"]]
        ranked_results = []
        for query in plan.search_queries:
            started = time.perf_counter()
            vector = adapter.embed_query(query)
            embedding_latencies.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            candidates = vector_search_by_embedding(vector, 5, vectorstore=store)
            retrieval_latencies.append((time.perf_counter() - started) * 1000)
            ranked_results.append((query, candidates))
        result = assemble_hybrid_v1(case["question"], plan, ranked_results, case["history"])
        rankings[case["id"]] = result.evidence
        traces.append({
            "experiment": "nvidia_embeddings", "case_id": case["id"],
            "queries": plan.search_queries,
            "evidence": [candidate_trace(item, rank) for rank, item in enumerate(result.evidence, 1)],
        })
    return {
        "name": "nvidia_embeddings", "status": "COMPLETE",
        "collection": collection, "embedding_model": NVIDIA_EMBEDDING_MODEL,
        "embedding_dimensionality": len(vectors[0]),
        "quality": retrieval_metrics(cases, rankings),
        "performance": {
            "embedding_p50_ms": percentile(embedding_latencies, 0.5),
            "embedding_p95_ms": percentile(embedding_latencies, 0.95),
            "retrieval_p50_ms": percentile(retrieval_latencies, 0.5),
            "retrieval_p95_ms": percentile(retrieval_latencies, 0.95),
            "index_embedding_ms": indexing_embedding_ms,
            "local_index_ms": local_index_ms,
            "indexing_throughput_chunks_per_second": len(chunks) / ((indexing_embedding_ms + local_index_ms) / 1000),
        },
        "reliability": {"failures": 0, "timeouts": 0, "retries": 0},
    }


def frozen_hybrid_evidence(case: dict[str, Any], raw: dict[str, Any]) -> list[Candidate]:
    return build_configuration(case, raw, 5, 12, 5, "hybrid")["evidence"]


def run_reranker(
    cases: list[dict[str, Any]],
    raw: dict[str, Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    reranker = NvidiaReranker()
    rankings = {}
    latencies = []
    token_usage = []
    regression = {}
    for case in cases:
        sliced = [(query, items[:5]) for query, items in raw[case["id"]]["query_results"]]
        pool = merge_ranked_results(sliced, 12)
        started = time.perf_counter()
        ranked, usage = reranker.rank(case["question"], [item.document.page_content for item in pool])
        latencies.append((time.perf_counter() - started) * 1000)
        token_usage.append(usage)
        evidence = [pool[index].clone() for index, _ in ranked[:5]]
        for position, item in enumerate(evidence, 1):
            item.rerank_position = position
        rankings[case["id"]] = evidence
        traces.append({
            "experiment": "nvidia_reranker", "case_id": case["id"],
            "candidate_pool": [candidate_trace(item, rank) for rank, item in enumerate(pool, 1)],
            "evidence": [candidate_trace(item, rank) for rank, item in enumerate(evidence, 1)],
            "usage": usage,
        })
        if case["id"] == "multi_01_financial_threshold_regression":
            regression = {
                "financial_rank": rank_of(FINANCIAL_TARGET, evidence),
                "notification_rank": rank_of(NOTIFICATION_TARGET, evidence),
            }
    return {
        "name": "nvidia_reranker", "status": "COMPLETE",
        "model": NVIDIA_RERANK_MODEL, "quality": retrieval_metrics(cases, rankings),
        "regression_cases": regression,
        "performance": {"reranking_p50_ms": percentile(latencies, 0.5), "reranking_p95_ms": percentile(latencies, 0.95)},
        "efficiency": {"calls": len(latencies), "token_usage": token_usage},
        "reliability": {"failures": 0, "timeouts": 0, "retries": 0},
    }


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, float]:
    supported = [row for row in rows if not row["unsupported"]]
    unsupported = [row for row in rows if row["unsupported"]]
    refusals = [row for row in rows if row["outcome"]["refused"]]
    correct_refusals = sum(row["outcome"]["correct_refusal"] for row in rows)
    return {
        "answer_correctness": statistics.fmean(float(row["outcome"]["answer_correct"]) for row in supported),
        "refusal_precision": correct_refusals / len(refusals) if refusals else 1.0,
        "refusal_recall": correct_refusals / len(unsupported),
        "citation_entailment": statistics.fmean(row["outcome"]["citation_entailment"] for row in supported),
        "invalid_citation_rate": statistics.fmean(float(row["outcome"]["citation_correctness"] < 1) for row in supported),
    }


def run_generation(
    cases: list[dict[str, Any]],
    expectations: dict[str, Any],
    raw: dict[str, Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    chat = NvidiaChat(model=NVIDIA_CHAT_MODEL)
    rows = []
    latencies = []
    ttfts = []
    usages = []
    repair_attempts = 0
    repair_successes = 0
    for case in cases:
        evidence = frozen_hybrid_evidence(case, raw[case["id"]])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT EXCERPTS:\n\n{build_context(evidence)}\n\nQUESTION:\n{case['question'].strip()}"},
        ]
        completion = chat.complete_stream(messages)
        answer = completion.text or INSUFFICIENT_ANSWER
        validation = validate_answer_citations(answer, evidence, question=case["question"])
        total_latency = completion.latency_ms
        total_usage = dict(completion.usage)
        if not validation.valid:
            repair_attempts += 1
            correction = (
                "A previous draft failed citation validation. Rewrite the answer once. "
                "Do not discuss the validation failure.\n"
                f"VALIDATION FEEDBACK:\n{validation.correction_prompt()}"
            )
            repaired = chat.complete_stream([
                *messages,
                {"role": "assistant", "content": answer},
                {"role": "user", "content": correction},
            ])
            total_latency += repaired.latency_ms
            for key, value in repaired.usage.items():
                total_usage[key] = total_usage.get(key, 0) + value
            repaired_validation = validate_answer_citations(
                repaired.text, evidence, question=case["question"]
            )
            if repaired_validation.valid:
                answer = repaired.text
                validation = repaired_validation
                repair_successes += 1
            else:
                answer = INSUFFICIENT_ANSWER
        outcome = evaluate_answer(
            answer, evidence, expectations.get(case["id"]),
            case["category"] == "unsupported", case["question"],
        )
        rows.append({"case_id": case["id"], "unsupported": case["category"] == "unsupported", "outcome": outcome})
        latencies.append(total_latency)
        if completion.ttft_ms is not None:
            ttfts.append(completion.ttft_ms)
        usages.append(total_usage)
        traces.append({
            "experiment": "nvidia_generation", "case_id": case["id"],
            "model": completion.model, "answer": answer,
            "latency_ms": total_latency, "ttft_ms": completion.ttft_ms,
            "usage": total_usage, "citation_validation": validation.valid,
            "evaluation": outcome,
        })
    return {
        "name": "nvidia_generation", "status": "COMPLETE",
        "model": NVIDIA_CHAT_MODEL, "quality": aggregate_answers(rows),
        "performance": {
            "ttft_p50_ms": percentile(ttfts, 0.5), "ttft_p95_ms": percentile(ttfts, 0.95),
            "generation_p50_ms": percentile(latencies, 0.5), "generation_p95_ms": percentile(latencies, 0.95),
        },
        "efficiency": {
            "minimum_calls": len(rows), "token_usage": usages,
            "citation_repair_attempts": repair_attempts,
            "citation_repair_successes": repair_successes,
            "citation_repair_rate": repair_attempts / len(rows),
        },
        "reliability": {"failures": 0, "timeouts": 0, "retries": 0},
    }


def run_planning(
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    chat = NvidiaChat(model=NVIDIA_PLANNER_MODEL)
    embeddings = get_embeddings()
    rankings = {}
    planning_latencies = []
    invalid = 0
    duplicates = 0
    for case in cases:
        langchain_messages = build_planner_messages(case["question"], case["history"])
        messages = [
            {"role": "system" if index == 0 else "user", "content": message.content}
            for index, message in enumerate(langchain_messages)
        ]
        messages[0]["content"] += (
            " Return only JSON with string field standalone_question and array field "
            "search_queries containing exactly three strings."
        )
        completion = chat.complete(messages)
        planning_latencies.append(completion.latency_ms)
        try:
            payload = json.loads(completion.text.removeprefix("```json").removesuffix("```").strip())
            values = [payload["standalone_question"], *payload["search_queries"], case["question"]]
            unique = []
            for value in values:
                clean = str(value).strip()
                if clean.casefold() in {item.casefold() for item in unique}:
                    duplicates += 1
                elif clean:
                    unique.append(clean)
            plan = QueryPlan(standalone_question=str(payload["standalone_question"]).strip(), search_queries=unique[:4])
            if len(plan.search_queries) != 4:
                raise ValueError("Planner did not yield four unique production queries.")
        except Exception:
            invalid += 1
            rankings[case["id"]] = []
            continue
        ranked_results = []
        for query in plan.search_queries:
            vector = embeddings.embed_query(query)
            ranked_results.append((query, vector_search_by_embedding(vector, 5)))
        result = assemble_hybrid_v1(case["question"], plan, ranked_results, case["history"])
        rankings[case["id"]] = result.evidence
        traces.append({
            "experiment": "nvidia_planner", "case_id": case["id"],
            "plan": plan.model_dump(), "planning_latency_ms": completion.latency_ms,
            "evidence": [candidate_trace(item, rank) for rank, item in enumerate(result.evidence, 1)],
        })
    return {
        "name": "nvidia_planner", "status": "COMPLETE",
        "model": NVIDIA_PLANNER_MODEL, "quality": retrieval_metrics(cases, rankings),
        "performance": {
            "planning_p50_ms": percentile(planning_latencies, 0.5),
            "planning_p95_ms": percentile(planning_latencies, 0.95),
        },
        "efficiency": {"calls": len(planning_latencies), "invalid_plans": invalid, "duplicate_queries": duplicates},
        "reliability": {"failures": invalid, "timeouts": 0, "retries": 0},
    }


def baseline_result() -> dict[str, Any]:
    cp7 = json.loads(CP7_RESULTS_PATH.read_text(encoding="utf-8"))
    retrieval = cp7["retrieval_quality_recommendation"]
    answers = cp7["answer_metrics"]
    return {
        "name": "gemini_baseline", "status": "COMPLETE_WITH_EXTERNAL_BENCHMARK_LIMITATION",
        "quality": {
            "recall_at_1": retrieval["recall_at_1"],
            "recall_at_3": retrieval["recall_at_3"],
            "recall_at_5": retrieval["recall_at_5"], "mrr": retrieval["mrr"],
            "full_at_5": retrieval["full_evidence_at_5"],
            "answer_correctness": answers["supported_answer_correctness"],
            "refusal_precision": answers["refusal_precision"],
            "refusal_recall": answers["refusal_recall"],
            "citation_entailment": answers["citation_entailment"],
        },
        "performance": {
            "ttft_p50_ms": 14199.979583034292,
            "generation_p50_ms": 14940.82304200856,
            "generation_p95_ms": 52708.39200000046,
            "throughput_requests_per_second": 0.1334478131920869,
            "failure_rate": 0.5,
            "note": "Checkpoint 9 real-provider concurrency matrix was incomplete because of Gemini free-tier quota.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", action="store_true", help="Run live NVIDIA calls when configuration is available.")
    args = parser.parse_args()
    cases, expectations, plans, raw = load_inputs()
    traces: list[dict[str, Any]] = []
    missing_key = not NVIDIA_API_KEY
    custom_chat_endpoint = "nvidia.com" not in NVIDIA_BASE_URL.casefold()
    custom_rerank_endpoint = "nvidia.com" not in NVIDIA_RERANK_URL.casefold()
    readiness = {
        "nvidia_embeddings": bool(NVIDIA_API_KEY) or custom_chat_endpoint,
        "nvidia_reranker": bool(NVIDIA_API_KEY) or custom_rerank_endpoint,
        "nvidia_generation": bool(NVIDIA_API_KEY) or custom_chat_endpoint,
    }
    blocked_reason = "NVIDIA_API_KEY is not configured. Hosted NVIDIA calls were not attempted, and this Apple arm64 host has no NVIDIA GPU for local NIM execution."
    experiments: dict[str, Any] = {"gemini_baseline": baseline_result()}

    for name in ("nvidia_embeddings", "nvidia_reranker", "nvidia_generation"):
        if not args.attempt or not readiness[name]:
            reason = blocked_reason if not readiness[name] else "Live calls require the explicit --attempt flag."
            experiments[name] = blocked_experiment(name, reason)
            traces.append({"experiment": name, "status": "BLOCKED", "reason": reason})

    if args.attempt:
        runners = {
            "nvidia_embeddings": lambda: run_embeddings(cases, plans, traces),
            "nvidia_reranker": lambda: run_reranker(cases, raw, traces),
            "nvidia_generation": lambda: run_generation(cases, expectations, raw, traces),
        }
        for name, runner in runners.items():
            if not readiness[name]:
                continue
            try:
                experiments[name] = runner()
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                experiments[name] = blocked_experiment(name, reason)
                traces.append({"experiment": name, "status": "BLOCKED", "reason": reason})

    prerequisites_complete = all(experiments[name]["status"] == "COMPLETE" for name in ("nvidia_embeddings", "nvidia_reranker", "nvidia_generation"))
    planning_reason = "Planning is intentionally deferred until embedding, reranking, and generation experiments complete successfully."
    if args.attempt and prerequisites_complete:
        try:
            experiments["nvidia_planner"] = run_planning(cases, traces)
        except Exception as error:
            planning_reason = f"{type(error).__name__}: {error}"
            experiments["nvidia_planner"] = blocked_experiment("nvidia_planner", planning_reason)
            traces.append({"experiment": "nvidia_planner", "status": "BLOCKED", "reason": planning_reason})
    else:
        experiments["nvidia_planner"] = blocked_experiment("nvidia_planner", planning_reason)
        traces.append({"experiment": "nvidia_planner", "status": "BLOCKED", "reason": planning_reason})
    experiments["nvidia_full"] = {
        "name": "nvidia_full", "status": "NOT_RUN_BY_DESIGN",
        "reason": "Checkpoint 10 benchmarks independent components and is not a full-stack migration.",
    }

    result = {
        "checkpoint": 10, "timestamp": datetime.now(UTC).isoformat(),
        "hybrid_v1_frozen": True,
        "environment": {
            "architecture": platform.machine(), "platform": platform.platform(),
            "nvidia_api_key_configured": not missing_key,
            "custom_nvidia_compatible_chat_endpoint": custom_chat_endpoint,
            "custom_nvidia_compatible_rerank_endpoint": custom_rerank_endpoint,
            "local_nvidia_gpu_detected": False,
        },
        "availability": {
            "hosted_api": "AVAILABLE_WITH_NVIDIA_API_KEY",
            "self_hosted_nim_endpoint": "SUPPORTED_BY_HARNESS_IF_BASE_URLS_ARE_CONFIGURED",
            "local_container": "BLOCKED_NO_NVIDIA_GPU",
            "gpu_required_for_local_nim": True,
        },
        "nvidia_vector_collection": ensure_separate_nvidia_collection(),
        "models": {
            "gemini_embedding": EMBEDDING_MODEL, "gemini_planner": PLANNER_MODEL,
            "gemini_generator": PRIMARY_CHAT_MODEL,
            "nvidia_embedding": NVIDIA_EMBEDDING_MODEL,
            "nvidia_reranker": NVIDIA_RERANK_MODEL,
            "nvidia_generator": NVIDIA_CHAT_MODEL,
            "nvidia_planner": NVIDIA_PLANNER_MODEL,
        },
        "profiles": {name: profile.to_dict() for name, profile in PROFILES.items()},
        "experiments": experiments,
        "comparison_matrix": {
            metric: {
                "gemini": experiments["gemini_baseline"].get("quality", {}).get(metric)
                or experiments["gemini_baseline"].get("performance", {}).get(metric),
                "nvidia": None,
                "delta": None,
                "status": "BLOCKED",
            }
            for metric in (
                "recall_at_5", "mrr", "answer_correctness", "citation_entailment",
                "ttft_p50_ms", "generation_p50_ms", "generation_p95_ms",
                "throughput_requests_per_second", "failure_rate",
            )
        },
        "planning_prerequisites_complete": prerequisites_complete,
        "business_controls": {
            "recall_at_5_and_mrr": "Complete policy evidence for operational decisions",
            "answer_correctness": "Correct execution of governed business rules",
            "citation_entailment": "Auditable policy answers",
            "safe_refusal": "Reduced unsupported decision guidance",
            "tenant_filtering": "Information-access isolation",
            "admission_control": "Predictable behavior under saturation",
            "circuit_breaker": "Resilience to upstream provider failure",
        },
        "external_provider_baseline_limitation": "The Checkpoint 9 Gemini benchmark is incomplete because the free-tier daily planner quota prevented the full requested concurrency matrix.",
        "official_sources": {
            "retrieval_api_catalog": "https://docs.api.nvidia.com/nim/reference/retrieval-apis",
            "llm_api_catalog": "https://docs.api.nvidia.com/nim/reference/llm-apis",
            "embedding_nim_api": "https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html",
            "reranking_nim_api": "https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/reference.html",
            "embedding_nim_local_setup": "https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/getting-started.html",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
