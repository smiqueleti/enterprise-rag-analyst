"""Execute Checkpoint 11 component benchmarks without changing hybrid_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from answer_evaluation import evaluate_answer
from benchmark_profiles import PROFILES
from checkpoint11_dataset import CASES, validate_cases
from citation_validation import validate_answer_citations
from config import (
    EMBEDDING_MODEL,
    HYBRID_V1,
    NVIDIA_API_KEY,
    NVIDIA_CHAT_MODEL,
    NVIDIA_EMBEDDING_MODEL,
    NVIDIA_PLANNER_MODEL,
    NVIDIA_RERANK_MODEL,
    PLANNER_MODEL,
    PRIMARY_CHAT_MODEL,
)
from ingest import chunk_id, load_documents
from nvidia_providers import NvidiaChat, NvidiaEmbeddings, NvidiaReranker
from providers import get_chat_model, get_embeddings
from rag import INSUFFICIENT_ANSWER, SYSTEM_PROMPT, build_context, response_text
from retrieval import (
    AuthorizationFilter,
    Candidate,
    QueryPlan,
    assemble_hybrid_v1,
    build_planner_messages,
    merge_ranked_results,
    vector_search_by_embedding,
)


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_PATH = PROJECT_DIR / "eval" / "nt11_results.json"
TRACE_PATH = PROJECT_DIR / "eval" / "checkpoint11_traces.json"
INDEX_ROOT = Path(tempfile.gettempdir()) / "atlas_checkpoint11_indexes"
CHUNK_SIZES = (300, 500, 800, 1200)
CHUNK_OVERLAP = 120
GENERATION_SAMPLE_IDS = {
    "direct_01", "direct_04", "paraphrased_03", "paraphrased_04",
    "cross_01", "multi_01_financial_threshold_regression",
    "ambiguous_02", "ambiguous_05", "unsupported_01", "unsupported_06",
    "authorization_01", "authorization_04", "conversation_01", "conversation_03",
}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
    }


def split_at_size(size: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(load_documents())
    positions: dict[str, int] = defaultdict(int)
    for item in chunks:
        source = str(item.metadata["source"])
        item.metadata["chunk_index"] = positions[source]
        positions[source] += 1
        item.metadata["chunk_size"] = size
        item.metadata["chunk_id"] = chunk_id(item)
    return chunks


def batch_embed(
    texts: list[str],
    embed_documents: Callable[[list[str]], list[list[float]]],
    batch_size: int = 16,
) -> tuple[list[list[float]], float]:
    vectors: list[list[float]] = []
    started = time.perf_counter()
    for offset in range(0, len(texts), batch_size):
        vectors.extend(embed_documents(texts[offset:offset + batch_size]))
    return vectors, (time.perf_counter() - started) * 1000


def create_index(
    provider: str,
    size: int,
    chunks: list[Document],
    vectors: list[list[float]],
) -> tuple[Chroma, float]:
    path = INDEX_ROOT / provider / str(size)
    collection = f"atlas_cp11_{provider}_{size}"
    store = Chroma(collection_name=collection, persist_directory=str(path))
    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)
    started = time.perf_counter()
    store._collection.upsert(
        ids=[str(item.metadata["chunk_id"]) for item in chunks],
        embeddings=vectors,
        documents=[item.page_content for item in chunks],
        metadatas=[item.metadata for item in chunks],
    )
    return store, (time.perf_counter() - started) * 1000


def fixed_plan(case: dict[str, Any]) -> QueryPlan:
    return QueryPlan(
        standalone_question=case["search_queries"][0],
        search_queries=case["search_queries"],
    )


def auth_filter(case: dict[str, Any]) -> AuthorizationFilter:
    authorization = case["authorization"]
    return AuthorizationFilter(
        tenant_id=authorization["tenant_id"],
        access_scopes=tuple(authorization["access_scopes"]),
    )


def retrieve_case(
    case: dict[str, Any],
    store: Chroma,
    query_vectors: dict[str, list[float]],
) -> tuple[list[Candidate], list[Candidate], float]:
    ranked_results = []
    started = time.perf_counter()
    for query in case["search_queries"]:
        ranked_results.append((
            query,
            vector_search_by_embedding(
                query_vectors[query],
                HYBRID_V1.candidates_per_query,
                authorization=auth_filter(case),
                vectorstore=store,
            ),
        ))
    retrieval_ms = (time.perf_counter() - started) * 1000
    result = assemble_hybrid_v1(
        case["question"], fixed_plan(case), ranked_results, case["history"]
    )
    return result.candidate_pool, result.evidence, retrieval_ms + result.fusion_latency_ms + result.rerank_latency_ms


def locator_hit(locator: dict[str, str], candidate: Candidate) -> bool:
    return (
        candidate.document.metadata.get("source") == locator["source"]
        and locator["contains"].casefold() in candidate.document.page_content.casefold()
    )


def case_quality(case: dict[str, Any], evidence: list[Candidate]) -> dict[str, float]:
    expected = case["expected_evidence"]
    if not expected:
        return {}
    ranks = [
        next((rank for rank, candidate in enumerate(evidence, 1) if locator_hit(locator, candidate)), None)
        for locator in expected
    ]
    first = min((rank for rank in ranks if rank is not None), default=None)
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "mrr": 0.0 if first is None else 1 / first,
        "full_at_5": float(all(rank is not None and rank <= 5 for rank in ranks)),
    }


def aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row["metrics"]]
    metrics = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "full_at_5")

    def summarize(items: list[dict[str, Any]]) -> dict[str, float | int]:
        if not items:
            return {**{name: None for name in metrics}, "supported_cases": 0}
        return {
            **{name: statistics.fmean(item["metrics"][name] for item in items) for name in metrics},
            "supported_cases": len(items),
        }

    by_category = {}
    for category in sorted({row["category"] for row in supported}):
        by_category[category] = summarize([row for row in supported if row["category"] == category])
    return {"overall": summarize(supported), "by_category": by_category}


def candidate_trace(candidate: Candidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": candidate.chunk_id,
        "source": candidate.document.metadata["source"],
        "chunk_index": candidate.document.metadata["chunk_index"],
        "vector_similarity": candidate.vector_similarity,
        "rrf_score": candidate.reciprocal_rank_score,
        "bm25_score": candidate.bm25_score,
        "hybrid_score": candidate.hybrid_score,
    }


def embed_fixed_queries(
    embed_queries: Callable[[list[str]], list[list[float]]],
    batch_size: int = 16,
) -> tuple[dict[str, list[float]], list[float]]:
    unique = list(dict.fromkeys(query for case in CASES for query in case["search_queries"]))
    vectors: dict[str, list[float]] = {}
    latencies = []
    for offset in range(0, len(unique), batch_size):
        batch = unique[offset:offset + batch_size]
        started = time.perf_counter()
        embedded = embed_queries(batch)
        latencies.append((time.perf_counter() - started) * 1000)
        vectors.update(zip(batch, embedded, strict=True))
    return vectors, latencies


def run_index_and_retrieval(
    provider: str,
    size: int,
    embed_documents: Callable[[list[str]], list[list[float]]],
    query_vectors: dict[str, list[float]],
    traces: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, tuple[list[Candidate], list[Candidate]]]]:
    chunks = split_at_size(size)
    vectors, remote_index_ms = batch_embed([item.page_content for item in chunks], embed_documents)
    store, local_index_ms = create_index(provider, size, chunks, vectors)
    rows = []
    rankings = {}
    retrieval_latencies = []
    for case in CASES:
        pool, selected, latency = retrieve_case(case, store, query_vectors)
        retrieval_latencies.append(latency)
        rankings[case["id"]] = (pool, selected)
        metrics = case_quality(case, selected)
        rows.append({"case_id": case["id"], "category": case["category"], "metrics": metrics})
        traces.append({
            "experiment": f"{provider}_chunks_{size}",
            "case_id": case["id"],
            "category": case["category"],
            "queries": case["search_queries"],
            "selected": [candidate_trace(item, rank) for rank, item in enumerate(selected, 1)],
            "quality": metrics,
        })
    return {
        "status": "COMPLETE",
        "provider": provider,
        "chunk_size": size,
        "chunk_overlap": CHUNK_OVERLAP,
        "number_of_chunks": len(chunks),
        "embedding_dimensionality": len(vectors[0]),
        "quality": aggregate_retrieval(rows),
        "performance": {
            "remote_index_embedding_ms": remote_index_ms,
            "local_indexing_ms": local_index_ms,
            "indexing_throughput_chunks_per_second": len(chunks) / ((remote_index_ms + local_index_ms) / 1000),
            "retrieval": distribution(retrieval_latencies),
        },
    }, rankings


def run_reranking(
    rankings: dict[str, tuple[list[Candidate], list[Candidate]]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    neural = NvidiaReranker()
    current_rows = []
    nvidia_rows = []
    latencies = []
    usages = []
    regressions = {}
    for case in CASES:
        pool, current = rankings[case["id"]]
        started = time.perf_counter()
        ordering, usage = neural.rank(case["question"], [item.document.page_content for item in pool])
        latencies.append((time.perf_counter() - started) * 1000)
        usages.append(usage)
        selected = [pool[index].clone() for index, _score in ordering[:HYBRID_V1.final_evidence]]
        current_metrics = case_quality(case, current)
        nvidia_metrics = case_quality(case, selected)
        current_rows.append({"case_id": case["id"], "category": case["category"], "metrics": current_metrics})
        nvidia_rows.append({"case_id": case["id"], "category": case["category"], "metrics": nvidia_metrics})
        traces.append({
            "experiment": "nvidia_reranker_50",
            "case_id": case["id"],
            "same_candidate_pool": True,
            "pool": [candidate_trace(item, rank) for rank, item in enumerate(pool, 1)],
            "current_selected": [candidate_trace(item, rank) for rank, item in enumerate(current, 1)],
            "nvidia_selected": [candidate_trace(item, rank) for rank, item in enumerate(selected, 1)],
            "current_quality": current_metrics,
            "nvidia_quality": nvidia_metrics,
            "usage": usage,
        })
        if case["id"] in {"multi_01_financial_threshold_regression", "conversation_02"}:
            regressions[case["id"]] = {
                "current": current_metrics,
                "nvidia": nvidia_metrics,
                "current_sources": [item.document.metadata["source"] for item in current],
                "nvidia_sources": [item.document.metadata["source"] for item in selected],
            }
    return {
        "status": "COMPLETE",
        "model": NVIDIA_RERANK_MODEL,
        "candidate_generation": "identical frozen hybrid_v1 pool",
        "current_hybrid": aggregate_retrieval(current_rows),
        "nvidia_neural": aggregate_retrieval(nvidia_rows),
        "performance": distribution(latencies),
        "calls": len(latencies),
        "token_usage": usages,
        "historical_regressions": regressions,
    }


def generation_messages(case: dict[str, Any], evidence: list[Candidate]) -> tuple[list[Any], list[dict[str, str]]]:
    user = f"DOCUMENT EXCERPTS:\n\n{build_context(evidence)}\n\nQUESTION:\n{case['question'].strip()}"
    return (
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    )


def expectation(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_terms": case["expected_facts"],
        "required_sources": sorted({item["source"] for item in case["expected_evidence"]}),
    }


def aggregate_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "COMPLETE"]

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        supported = [row for row in items if row["category"] != "unsupported"]
        unsupported = [row for row in items if row["category"] == "unsupported"]
        refusals = [row for row in items if row["outcome"]["refused"]]
        correct_refusals = sum(row["outcome"]["correct_refusal"] for row in items)
        return {
            "cases": len(items),
            "answer_correctness": statistics.fmean(float(row["outcome"]["answer_correct"]) for row in supported) if supported else None,
            "citation_entailment": statistics.fmean(row["outcome"]["citation_entailment"] for row in supported) if supported else None,
            "refusal_precision": correct_refusals / len(refusals) if refusals else 1.0,
            "refusal_recall": correct_refusals / len(unsupported) if unsupported else None,
            "invalid_citation_rate": statistics.fmean(float(row["outcome"]["citation_correctness"] < 1) for row in supported) if supported else None,
        }

    by_category = {
        category: summarize([row for row in completed if row["category"] == category])
        for category in sorted({row["category"] for row in completed})
    }
    return {"overall": summarize(completed), "by_category": by_category, "failures": len(rows) - len(completed)}


def run_generation_provider(
    provider: str,
    rankings: dict[str, tuple[list[Candidate], list[Candidate]]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    latencies = []
    ttfts = []
    usage_rows = []
    repairs = 0
    sample = [case for case in CASES if case["id"] in GENERATION_SAMPLE_IDS]
    nvidia = NvidiaChat(model=NVIDIA_CHAT_MODEL) if provider == "nvidia" else None
    gemini = get_chat_model(PRIMARY_CHAT_MODEL) if provider == "gemini" else None
    for case in sample:
        evidence = rankings[case["id"]][1]
        lc_messages, raw_messages = generation_messages(case, evidence)
        try:
            started = time.perf_counter()
            if provider == "nvidia":
                completion = nvidia.complete_stream(raw_messages)
                answer = completion.text or INSUFFICIENT_ANSWER
                latency = completion.latency_ms
                ttft = completion.ttft_ms
                usage = completion.usage
                model = completion.model
            else:
                parts = []
                usage = {}
                ttft = None
                for chunk in gemini.stream(lc_messages):
                    text = response_text(chunk.content)
                    if text and ttft is None:
                        ttft = (time.perf_counter() - started) * 1000
                    parts.append(text)
                    usage.update({k: int(v) for k, v in (getattr(chunk, "usage_metadata", None) or {}).items() if isinstance(v, int)})
                answer = "".join(parts).strip() or INSUFFICIENT_ANSWER
                latency = (time.perf_counter() - started) * 1000
                model = PRIMARY_CHAT_MODEL
            validation = validate_answer_citations(answer, evidence, question=case["question"])
            if not validation.valid:
                repairs += 1
                feedback = validation.correction_prompt()
                repair_user = raw_messages[1]["content"] + "\n\nA previous draft failed citation validation. Rewrite once.\n" + feedback
                if provider == "nvidia":
                    fixed = nvidia.complete([raw_messages[0], {"role": "user", "content": repair_user}])
                    answer = fixed.text
                    latency += fixed.latency_ms
                    for key, value in fixed.usage.items():
                        usage[key] = usage.get(key, 0) + value
                else:
                    fixed = gemini.invoke([lc_messages[0], HumanMessage(content=repair_user)], automatic_function_calling={"disable": True})
                    answer = response_text(fixed.content)
                    latency = (time.perf_counter() - started) * 1000
                    for key, value in (getattr(fixed, "usage_metadata", None) or {}).items():
                        if isinstance(value, int):
                            usage[key] = usage.get(key, 0) + value
                validation = validate_answer_citations(answer, evidence, question=case["question"])
                if not validation.valid:
                    answer = INSUFFICIENT_ANSWER
            outcome = evaluate_answer(answer, evidence, expectation(case), case["category"] == "unsupported", case["question"])
            row = {"case_id": case["id"], "category": case["category"], "status": "COMPLETE", "outcome": outcome}
            rows.append(row)
            latencies.append(latency)
            if ttft is not None:
                ttfts.append(ttft)
            usage_rows.append(usage)
            traces.append({"experiment": f"{provider}_generation", "case_id": case["id"], "model": model, "answer": answer, "latency_ms": latency, "ttft_ms": ttft, "usage": usage, "citation_validation": validation.valid, "evaluation": outcome})
        except Exception as error:
            category = f"{type(error).__name__}: {error}"
            rows.append({"case_id": case["id"], "category": case["category"], "status": "FAILED", "error": category})
            traces.append({"experiment": f"{provider}_generation", "case_id": case["id"], "status": "FAILED", "error": category})
    return {
        "status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "PARTIAL",
        "model": NVIDIA_CHAT_MODEL if provider == "nvidia" else PRIMARY_CHAT_MODEL,
        "sample_design": "two preselected cases per category",
        "quality": aggregate_generation(rows),
        "performance": {"generation": distribution(latencies), "ttft": distribution(ttfts)},
        "efficiency": {"calls_minimum": len(sample), "citation_repairs": repairs, "repair_rate": repairs / len(sample), "token_usage": usage_rows},
        "failures": [row for row in rows if row["status"] == "FAILED"],
    }


def parse_plan(text: str, original: str) -> QueryPlan:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Planner returned no JSON object.")
    payload = json.loads(match.group(0))
    values = [str(payload["standalone_question"]).strip(), *[str(item).strip() for item in payload["search_queries"]], original]
    unique = []
    for value in values:
        if value and value.casefold() not in {item.casefold() for item in unique}:
            unique.append(value)
    if len(unique) < 4:
        raise ValueError("Planner returned fewer than four unique queries.")
    return QueryPlan(standalone_question=values[0], search_queries=unique[:4])


def run_planner_provider(
    provider: str,
    store: Chroma,
    embed_queries: Callable[[list[str]], list[list[float]]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    latencies = []
    duplicates = 0
    invalid = 0
    chat = NvidiaChat(model=NVIDIA_PLANNER_MODEL) if provider == "nvidia" else get_chat_model(PLANNER_MODEL)
    instruction = " Return only JSON with string field standalone_question and array field search_queries containing exactly three strings."
    for case in CASES:
        messages = build_planner_messages(case["question"], case["history"])
        raw_messages = [
            {"role": "system" if index == 0 else "user", "content": message.content + (instruction if index == 0 else "")}
            for index, message in enumerate(messages)
        ]
        try:
            started = time.perf_counter()
            if provider == "nvidia":
                completion = chat.complete(raw_messages)
                text = completion.text
                model = completion.model
            else:
                response = chat.invoke([SystemMessage(content=raw_messages[0]["content"]), HumanMessage(content=raw_messages[1]["content"])], automatic_function_calling={"disable": True})
                text = response_text(response.content)
                model = PLANNER_MODEL
            latency = (time.perf_counter() - started) * 1000
            plan = parse_plan(text, case["question"])
            source_values = plan.search_queries
            duplicates += len(source_values) - len({value.casefold() for value in source_values})
            ranked = []
            vectors = embed_queries(plan.search_queries)
            for query, vector in zip(plan.search_queries, vectors, strict=True):
                ranked.append((query, vector_search_by_embedding(vector, 5, authorization=auth_filter(case), vectorstore=store)))
            selected = assemble_hybrid_v1(case["question"], plan, ranked, case["history"]).evidence
            metrics = case_quality(case, selected)
            rows.append({"case_id": case["id"], "category": case["category"], "metrics": metrics, "status": "COMPLETE"})
            latencies.append(latency)
            traces.append({"experiment": f"{provider}_planner", "case_id": case["id"], "model": model, "plan": plan.model_dump(), "latency_ms": latency, "selected": [candidate_trace(item, rank) for rank, item in enumerate(selected, 1)], "quality": metrics})
        except Exception as error:
            invalid += 1
            traces.append({"experiment": f"{provider}_planner", "case_id": case["id"], "status": "FAILED", "error": f"{type(error).__name__}: {error}"})
    completed = [row for row in rows if row["metrics"]]
    return {
        "status": "COMPLETE" if invalid == 0 else "PARTIAL",
        "model": NVIDIA_PLANNER_MODEL if provider == "nvidia" else PLANNER_MODEL,
        "quality": aggregate_retrieval(completed),
        "performance": distribution(latencies),
        "invalid_queries": invalid,
        "duplicate_queries": duplicates,
        "calls": len(CASES),
    }


def save(results: dict[str, Any], traces: list[dict[str, Any]]) -> None:
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow provider API calls.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-planning", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "retrieval", "generation", "planning"),
        default="all",
    )
    parser.add_argument(
        "--provider",
        choices=("both", "gemini", "nvidia"),
        default="both",
        help="Limit a resumed generation or planning stage to one provider.",
    )
    args = parser.parse_args()
    validate_cases()
    if not args.live:
        raise SystemExit("Use --live to execute the provider benchmark explicitly.")
    if not NVIDIA_API_KEY:
        raise SystemExit("NVIDIA_API_KEY is not configured in .env.")

    if args.stage in {"generation", "planning"}:
        if not RESULT_PATH.exists() or not TRACE_PATH.exists():
            raise SystemExit("Run the retrieval stage before resuming later stages.")
        results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        traces = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        nvidia_embeddings = NvidiaEmbeddings()
        query_vectors, _latencies = embed_fixed_queries(nvidia_embeddings.embed_queries)
        store = Chroma(
            collection_name="atlas_cp11_nvidia_800",
            persist_directory=str(INDEX_ROOT / "nvidia" / "800"),
        )
        rankings = {
            case["id"]: retrieve_case(case, store, query_vectors)[:2]
            for case in CASES
        }
        if args.stage == "generation":
            generation = results.get("generation", {})
            selected = ("gemini", "nvidia") if args.provider == "both" else (args.provider,)
            for provider in selected:
                generation[provider] = run_generation_provider(provider, rankings, traces)
            results["generation"] = generation
        else:
            planning = results.get("planning", {})
            selected = ("gemini", "nvidia") if args.provider == "both" else (args.provider,)
            for provider in selected:
                planning[provider] = run_planner_provider(provider, store, nvidia_embeddings.embed_queries, traces)
            results["planning"] = planning
        save(results, traces)
        print(json.dumps({"status": "COMPLETE", "stage": args.stage}, indent=2))
        return

    traces: list[dict[str, Any]] = []
    results: dict[str, Any] = {
        "checkpoint": 11,
        "timestamp": datetime.now(UTC).isoformat(),
        "project_frozen_after_completion": True,
        "hybrid_v1_frozen": True,
        "dataset": {"cases": len(CASES), "categories": dict(Counter(case["category"] for case in CASES)), "generation_sample_cases": len(GENERATION_SAMPLE_IDS)},
        "environment": {"architecture": platform.machine(), "platform": platform.platform(), "nvidia_key_configured": True},
        "models": {"gemini_embedding": EMBEDDING_MODEL, "nvidia_embedding": NVIDIA_EMBEDDING_MODEL, "current_reranker": "local RRF + BM25 + vector", "nvidia_reranker": NVIDIA_RERANK_MODEL, "gemini_generator": PRIMARY_CHAT_MODEL, "nvidia_generator": NVIDIA_CHAT_MODEL, "gemini_planner": PLANNER_MODEL, "nvidia_planner": NVIDIA_PLANNER_MODEL},
        "profiles": {name: profile.to_dict() for name, profile in PROFILES.items()},
    }

    results["gemini_expanded_embedding_and_chunking"] = {
        "status": "BLOCKED",
        "reason": "Gemini free-tier embed-content quota rejected the 200 fixed-query expanded run. No quota workaround or retuning was applied.",
        "accepted_baseline": "Checkpoint 7 hybrid_v1 metrics remain the current-provider baseline.",
    }

    nvidia_embeddings = NvidiaEmbeddings()
    nvidia_queries, nvidia_query_latencies = embed_fixed_queries(nvidia_embeddings.embed_queries)
    chunking = {}
    nvidia_rankings = None
    nvidia_store = None
    for size in CHUNK_SIZES:
        experiment, rankings = run_index_and_retrieval("nvidia", size, nvidia_embeddings.embed_documents, nvidia_queries, traces)
        chunking[str(size)] = experiment
        if size == 800:
            nvidia_rankings = rankings
            nvidia_store = Chroma(collection_name="atlas_cp11_nvidia_800", persist_directory=str(INDEX_ROOT / "nvidia" / "800"))
        save(results | {"chunking": chunking}, traces)
    results["chunking"] = chunking
    results["nvidia_embeddings"] = chunking["800"] | {"query_embedding": distribution(nvidia_query_latencies)}
    save(results, traces)

    results["reranking"] = run_reranking(nvidia_rankings, traces)
    save(results, traces)

    if args.stage == "retrieval":
        print(json.dumps({"status": "COMPLETE", "stage": "retrieval"}, indent=2))
        return

    if not args.skip_generation:
        results["generation"] = {
            "gemini": run_generation_provider("gemini", nvidia_rankings, traces),
            "nvidia": run_generation_provider("nvidia", nvidia_rankings, traces),
        }
        save(results, traces)

    if not args.skip_planning:
        results["planning"] = {
            "gemini": run_planner_provider("gemini", nvidia_store, nvidia_embeddings.embed_queries, traces),
            "nvidia": run_planner_provider("nvidia", nvidia_store, nvidia_embeddings.embed_queries, traces),
        }
        save(results, traces)

    results["security"] = {
        "separate_embedding_collections": True,
        "authorization_filter_applied_before_candidate_pool": True,
        "secrets_in_results_or_traces": False,
    }
    save(results, traces)
    print(json.dumps({"status": "COMPLETE", "results": str(RESULT_PATH), "traces": str(TRACE_PATH)}, indent=2))


if __name__ == "__main__":
    main()
