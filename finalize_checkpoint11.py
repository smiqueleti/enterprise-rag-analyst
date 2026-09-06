"""Consolidate measured Checkpoint 11 artifacts and write the engineering report."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from checkpoint11_dataset import CASES
from config import NVIDIA_API_KEY, PROJECT_DIR


RESULT_PATH = PROJECT_DIR / "eval" / "nt11_results.json"
TRACE_PATH = PROJECT_DIR / "eval" / "checkpoint11_traces.json"
REPORT_PATH = PROJECT_DIR / "eval" / "CHECKPOINT_11_REPORT.md"
DATASET_PATH = PROJECT_DIR / "eval" / "checkpoint11_questions.json"
CP7_PATH = PROJECT_DIR / "eval" / "checkpoint7_results.json"
TEMP_DIR = Path(tempfile.gettempdir())
LEGACY_EMBED = TEMP_DIR / "atlas_cp11_nvidia_embeddings_legacy.json"
LEGACY_RERANK = TEMP_DIR / "atlas_cp11_reranker.json"


def pct(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.1f}%"


def number(value: float | None, digits: int = 3) -> str:
    return "not measured" if value is None else f"{value:.{digits}f}"


def ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.1f} ms"


def error_categories(traces: list[dict[str, Any]], experiment: str) -> dict[str, int]:
    values = []
    for item in traces:
        if item.get("experiment") != experiment or item.get("status") != "FAILED":
            continue
        error = str(item.get("error", "Unknown"))
        if "429" in error or "RESOURCE_EXHAUSTED" in error:
            values.append("quota_or_rate_limit")
        elif "503" in error or "UNAVAILABLE" in error:
            values.append("temporary_unavailable")
        elif "JSON" in error or "Planner" in error:
            values.append("malformed_response")
        else:
            values.append(error.split(":", 1)[0])
    return dict(Counter(values))


def main() -> None:
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    traces = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    cp7 = json.loads(CP7_PATH.read_text(encoding="utf-8"))
    baseline = cp7["retrieval_quality_recommendation"]
    answer_baseline = cp7["answer_metrics"]

    legacy_embedding = json.loads(LEGACY_EMBED.read_text(encoding="utf-8")) if LEGACY_EMBED.exists() else None
    legacy_rerank = json.loads(LEGACY_RERANK.read_text(encoding="utf-8")) if LEGACY_RERANK.exists() else None
    if legacy_embedding:
        results["matched_legacy_embedding_comparison"] = {
            "cases": 19,
            "current_gemini": {
                "recall_at_1": baseline["recall_at_1"],
                "recall_at_3": baseline["recall_at_3"],
                "recall_at_5": baseline["recall_at_5"],
                "mrr": baseline["mrr"],
                "full_at_5": baseline["full_evidence_at_5"],
            },
            "nvidia": legacy_embedding["result"],
        }
        traces.extend(legacy_embedding["traces"])
    if legacy_rerank:
        results["matched_legacy_reranker_comparison"] = legacy_rerank["result"]
        traces.extend(legacy_rerank["traces"])

    nvidia_plans = [item for item in traces if item.get("experiment") == "nvidia_planner" and item.get("plan")]
    duplicate_queries = sum(
        len(item["plan"]["search_queries"])
        - len({query.casefold() for query in item["plan"]["search_queries"]})
        for item in nvidia_plans
    )
    if "planning" in results and "nvidia" in results["planning"]:
        results["planning"]["nvidia"]["duplicate_queries"] = duplicate_queries
        results["planning"]["nvidia"]["provider_errors"] = error_categories(traces, "nvidia_planner")
    if "planning" in results and "gemini" in results["planning"]:
        gemini_errors = error_categories(traces, "gemini_planner")
        results["planning"]["gemini"]["invalid_queries"] = gemini_errors.get("malformed_response", 0)
        results["planning"]["gemini"]["provider_errors"] = gemini_errors

    for provider in ("gemini", "nvidia"):
        if provider in results.get("generation", {}):
            results["generation"][provider]["provider_errors"] = error_categories(traces, f"{provider}_generation")
    results["harness_corrections"] = {
        "nvidia_stream_parser": "The first NVIDIA generation attempt rejected usage-only SSE events. The parser was fixed, tested, and the NVIDIA run was repeated. Both attempts remain in the trace.",
        "planner_duplicate_counter": "Duplicates are counted only within the final four atomic queries, not between the standalone field and its intentional copy in search_queries.",
    }

    chunking = results["chunking"]
    best_size = max(
        chunking,
        key=lambda size: (
            chunking[size]["quality"]["overall"]["recall_at_5"],
            chunking[size]["quality"]["overall"]["mrr"],
            chunking[size]["quality"]["overall"]["full_at_5"],
            -chunking[size]["performance"]["retrieval"]["p95_ms"],
        ),
    )
    results["chunking_decision"] = {
        "selected_size": int(best_size),
        "production_runtime_changed": False,
        "reason": "500 characters achieved Recall@5 1.000, the highest MRR among the full-recall configurations, and Full@5 1.000. The production index was not rebuilt because the current Gemini embedding quota blocked a safe synchronized migration.",
    }

    matched_embed = results.get("matched_legacy_embedding_comparison", {})
    current_embed = matched_embed.get("current_gemini", {})
    nvidia_embed = matched_embed.get("nvidia", {}).get("quality", {})
    rerank = results["reranking"]
    current_rerank = rerank["current_hybrid"]["overall"]
    nvidia_rerank = rerank["nvidia_neural"]["overall"]
    nvidia_generation = results["generation"]["nvidia"]
    nvidia_planner = results["planning"]["nvidia"]

    results["quality_summary"] = {
        "accepted_current_baseline": {
            "recall_at_5": baseline["recall_at_5"],
            "mrr": baseline["mrr"],
            "answer_correctness": answer_baseline["supported_answer_correctness"],
            "refusal_precision": answer_baseline["refusal_precision"],
            "refusal_recall": answer_baseline["refusal_recall"],
            "citation_entailment": answer_baseline["citation_entailment"],
        },
        "expanded_nvidia_embedding_current_rerank": chunking["800"]["quality"],
        "expanded_nvidia_generation_sample": nvidia_generation["quality"],
    }
    results["component_matrix"] = {
        "embeddings": {
            "current": current_embed,
            "nvidia": nvidia_embed,
            "winner": "Gemini/current",
            "reason": "On the matched 19-case set, NVIDIA lowered Recall@5 and MRR. Its latency advantage does not pass the quality gate.",
        },
        "reranking": {
            "current": current_rerank,
            "nvidia": nvidia_rerank,
            "winner": "Current hybrid",
            "reason": "NVIDIA improved rank 1 and MRR but reduced Recall@3, Recall@5, and Full@5 while adding hosted inference latency.",
        },
        "generation": {
            "current": results["generation"]["gemini"]["quality"],
            "nvidia": nvidia_generation["quality"],
            "winner": "Gemini/current retained",
            "reason": "NVIDIA preserved safe refusal but only 9.1% of supported completed cases passed answer correctness after strict citation validation. The live Gemini comparison was quota-limited, so the accepted current quality baseline remains controlling.",
        },
        "planning": {
            "current": results["planning"]["gemini"],
            "nvidia": nvidia_planner,
            "winner": "Gemini/current retained",
            "reason": "The NVIDIA planner produced lower downstream quality than the fixed-query reference and had multi-second latency. Gemini could not complete the direct comparison because of quota exhaustion.",
        },
    }
    results["final_architecture"] = {
        "planner": "Gemini current planner",
        "embeddings": "Gemini embedding",
        "chunking": "500-character target after a controlled reindex; current runtime remains 800 until then",
        "vector_database": "Chroma with separate model-bound collections",
        "fusion": "RRF",
        "reranker": "local BM25 + vector hybrid",
        "generator": "Gemini current generator",
        "citation_validation": "deterministic claim-level validator with one bounded repair",
        "retrieval_profile": "hybrid_v1 unchanged",
    }
    results["business_controls"] = {
        "citation_entailment": "Auditable policy answers",
        "safe_refusal": "Reduced unsupported decision guidance",
        "tenant_filtering": "Information-access isolation",
        "admission_control": "Predictable behavior under saturation",
        "circuit_breaker": "Resilience to upstream provider failure",
        "recall_at_5_and_full_at_5": "Complete policy evidence for governed operational decisions",
    }
    results["remaining_limitations"] = [
        "Gemini free-tier quotas blocked the expanded embedding, generation, and planner comparison.",
        "Generation used a fixed 14-case stratified sample, not all 50 questions.",
        "The corpus has only five fictional documents and 12 to 66 chunks depending on configuration.",
        "NVIDIA hosted trial behavior does not predict dedicated NIM throughput or cost.",
        "The deterministic citation entailment check is lexical and can reject valid paraphrases.",
        "The 500-character chunking winner has not been promoted to the runtime index.",
    ]

    serialized = json.dumps(results, indent=2) + "\n"
    trace_serialized = json.dumps(traces, indent=2) + "\n"
    if NVIDIA_API_KEY and (NVIDIA_API_KEY in serialized or NVIDIA_API_KEY in trace_serialized):
        raise RuntimeError("Secret detected in Checkpoint 11 artifacts.")
    RESULT_PATH.write_text(serialized, encoding="utf-8")
    TRACE_PATH.write_text(trace_serialized, encoding="utf-8")
    DATASET_PATH.write_text(json.dumps(CASES, indent=2) + "\n", encoding="utf-8")

    chunk_rows = []
    for size in ("300", "500", "800", "1200"):
        item = chunking[size]
        quality = item["quality"]["overall"]
        perf = item["performance"]
        chunk_rows.append(
            f"| {size} | {item['number_of_chunks']} | {number(quality['recall_at_1'])} | {number(quality['recall_at_3'])} | {number(quality['recall_at_5'])} | {number(quality['mrr'])} | {perf['remote_index_embedding_ms']:.1f} | {perf['retrieval']['p95_ms']:.2f} |"
        )

    category_rows = []
    for category, quality in chunking["800"]["quality"]["by_category"].items():
        category_rows.append(
            f"| {category} | {quality['supported_cases']} | {number(quality['recall_at_1'])} | {number(quality['recall_at_3'])} | {number(quality['recall_at_5'])} | {number(quality['mrr'])} |"
        )

    generation_quality = nvidia_generation["quality"]["overall"]
    generation_rows = []
    for category, quality in nvidia_generation["quality"]["by_category"].items():
        generation_rows.append(
            f"| {category} | {quality['cases']} | {pct(quality['answer_correctness'])} | {pct(quality['citation_entailment'])} | {pct(quality['refusal_precision'])} | {pct(quality['refusal_recall'])} |"
        )
    report = f"""# Checkpoint 11 - Benchmark and Evidence Closure

## Executive decision

Checkpoint 11 is complete with explicit external-provider limitations.
No new serving architecture was introduced, and `hybrid_v1` remains frozen.
The evidence does not justify replacing any production component with the tested NVIDIA alternative.
The measured chunking winner is 500 characters, but the runtime remains at 800 characters until a synchronized Gemini reindex can be completed safely.
After this report, the project is frozen and there is no Checkpoint 12.

## Experimental controls

The evaluation dataset contains 50 manually curated questions across direct, paraphrased, cross-document, ambiguous, unsupported, authorization, and conversational categories.
Every supported question has policy-derived expected evidence and deterministic expected facts.
Four fixed retrieval queries were defined before the provider comparisons.
The corpus, questions, `hybrid_v1` values, evidence prompt, citation rules, and one-repair policy were held constant where applicable.
NVIDIA and Gemini vectors were never mixed in one collection.

## Chunking experiment

NVIDIA `nvidia/nemotron-3-embed-1b` embeddings were held constant across all four isolated indexes because the Gemini free-tier embedding quota blocked the expanded run.
The overlap remained 120 characters for every configuration.

| Chunk size | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Remote indexing ms | Retrieval p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(chunk_rows)}

The 500-character configuration is the measured winner because it reached Recall@5 1.000 and Full@5 1.000 while posting MRR 0.964.
The current 800-character configuration reached Recall@5 0.992, Full@5 0.976, and MRR 0.960.
This decision is recorded but not applied to the runtime index because doing so without a complete reindex would desynchronize code and stored vectors.

## Expanded retrieval quality by category

These results use NVIDIA embeddings, 800-character chunks, and the current local hybrid reranker over 42 supported cases.

| Category | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
{chr(10).join(category_rows)}

Unsupported cases are excluded from retrieval recall because they intentionally have no relevant evidence.

## Embedding comparison

The matched comparison uses the 19 legacy cases with complete accepted query-plan caches.
Gemini achieved Recall@1 {number(current_embed.get('recall_at_1'))}, Recall@3 {number(current_embed.get('recall_at_3'))}, Recall@5 {number(current_embed.get('recall_at_5'))}, and MRR {number(current_embed.get('mrr'))}.
NVIDIA achieved Recall@1 {number(nvidia_embed.get('recall_at_1'))}, Recall@3 {number(nvidia_embed.get('recall_at_3'))}, Recall@5 {number(nvidia_embed.get('recall_at_5'))}, and MRR {number(nvidia_embed.get('mrr'))}.
NVIDIA returned 2,048-dimensional embeddings and query embedding p50 was {ms(matched_embed.get('nvidia', {}).get('performance', {}).get('embedding_p50_ms'))} on one-query calls.
Gemini/current wins because the NVIDIA quality regression at Recall@5 and MRR outweighs its latency advantage.

## Neural reranking comparison

Both rerankers received the exact same 12-item candidate pool for every expanded case.
The current hybrid reranker achieved Recall@1 {number(current_rerank['recall_at_1'])}, Recall@3 {number(current_rerank['recall_at_3'])}, Recall@5 {number(current_rerank['recall_at_5'])}, MRR {number(current_rerank['mrr'])}, and Full@5 {number(current_rerank['full_at_5'])}.
The NVIDIA neural reranker achieved Recall@1 {number(nvidia_rerank['recall_at_1'])}, Recall@3 {number(nvidia_rerank['recall_at_3'])}, Recall@5 {number(nvidia_rerank['recall_at_5'])}, MRR {number(nvidia_rerank['mrr'])}, and Full@5 {number(nvidia_rerank['full_at_5'])}.
NVIDIA reranking latency was p50 {ms(rerank['performance']['p50_ms'])} and p95 {ms(rerank['performance']['p95_ms'])}.
NVIDIA improved early ranking but reduced evidence completeness, so the local hybrid reranker remains selected.

On the matched legacy financial-approval regression, NVIDIA moved the financial chunk from rank 4 to rank 3 but moved the priority-notification chunk from rank 2 to rank 4.
On the expanded 800-character NVIDIA-embedding run, neither reranker achieved Full@5 for that compound case.
The dedicated priority-notification conversational case remained at Recall@5 1.000 for both rerankers.

## Generation comparison

The generation experiment used the same 14 preselected cases, with exactly two cases per category.
Gemini completed only 1 of 14 cases because one request returned 503 and the remaining requests hit free-tier quota limits.
That live sample is not sufficient for a new Gemini quality estimate, so the accepted current baseline remains controlling.
NVIDIA completed 13 of 14 cases, with one provider 503.
NVIDIA supported-answer correctness was {pct(generation_quality['answer_correctness'])}.
NVIDIA refusal precision was {pct(generation_quality['refusal_precision'])}, refusal recall was {pct(generation_quality['refusal_recall'])}, and citation entailment among surviving factual answers was {pct(generation_quality['citation_entailment'])}.
The final invalid-citation rate was {pct(generation_quality['invalid_citation_rate'])}, but {nvidia_generation['efficiency']['citation_repairs']} of 14 cases required repair and most unsupported drafts became safe refusals after validation.
NVIDIA TTFT p50 was {ms(nvidia_generation['performance']['ttft']['p50_ms'])}, TTFT p95 was {ms(nvidia_generation['performance']['ttft']['p95_ms'])}, generation p50 was {ms(nvidia_generation['performance']['generation']['p50_ms'])}, and generation p95 was {ms(nvidia_generation['performance']['generation']['p95_ms'])}.
NVIDIA generation does not preserve the current answer-correctness gate and is not selected.

### NVIDIA generation quality by category

| Category | Completed cases | Answer correctness | Citation entailment | Refusal precision | Refusal recall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(generation_rows)}

Each category has two selected cases, except `direct`, where one NVIDIA provider 503 left one completed case.
Metrics marked not measured have no applicable supported or unsupported case in that category.

## Planner comparison

The NVIDIA planner completed all 50 cases with no malformed plans and no duplicate final atomic queries.
Its downstream Recall@5 was {number(nvidia_planner['quality']['overall']['recall_at_5'])}, MRR was {number(nvidia_planner['quality']['overall']['mrr'])}, planning p50 was {ms(nvidia_planner['performance']['p50_ms'])}, and planning p95 was {ms(nvidia_planner['performance']['p95_ms'])}.
Cross-document Recall@5 was {number(nvidia_planner['quality']['by_category']['cross_document']['recall_at_5'])}.
Gemini completed zero live planner cases because of external-provider quota exhaustion, so a direct live planner delta cannot be claimed.
The current planner remains selected because NVIDIA underperformed the fixed-query retrieval reference and added substantial latency.

## Final component matrix

| Component | Gemini/current | NVIDIA | Winner | Decision basis |
|---|---|---|---|---|
| Embeddings | Recall@5 {number(current_embed.get('recall_at_5'))}, MRR {number(current_embed.get('mrr'))} | Recall@5 {number(nvidia_embed.get('recall_at_5'))}, MRR {number(nvidia_embed.get('mrr'))} | Gemini/current | NVIDIA latency is attractive, but matched quality regressed. |
| Reranking | Recall@5 {number(current_rerank['recall_at_5'])}, Full@5 {number(current_rerank['full_at_5'])} | Recall@5 {number(nvidia_rerank['recall_at_5'])}, Full@5 {number(nvidia_rerank['full_at_5'])} | Current hybrid | Neural ranking improved rank 1 but lost complete evidence and added a hosted call. |
| Generation | Accepted quality baseline remains 1.000; live comparison quota-limited | Correctness {pct(generation_quality['answer_correctness'])}; repair rate {pct(nvidia_generation['efficiency']['repair_rate'])} | Gemini/current retained | NVIDIA failed the grounded-answer usability gate. |
| Planning | Live comparison quota-limited | Recall@5 {number(nvidia_planner['quality']['overall']['recall_at_5'])}; p95 {ms(nvidia_planner['performance']['p95_ms'])} | Gemini/current retained | NVIDIA was slower and weaker than the fixed-query reference. |

## Final recommended Atlas architecture

- Planner: current Gemini planner.
- Embeddings: current Gemini embedding model.
- Chunking: 500-character target after a controlled reindex; runtime remains 800 until then.
- Vector database: Chroma with model-bound collections.
- Retrieval: frozen `hybrid_v1` with four atomic queries, five candidates per query, candidate pool 12, and final evidence 5.
- Fusion: RRF.
- Reranking: local BM25 plus vector hybrid.
- Generation: current Gemini generator.
- Guardrail: deterministic citation validation with one bounded repair and exact safe refusal.

The recommended stack is not NVIDIA-based because none of the tested NVIDIA replacements passed the relevant quality gate.
GPU-backed inference helped embedding and reranking latency, but the five-document corpus is too small for that acceleration to offset quality loss and hosted operational complexity.
Local fusion, BM25, Chroma search, citation validation, authorization filtering, admission control, and circuit breaking clearly do not need GPU acceleration here.

## Enterprise control mapping

| Engineering gate | Enterprise requirement protected |
|---|---|
| Citation entailment | Auditable policy answers |
| Safe refusal | Reduced unsupported decision guidance |
| Tenant filtering | Information-access isolation |
| Admission control | Predictable behavior under saturation |
| Circuit breaker | Resilience to upstream provider failure |
| Recall@5 and Full@5 | Complete policy evidence for governed operational decisions |

## Remaining limitations

- Gemini free-tier quotas blocked the expanded embedding comparison and most live generation and planner comparisons.
- The generation benchmark used 14 stratified cases rather than all 50 questions.
- The corpus contains only five fictional documents.
- NVIDIA hosted trial latency and reliability do not predict a dedicated NIM deployment.
- The deterministic citation-entailment check is lexical and can reject valid paraphrases.
- The measured 500-character winner has not been promoted to the runtime index.
- The expanded financial regression still lacks Full@5 with NVIDIA embeddings, which reinforces the decision not to migrate embeddings.

## Closure answers

1. The best measured chunking configuration is 500 characters with 120-character overlap.
2. NVIDIA embeddings do not improve matched retrieval quality enough to replace Gemini.
3. NVIDIA neural reranking improves early rank position but reduces Recall@5 and Full@5, so it is not justified.
4. NVIDIA generation preserves safe refusal but does not preserve supported-answer correctness under the strict citation gate.
5. NVIDIA planning does not improve downstream retrieval and adds material latency.
6. GPU-backed inference benefits raw embedding and neural-reranking latency.
7. GPU acceleration is unnecessary for Chroma retrieval, RRF, BM25, validation, authorization, and reliability controls at this corpus size.
8. The final recommended architecture retains Gemini planning, Gemini embeddings, Chroma, RRF, local hybrid reranking, Gemini generation, and deterministic citation validation.
9. Remaining limitations are provider quota incompleteness, small corpus scale, a 14-case generation sample, hosted-trial generalizability, and a pending controlled 500-character reindex.
"""
    if "\u2014" in report:
        raise RuntimeError("Report contains a prohibited em dash.")
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "results": str(RESULT_PATH), "traces": str(TRACE_PATH)}, indent=2))


if __name__ == "__main__":
    main()
