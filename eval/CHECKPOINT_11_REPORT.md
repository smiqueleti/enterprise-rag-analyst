# Checkpoint 11 - Benchmark and Evidence Closure

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
| 300 | 66 | 0.837 | 0.956 | 0.980 | 0.960 | 740.2 | 4.05 |
| 500 | 31 | 0.853 | 0.984 | 1.000 | 0.964 | 380.8 | 3.69 |
| 800 | 16 | 0.837 | 0.976 | 0.992 | 0.960 | 277.6 | 3.73 |
| 1200 | 12 | 0.833 | 0.976 | 1.000 | 0.952 | 231.4 | 3.96 |

The 500-character configuration is the measured winner because it reached Recall@5 1.000 and Full@5 1.000 while posting MRR 0.964.
The current 800-character configuration reached Recall@5 0.992, Full@5 0.976, and MRR 0.960.
This decision is recorded but not applied to the runtime index because doing so without a complete reindex would desynchronize code and stored vectors.

## Expanded retrieval quality by category

These results use NVIDIA embeddings, 800-character chunks, and the current local hybrid reranker over 42 supported cases.

| Category | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| ambiguous | 7 | 0.857 | 1.000 | 1.000 | 0.929 |
| authorization | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 7 | 0.929 | 1.000 | 1.000 | 1.000 |
| cross_document | 7 | 0.381 | 0.857 | 0.952 | 0.905 |
| direct | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrased | 7 | 0.857 | 1.000 | 1.000 | 0.929 |

Unsupported cases are excluded from retrieval recall because they intentionally have no relevant evidence.

## Embedding comparison

The matched comparison uses the 19 legacy cases with complete accepted query-plan caches.
Gemini achieved Recall@1 0.630, Recall@3 0.796, Recall@5 1.000, and MRR 0.944.
NVIDIA achieved Recall@1 0.463, Recall@3 0.815, Recall@5 0.963, and MRR 0.833.
NVIDIA returned 2,048-dimensional embeddings and query embedding p50 was 81.1 ms on one-query calls.
Gemini/current wins because the NVIDIA quality regression at Recall@5 and MRR outweighs its latency advantage.

## Neural reranking comparison

Both rerankers received the exact same 12-item candidate pool for every expanded case.
The current hybrid reranker achieved Recall@1 0.837, Recall@3 0.976, Recall@5 0.992, MRR 0.960, and Full@5 0.976.
The NVIDIA neural reranker achieved Recall@1 0.869, Recall@3 0.944, Recall@5 0.984, MRR 0.988, and Full@5 0.952.
NVIDIA reranking latency was p50 138.2 ms and p95 230.1 ms.
NVIDIA improved early ranking but reduced evidence completeness, so the local hybrid reranker remains selected.

On the matched legacy financial-approval regression, NVIDIA moved the financial chunk from rank 4 to rank 3 but moved the priority-notification chunk from rank 2 to rank 4.
On the expanded 800-character NVIDIA-embedding run, neither reranker achieved Full@5 for that compound case.
The dedicated priority-notification conversational case remained at Recall@5 1.000 for both rerankers.

## Generation comparison

The generation experiment used the same 14 preselected cases, with exactly two cases per category.
Gemini completed only 1 of 14 cases because one request returned 503 and the remaining requests hit free-tier quota limits.
That live sample is not sufficient for a new Gemini quality estimate, so the accepted current baseline remains controlling.
NVIDIA completed 13 of 14 cases, with one provider 503.
NVIDIA supported-answer correctness was 9.1%.
NVIDIA refusal precision was 16.7%, refusal recall was 100.0%, and citation entailment among surviving factual answers was 100.0%.
The final invalid-citation rate was 0.0%, but 11 of 14 cases required repair and most unsupported drafts became safe refusals after validation.
NVIDIA TTFT p50 was 1982.6 ms, TTFT p95 was 32196.0 ms, generation p50 was 4370.8 ms, and generation p95 was 52062.9 ms.
NVIDIA generation does not preserve the current answer-correctness gate and is not selected.

### NVIDIA generation quality by category

| Category | Completed cases | Answer correctness | Citation entailment | Refusal precision | Refusal recall |
|---|---:|---:|---:|---:|---:|
| ambiguous | 2 | 0.0% | 100.0% | 0.0% | not measured |
| authorization | 2 | 0.0% | 100.0% | 0.0% | not measured |
| conversational | 2 | 0.0% | 100.0% | 0.0% | not measured |
| cross_document | 2 | 0.0% | 100.0% | 0.0% | not measured |
| direct | 1 | 0.0% | 100.0% | 0.0% | not measured |
| paraphrased | 2 | 50.0% | 100.0% | 0.0% | not measured |
| unsupported | 2 | not measured | not measured | 100.0% | 100.0% |

Each category has two selected cases, except `direct`, where one NVIDIA provider 503 left one completed case.
Metrics marked not measured have no applicable supported or unsupported case in that category.

## Planner comparison

The NVIDIA planner completed all 50 cases with no malformed plans and no duplicate final atomic queries.
Its downstream Recall@5 was 0.948, MRR was 0.772, planning p50 was 6283.8 ms, and planning p95 was 17756.9 ms.
Cross-document Recall@5 was 0.833.
Gemini completed zero live planner cases because of external-provider quota exhaustion, so a direct live planner delta cannot be claimed.
The current planner remains selected because NVIDIA underperformed the fixed-query retrieval reference and added substantial latency.

## Final component matrix

| Component | Gemini/current | NVIDIA | Winner | Decision basis |
|---|---|---|---|---|
| Embeddings | Recall@5 1.000, MRR 0.944 | Recall@5 0.963, MRR 0.833 | Gemini/current | NVIDIA latency is attractive, but matched quality regressed. |
| Reranking | Recall@5 0.992, Full@5 0.976 | Recall@5 0.984, Full@5 0.952 | Current hybrid | Neural ranking improved rank 1 but lost complete evidence and added a hosted call. |
| Generation | Accepted quality baseline remains 1.000; live comparison quota-limited | Correctness 9.1%; repair rate 78.6% | Gemini/current retained | NVIDIA failed the grounded-answer usability gate. |
| Planning | Live comparison quota-limited | Recall@5 0.948; p95 17756.9 ms | Gemini/current retained | NVIDIA was slower and weaker than the fixed-query reference. |

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
