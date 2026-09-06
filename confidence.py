"""Retrieval confidence signals and offline threshold calibration."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

from retrieval import Candidate


@dataclass(frozen=True)
class ConfidenceSignals:
    """Inspectable signals only, with no production abstention decision."""

    top1_vector_similarity: float
    top1_top2_margin: float
    average_topk_similarity: float
    similarity_mean: float
    similarity_stddev: float
    similarity_min: float
    similarity_max: float
    top_reranker_score: float
    average_reranker_score: float
    unique_source_documents: int
    vector_reranker_rank_agreement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_confidence(
    candidates: list[Candidate],
    evidence: list[Candidate],
) -> ConfidenceSignals:
    """Measure score distribution, source diversity, and rank disagreement."""
    if not candidates:
        return ConfidenceSignals(*(0.0 for _ in range(9)), 0, 0.0)

    vector_order = sorted(candidates, key=lambda item: (item.distance, item.chunk_id))
    similarities = [item.vector_similarity for item in vector_order]
    reranker_scores = [item.bm25_score for item in evidence]
    top1 = similarities[0]
    top2 = similarities[1] if len(similarities) > 1 else top1

    return ConfidenceSignals(
        top1_vector_similarity=top1,
        top1_top2_margin=top1 - top2,
        average_topk_similarity=statistics.fmean(similarities[:5]),
        similarity_mean=statistics.fmean(similarities),
        similarity_stddev=statistics.pstdev(similarities),
        similarity_min=min(similarities),
        similarity_max=max(similarities),
        top_reranker_score=max(reranker_scores, default=0.0),
        average_reranker_score=(statistics.fmean(reranker_scores) if reranker_scores else 0.0),
        unique_source_documents=len(
            {str(item.document.metadata["source"]) for item in evidence}
        ),
        vector_reranker_rank_agreement=_rank_agreement(candidates, evidence),
    )


def _rank_agreement(candidates: list[Candidate], evidence: list[Candidate]) -> float:
    """Return normalized rank agreement over the final evidence intersection."""
    if len(evidence) < 2:
        return 1.0
    vector_ranks = {
        item.chunk_id: rank
        for rank, item in enumerate(
            sorted(candidates, key=lambda candidate: (candidate.distance, candidate.chunk_id)),
            start=1,
        )
    }
    rerank_ranks = {item.chunk_id: rank for rank, item in enumerate(evidence, start=1)}
    ids = [item.chunk_id for item in evidence if item.chunk_id in vector_ranks]
    observed = sum(abs(vector_ranks[item] - rerank_ranks[item]) for item in ids)
    worst = sum(abs((len(candidates) - index + 1) - index) for index in range(1, len(ids) + 1))
    return max(0.0, 1.0 - observed / max(1, worst))


def calibrate_thresholds(
    rows: list[dict[str, Any]],
    signal_names: list[str],
) -> dict[str, Any]:
    """Evaluate every observed cutoff without selecting a production threshold."""
    results: dict[str, Any] = {}
    for signal in signal_names:
        values = sorted({float(row[signal]) for row in rows})
        trials = []
        for threshold in values:
            predictions = [float(row[signal]) < threshold for row in rows]
            actual = [bool(row["unsupported"]) for row in rows]
            tp = sum(predicted and label for predicted, label in zip(predictions, actual, strict=True))
            fp = sum(predicted and not label for predicted, label in zip(predictions, actual, strict=True))
            fn = sum(not predicted and label for predicted, label in zip(predictions, actual, strict=True))
            tn = sum(not predicted and not label for predicted, label in zip(predictions, actual, strict=True))
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            trials.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_answer_rate": fn / max(1, tp + fn),
                    "false_refusal_rate": fp / max(1, fp + tn),
                }
            )
        results[signal] = max(
            trials,
            key=lambda trial: (trial["f1"], trial["precision"], -trial["false_refusal_rate"]),
        )
    return results


def add_hybrid_confidence(rows: list[dict[str, Any]]) -> None:
    """Add a fixed exploratory composite after dataset-level normalization."""
    components = {
        "top1_vector_similarity": 0.35,
        "average_topk_similarity": 0.25,
        "top_reranker_score": 0.25,
        "unique_source_documents": 0.10,
        "vector_reranker_rank_agreement": 0.05,
    }
    bounds = {
        name: (
            min(float(row[name]) for row in rows),
            max(float(row[name]) for row in rows),
        )
        for name in components
    }
    for row in rows:
        score = 0.0
        for name, weight in components.items():
            lower, upper = bounds[name]
            normalized = 1.0 if upper == lower else (float(row[name]) - lower) / (upper - lower)
            score += weight * normalized
        row["hybrid_confidence"] = score
