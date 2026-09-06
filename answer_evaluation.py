"""Deterministic answer, refusal, and citation-faithfulness evaluation."""

from __future__ import annotations

import re
from typing import Any

from rag import INSUFFICIENT_ANSWER
from retrieval import Candidate


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for", "from",
    "has", "in", "is", "it", "must", "of", "on", "or", "should", "that", "the",
    "their", "this", "to", "when", "with",
}


def is_refusal(answer: str) -> bool:
    """Require the exact evidence-only refusal contract."""
    return answer.strip() == INSUFFICIENT_ANSWER


def evaluate_answer(
    answer: str,
    evidence: list[Candidate],
    expectation: dict[str, Any] | None,
    unsupported: bool,
    question: str = "",
) -> dict[str, Any]:
    """Evaluate deterministic expected facts and claim-level citation support."""
    refused = is_refusal(answer)
    claims = [] if refused else extract_claims(answer)
    claim_checks = [evaluate_claim(claim, evidence, question=question) for claim in claims]
    terms = expectation.get("required_terms", []) if expectation else []
    required_sources = expectation.get("required_sources", []) if expectation else []
    answer_folded = answer.casefold()
    term_checks = [
        any(str(alternative).casefold() in answer_folded for alternative in alternatives)
        for alternatives in terms
    ]
    selected_sources = {str(item.document.metadata["source"]) for item in evidence}

    citation_claims = [claim for claim in claim_checks if claim["citations"]]
    valid_citations = [
        citation
        for claim in claim_checks
        for citation in claim["citations"]
        if 1 <= citation <= len(evidence)
    ]
    total_citations = sum(len(claim["citations"]) for claim in claim_checks)

    return {
        "refused": refused,
        "correct_refusal": unsupported and refused,
        "incorrect_answer": unsupported and not refused,
        "incorrect_refusal": not unsupported and refused,
        "answer_correct": (not unsupported and not refused and all(term_checks)),
        "required_term_checks": term_checks,
        "evidence_complete": set(required_sources).issubset(selected_sources),
        "citation_correctness": (
            len(valid_citations) / total_citations if total_citations else (1.0 if refused else 0.0)
        ),
        "citation_entailment": (
            sum(bool(claim["entailed"]) for claim in citation_claims) / len(citation_claims)
            if citation_claims else (1.0 if refused else 0.0)
        ),
        "unsupported_claim_count": sum(not claim["entailed"] for claim in claim_checks),
        "claims": claim_checks,
    }


def extract_claims(answer: str) -> list[str]:
    """Split prose and bullets into factual statements while preserving markers."""
    retained_lines = []
    for raw_line in answer.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized_line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped)
        normalized_line = re.sub(r"[*_`]+", "", normalized_line).strip()
        if normalized_line.endswith(":") and not re.search(r"\[\d", normalized_line):
            continue
        if normalized_line.casefold().startswith("based on the provided"):
            continue
        retained_lines.append(normalized_line)
    normalized = "\n".join(retained_lines)
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if item.strip() and re.search(r"[A-Za-z]", item)
    ]


def evaluate_claim(
    claim: str,
    evidence: list[Candidate],
    question: str = "",
) -> dict[str, Any]:
    """Check whether referenced chunks deterministically support one claim."""
    citations = [int(value) for marker in re.findall(r"\[([\d,\s]+)\]", claim) for value in re.findall(r"\d+", marker)]
    valid = [value for value in citations if 1 <= value <= len(evidence)]
    cited_text = " ".join(evidence[value - 1].document.page_content for value in valid).casefold()
    clean_claim = re.sub(r"\[[\d,\s]+\]", "", claim).casefold()
    claim_numbers = _numbers(clean_claim)
    evidence_numbers = _numbers(cited_text)
    question_numbers = _numbers(question.casefold())
    number_support = claim_numbers.issubset(evidence_numbers | question_numbers)
    tokens = {
        token for token in re.findall(r"[a-z€]+", clean_claim)
        if len(token) > 2 and token not in STOPWORDS
    }
    evidence_tokens = set(re.findall(r"[a-z€]+", cited_text))
    lexical_coverage = len(tokens & evidence_tokens) / max(1, len(tokens))
    entailed = bool(valid) and number_support and lexical_coverage >= 0.35
    return {
        "claim": claim,
        "citations": citations,
        "valid_citations": valid,
        "number_support": number_support,
        "lexical_coverage": lexical_coverage,
        "entailed": entailed,
    }


def _numbers(text: str) -> set[str]:
    """Extract normalized numeric values without trailing sentence punctuation."""
    return {
        value.rstrip(".,")
        for value in re.findall(r"(?:eur\s*)?\d[\d,.]*", text)
    }
