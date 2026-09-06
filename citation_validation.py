"""Runtime validation for grounded numbered citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from answer_evaluation import evaluate_claim
from rag import INSUFFICIENT_ANSWER
from retrieval import Candidate


@dataclass(frozen=True)
class CitationValidation:
    """Inspectable citation-validation result."""

    valid: bool
    citations: list[int]
    errors: list[str]
    claims_checked: int

    def correction_prompt(self) -> str:
        return "\n".join(f"- {error}" for error in self.errors)


def validate_answer_citations(
    answer: str,
    evidence: list[Candidate],
    question: str = "",
) -> CitationValidation:
    """Reject missing, out-of-range, malformed, or unsupported citations."""
    if answer.strip() == INSUFFICIENT_ANSWER:
        return CitationValidation(True, [], [], 0)
    if not answer.strip():
        return CitationValidation(False, [], ["The model returned an empty answer."], 0)

    errors = []
    citations: list[int] = []
    claims = _extract_runtime_claims(answer)
    selected_sources = {
        str(candidate.document.metadata.get("source", "")) for candidate in evidence
    }
    mentioned_sources = set(re.findall(r"[A-Za-z0-9_\-]+\.md", answer))
    unsupported_sources = mentioned_sources - selected_sources
    if unsupported_sources:
        errors.append(
            "The answer names unselected sources: " + ", ".join(sorted(unsupported_sources))
        )
    selected_chunk_ids = {candidate.chunk_id for candidate in evidence}
    mentioned_chunk_ids = set(re.findall(r"\b[a-f0-9]{64}\b", answer.casefold()))
    unselected_chunk_ids = mentioned_chunk_ids - selected_chunk_ids
    if unselected_chunk_ids:
        errors.append(
            "The answer names unselected chunk IDs: "
            + ", ".join(sorted(unselected_chunk_ids))
        )

    for claim in claims:
        check = evaluate_claim(claim, evidence, question=question)
        claim_citations = check["citations"]
        citations.extend(claim_citations)
        if not claim_citations:
            errors.append(f"Factual claim has no citation: {claim}")
            continue
        invalid = [value for value in claim_citations if not 1 <= value <= len(evidence)]
        if invalid:
            errors.append(
                f"Claim cites nonexistent evidence numbers {invalid}: {claim}"
            )
            continue
        if not check["entailed"]:
            errors.append(f"Cited evidence does not support the claim: {claim}")

    numeric_markers = re.findall(r"\[[\d,\s]+\]", answer)
    all_brackets = re.findall(r"\[[^\]]+\]", answer)
    malformed = [marker for marker in all_brackets if marker not in numeric_markers]
    if malformed:
        errors.append("Malformed citation markers: " + ", ".join(malformed))

    return CitationValidation(
        valid=not errors,
        citations=sorted(set(citations)),
        errors=errors,
        claims_checked=len(claims),
    )


def _extract_runtime_claims(answer: str) -> list[str]:
    """Split runtime claims while retaining factual colon-ended bridge lines."""
    retained_lines = []
    for raw_line in answer.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped)
        is_markdown_heading = normalized.startswith("**") and normalized.endswith("**")
        normalized = re.sub(r"[*_`]+", "", normalized).strip()
        if is_markdown_heading:
            continue
        if normalized.endswith(":") and not re.search(r"\[\d", normalized):
            factual_verb = re.search(
                r"\b(?:is|are|must|requires?|applies|apply|includes?|contains?|owns?)\b",
                normalized.casefold(),
            )
            if not factual_verb:
                continue
        if normalized.casefold().startswith("based on the provided"):
            continue
        retained_lines.append(normalized)
    joined = "\n".join(retained_lines)
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", joined)
        if item.strip() and re.search(r"[A-Za-z]", item)
    ]
