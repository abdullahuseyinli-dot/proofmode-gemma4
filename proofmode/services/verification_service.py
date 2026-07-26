"""Citation and claim verification for evidence-grounded Gemma responses.

Citation syntax is validated deterministically.  Claim support can then be judged
by an injected LLM callback (normally a second, low-temperature Gemma pass).  The
callback cannot introduce new evidence: ``supported`` is accepted only when the
claim cites source identifiers present in the supplied research pack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable, Mapping, Sequence

from .research_service import ResearchPack, ResearchSource


_CITATION_GROUP_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]", re.IGNORECASE)
_ANY_SOURCE_BRACKET_RE = re.compile(r"\[S[^\]]*\]", re.IGNORECASE)
_SINGLE_ID_RE = re.compile(r"S\d+", re.IGNORECASE)
_SOURCE_LIST_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?sources\s+used\s*:?.*$")


@dataclass(frozen=True)
class CitationValidation:
    """Purely programmatic checks over inline ``[S#]`` citations."""

    valid: bool
    known_source_ids: tuple[str, ...]
    used_source_ids: tuple[str, ...]
    unknown_source_ids: tuple[str, ...]
    malformed_citations: tuple[str, ...]
    uncited_claims: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimAssessment:
    """Evidence verdict for a single externally verifiable claim."""

    claim_id: str
    claim: str
    label: str
    cited_source_ids: tuple[str, ...]
    evidence_source_ids: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    """Display gate and inspectable details for one generated answer."""

    citation_validation: CitationValidation
    claims: tuple[ClaimAssessment, ...]
    used_llm_verifier: bool
    safe_to_show: bool
    summary: str

    @property
    def supported_count(self) -> int:
        return sum(claim.label == "supported" for claim in self.claims)

    @property
    def unsupported_count(self) -> int:
        return sum(claim.label == "unsupported" for claim in self.claims)

    @property
    def uncertain_count(self) -> int:
        return sum(claim.label == "uncertain" for claim in self.claims)

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_validation": self.citation_validation.as_dict(),
            "claims": [claim.as_dict() for claim in self.claims],
            "used_llm_verifier": self.used_llm_verifier,
            "safe_to_show": self.safe_to_show,
            "summary": self.summary,
            "counts": {
                "supported": self.supported_count,
                "unsupported": self.unsupported_count,
                "uncertain": self.uncertain_count,
            },
        }


def _source_sequence(
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
) -> tuple[ResearchSource | Mapping[str, Any], ...]:
    if isinstance(sources, ResearchPack):
        return sources.sources
    return tuple(sources)


def _source_value(source: ResearchSource | Mapping[str, Any], name: str, default: str = "") -> str:
    if isinstance(source, Mapping):
        return str(source.get(name, default))
    return str(getattr(source, name, default))


def _known_source_ids(
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
) -> tuple[str, ...]:
    identifiers: list[str] = []
    for index, source in enumerate(_source_sequence(sources), 1):
        source_id = _source_value(source, "source_id", f"S{index}").upper() or f"S{index}"
        if source_id not in identifiers:
            identifiers.append(source_id)
    return tuple(identifiers)


def extract_citation_ids(text: str) -> tuple[str, ...]:
    """Extract unique source IDs from valid ``[S1]`` or ``[S1, S2]`` groups."""

    identifiers: list[str] = []
    for group in _CITATION_GROUP_RE.findall(text):
        for source_id in _SINGLE_ID_RE.findall(group):
            normalized = source_id.upper()
            if normalized not in identifiers:
                identifiers.append(normalized)
    return tuple(identifiers)


def _answer_body(answer: str) -> str:
    """Exclude the final source bibliography from claim/citation coverage checks."""

    match = _SOURCE_LIST_RE.search(answer)
    return answer[: match.start()] if match else answer


def extract_claims(answer: str) -> tuple[str, ...]:
    """Split prose and bullets into conservative, citation-preserving claims."""

    body = _answer_body(answer)
    candidates: list[str] = []
    for line in body.splitlines():
        clean_line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        if not clean_line or clean_line.startswith("#"):
            continue
        candidates.extend(re.split(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])", clean_line))

    claims: list[str] = []
    for candidate in candidates:
        claim = candidate.strip()
        plain = _CITATION_GROUP_RE.sub("", claim).strip()
        word_count = len(re.findall(r"\b[\w'-]+\b", plain))
        if word_count < 3 or plain.endswith("?"):
            continue
        # A colon-terminated line introduces the claims that follow (for
        # example "How to prevent leakage:") rather than asserting a standalone
        # externally verifiable fact. The individual bullets still require
        # citations and verification.
        if plain.endswith(":"):
            continue
        # Labels and UI headings are not externally verifiable assertions.
        if plain.lower().rstrip(":") in {"answer", "explanation", "sources", "sources used"}:
            continue
        claims.append(claim)
    return tuple(claims)


def validate_citations(
    answer: str,
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
    *,
    require_every_claim: bool = True,
) -> CitationValidation:
    """Reject unknown/malformed citations and optionally uncited prose claims."""

    known = _known_source_ids(sources)
    body = _answer_body(answer)
    used = extract_citation_ids(body)
    unknown = tuple(source_id for source_id in used if source_id not in known)

    valid_group_texts = {match.group(0) for match in _CITATION_GROUP_RE.finditer(body)}
    malformed = tuple(
        dict.fromkeys(
            match.group(0)
            for match in _ANY_SOURCE_BRACKET_RE.finditer(body)
            if match.group(0) not in valid_group_texts
        )
    )
    uncited = tuple(
        claim
        for claim in extract_claims(body)
        if not extract_citation_ids(claim) and not _ANY_SOURCE_BRACKET_RE.search(claim)
    ) if require_every_claim else ()

    return CitationValidation(
        valid=not unknown and not malformed and not uncited and (bool(used) or not extract_claims(body)),
        known_source_ids=known,
        used_source_ids=used,
        unknown_source_ids=unknown,
        malformed_citations=malformed,
        uncited_claims=uncited,
    )


def build_verification_prompt(
    answer: str,
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
) -> str:
    """Build a strict second-pass fact-check prompt for an injected Gemma call."""

    source_blocks: list[str] = []
    for index, source in enumerate(_source_sequence(sources), 1):
        source_id = _source_value(source, "source_id", f"S{index}").upper() or f"S{index}"
        title = _source_value(source, "title", "Untitled")
        evidence = _source_value(source, "text") or _source_value(source, "snippet")
        source_blocks.append(f"[{source_id}] {title}\n{evidence[:7_000]}")

    claim_blocks = [f"C{index}: {claim}" for index, claim in enumerate(extract_claims(answer), 1)]
    return f"""You are a strict evidence verifier, not an answer writer.

Judge each claim using ONLY the evidence below. Do not use memory or infer missing facts.
Labels:
- supported: the cited evidence directly entails the whole claim
- unsupported: the evidence contradicts it or does not substantiate a material assertion
- uncertain: evidence is ambiguous, incomplete, or requires an inference

Return JSON only with this exact shape:
{{"claims":[{{"claim_id":"C1","label":"supported|unsupported|uncertain","evidence_source_ids":["S1"],"reason":"short evidence-based reason"}}]}}

EVIDENCE
{chr(10).join(source_blocks) or '[No evidence supplied]'}

CLAIMS
{chr(10).join(claim_blocks) or '[No factual claims detected]'}
"""


def _callback_content(result: Any) -> Any:
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, str):
        content = result.strip()
    elif hasattr(result, "text"):
        content = str(result.text).strip()
    elif getattr(result, "choices", None):
        content = str(result.choices[0].message.content).strip()
    else:
        content = str(result).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _call_verifier(callback: Callable[..., Any], prompt: str) -> Any:
    try:
        return callback(prompt)
    except TypeError:
        return callback(prompt=prompt)


_STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "before", "being", "between",
    "could", "does", "from", "have", "into", "more", "most", "other", "over", "such", "than",
    "that", "their", "there", "these", "they", "this", "through", "under", "using", "very", "were",
    "what", "when", "where", "which", "while", "with", "would", "your",
}


def _content_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]{4,}", _CITATION_GROUP_RE.sub("", text).lower())
        if term not in _STOPWORDS
    }


def _deterministic_assessment(
    claim_id: str,
    claim: str,
    source_by_id: Mapping[str, ResearchSource | Mapping[str, Any]],
) -> ClaimAssessment:
    cited = extract_citation_ids(claim)
    valid_cited = tuple(source_id for source_id in cited if source_id in source_by_id)
    if not cited:
        return ClaimAssessment(
            claim_id, claim, "unsupported", (), (), "No inline evidence citation was provided."
        )
    if len(valid_cited) != len(cited):
        return ClaimAssessment(
            claim_id, claim, "unsupported", cited, valid_cited, "The claim cites an unknown source identifier."
        )

    claim_terms = _content_terms(claim)
    evidence = " ".join(
        (_source_value(source_by_id[source_id], "text") or _source_value(source_by_id[source_id], "snippet"))
        for source_id in valid_cited
    )
    overlap = claim_terms & _content_terms(evidence)
    if claim_terms and (len(overlap) >= 3 or len(overlap) / len(claim_terms) >= 0.5):
        label = "supported"
        reason = "The cited evidence has strong direct lexical coverage of the claim."
    else:
        label = "uncertain"
        reason = "Citation exists, but semantic support needs the LLM verifier or human review."
    return ClaimAssessment(claim_id, claim, label, cited, valid_cited, reason)


def verify_answer(
    answer: str,
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
    *,
    llm_callback: Callable[..., Any] | None = None,
    require_every_claim: bool = True,
) -> VerificationReport:
    """Validate citations and label claims supported/unsupported/uncertain.

    ``llm_callback`` receives one strict JSON prompt.  It should be a low-
    temperature local Gemma call.  If it fails or emits invalid JSON, verification
    degrades safely to deterministic citation/overlap checks instead of approving
    unverified prose.
    """

    citation_validation = validate_citations(
        answer, sources, require_every_claim=require_every_claim
    )
    source_sequence = _source_sequence(sources)
    source_by_id: dict[str, ResearchSource | Mapping[str, Any]] = {}
    for index, source in enumerate(source_sequence, 1):
        source_id = _source_value(source, "source_id", f"S{index}").upper() or f"S{index}"
        source_by_id[source_id] = source

    claims = extract_claims(answer)
    deterministic = {
        f"C{index}": _deterministic_assessment(f"C{index}", claim, source_by_id)
        for index, claim in enumerate(claims, 1)
    }
    assessments = dict(deterministic)
    used_llm = False

    if llm_callback is not None and claims:
        try:
            parsed = _callback_content(_call_verifier(llm_callback, build_verification_prompt(answer, sources)))
            rows = parsed.get("claims", []) if isinstance(parsed, Mapping) else []
            if not isinstance(rows, list):
                raise ValueError("Verifier response claims must be a list")
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                claim_id = str(row.get("claim_id", "")).upper()
                if claim_id not in deterministic:
                    continue
                proposed_label = str(row.get("label", "uncertain")).lower()
                if proposed_label not in {"supported", "unsupported", "uncertain"}:
                    proposed_label = "uncertain"
                raw_evidence_ids = row.get("evidence_source_ids", [])
                evidence_ids = tuple(
                    dict.fromkeys(str(item).upper() for item in raw_evidence_ids if str(item).upper() in source_by_id)
                ) if isinstance(raw_evidence_ids, list) else ()
                base = deterministic[claim_id]

                # The verifier may judge content, but it cannot repair an omitted or
                # invented citation. Supported means cited *and* verified.
                usable_evidence = tuple(source_id for source_id in evidence_ids if source_id in base.cited_source_ids)
                if proposed_label == "supported" and not usable_evidence:
                    proposed_label = "uncertain" if base.cited_source_ids else "unsupported"
                    reason = "Verifier found no known evidence source also cited inline by this claim."
                else:
                    reason = str(row.get("reason") or "Evidence verifier supplied no reason.")[:600]
                assessments[claim_id] = ClaimAssessment(
                    claim_id=claim_id,
                    claim=base.claim,
                    label=proposed_label,
                    cited_source_ids=base.cited_source_ids,
                    evidence_source_ids=usable_evidence or evidence_ids,
                    reason=reason,
                )
            used_llm = True
        except Exception:
            # Failing closed is intentional.  UI callers can show the uncertain
            # verdict and invite retry/human review rather than leaking an answer.
            used_llm = False

    ordered = tuple(assessments[f"C{index}"] for index in range(1, len(claims) + 1))
    unsupported_count = sum(item.label == "unsupported" for item in ordered)
    uncertain_count = sum(item.label == "uncertain" for item in ordered)
    safe_to_show = bool(ordered) and citation_validation.valid and not unsupported_count and not uncertain_count
    if not ordered:
        summary = "No factual claims were detected; nothing was evidence-verified."
    elif safe_to_show:
        summary = f"Verified {len(ordered)} claim(s) against cited evidence."
    else:
        summary = (
            f"Answer held for review: {unsupported_count} unsupported and "
            f"{uncertain_count} uncertain claim(s); citation_valid={citation_validation.valid}."
        )
    return VerificationReport(
        citation_validation=citation_validation,
        claims=ordered,
        used_llm_verifier=used_llm,
        safe_to_show=safe_to_show,
        summary=summary,
    )


__all__ = [
    "CitationValidation",
    "ClaimAssessment",
    "VerificationReport",
    "build_verification_prompt",
    "extract_citation_ids",
    "extract_claims",
    "validate_citations",
    "verify_answer",
]
