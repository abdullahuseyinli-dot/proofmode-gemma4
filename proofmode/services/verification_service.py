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
    evidence_quote: str = ""

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

    extracted_claims = extract_claims(answer)
    cited_source_ids = {
        source_id
        for claim in extracted_claims
        for source_id in extract_citation_ids(claim)
    }
    source_blocks: list[str] = []
    for index, source in enumerate(_source_sequence(sources), 1):
        source_id = _source_value(source, "source_id", f"S{index}").upper() or f"S{index}"
        if source_id not in cited_source_ids:
            continue
        title = _source_value(source, "title", "Untitled")
        evidence = _source_value(source, "text") or _source_value(source, "snippet")
        source_blocks.append(f"[{source_id}] {title}\n{evidence[:3_500]}")

    claim_blocks = [f"C{index}: {claim}" for index, claim in enumerate(extracted_claims, 1)]
    expected_ids = ", ".join(f"C{index}" for index in range(1, len(extracted_claims) + 1)) or "none"
    return f"""You are a strict evidence verifier, not an answer writer.

Judge each claim using ONLY the evidence below. Do not use memory or infer missing facts.
Labels:
- supported: the cited evidence directly entails the whole claim
- unsupported: the evidence contradicts it or does not substantiate a material assertion
- uncertain: evidence is ambiguous, incomplete, or requires an inference

Return JSON only with this exact shape:
{{"claims":[{{"claim_id":"C1","label":"supported|unsupported|uncertain","evidence_source_ids":["S1"],"evidence_quote":"exact contiguous verbatim substring from one cited evidence source","reason":"short evidence-based reason"}}]}}

Every claim must appear exactly once. For a supported label, evidence_quote is
required and must be copied verbatim from one of evidence_source_ids. Treat
negation, numbers, dates, units, and material qualifiers as part of the claim.
Never label a claim supported when the reason says the evidence is absent,
insufficient, contradictory, or uncertain.

Completeness contract: return exactly {len(extracted_claims)} row(s), one for each
of these claim IDs and no others: {expected_ids}. Do not omit unsupported or
uncertain claims.

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


_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|lacks?|lacking|absent)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)
_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(
    r"(?<!\w)(?:%(?!\w)|(?:percent(?:age)?|milliseconds?|ms|seconds?|secs?|minutes?|mins?|"
    r"hours?|hrs?|days?|weeks?|months?|years?|mm|cm|km|meters?|metres?|inches?|"
    r"feet|ft|yards?|miles?|mg|grams?|kg|kilograms?|oz|ounces?|lb|lbs|pounds?|"
    r"kb|mb|gb|tb|bytes?|kbps|mbps|gbps|hz|khz|mhz|ghz|°c|°f|celsius|fahrenheit|"
    r"usd|gbp|eur|dollars?|pounds?|euros?)\b)",
    re.IGNORECASE,
)
_REASON_CONFLICT_RE = re.compile(
    r"(?:\b(?:does\s+not|do\s+not|did\s+not|doesn't|don't|didn't|cannot|can't|fails?\s+to)\s+"
    r"(?:\w+\s+){0,3}(?:support|supports|supported|substantiate|substantiates|substantiate[sd]|"
    r"establish|establishes|established|confirm|confirms|confirmed|show|shows|shown|"
    r"state|states|stated|prove|proves|proved|entail|entails|entailed)\b|"
    r"\bnot\s+(?:\w+\s+){0,3}(?:support|supported|substantiated|established|confirmed|"
    r"shown|stated|proven|entailed)\b|"
    r"\b(?:unsupported|uncertain|ambiguous|incomplete|insufficient|contradicts?|"
    r"contradictory|no\s+evidence|not\s+enough\s+evidence|unverified)\b)",
    re.IGNORECASE,
)
_OPPOSITE_TERM_PAIRS = (
    ({"increase", "increases", "increased", "increasing", "rise", "rises", "rose"},
     {"decrease", "decreases", "decreased", "decreasing", "fall", "falls", "fell"}),
    ({"improve", "improves", "improved", "improving", "better"},
     {"worsen", "worsens", "worsened", "worsening", "worse"}),
    ({"higher", "above"}, {"lower", "below"}),
    ({"before", "earlier"}, {"after", "later"}),
    ({"positive", "true"}, {"negative", "false"}),
    ({"safe"}, {"unsafe"}),
)
_MAX_VERIFIER_REASON_CHARS = 600
_MAX_EVIDENCE_QUOTE_CHARS = 2_000


def _source_evidence(source: ResearchSource | Mapping[str, Any]) -> str:
    return _source_value(source, "text") or _source_value(source, "snippet")


def _normalised_numbers(text: str) -> set[str]:
    return {
        match.group(0).lower().replace(",", "").lstrip("+").rstrip("%")
        for match in _NUMBER_RE.finditer(text)
    }


def _normalised_units(text: str) -> set[str]:
    aliases = {
        "%": "percent",
        "percentage": "percent",
        "milliseconds": "ms",
        "millisecond": "ms",
        "seconds": "second",
        "second": "second",
        "secs": "second",
        "sec": "second",
        "minutes": "minute",
        "minute": "minute",
        "mins": "minute",
        "min": "minute",
        "hours": "hour",
        "hour": "hour",
        "hrs": "hour",
        "hr": "hour",
        "days": "day",
        "weeks": "week",
        "months": "month",
        "years": "year",
        "meters": "meter",
        "metres": "meter",
        "metre": "meter",
        "inches": "inch",
        "yards": "yard",
        "miles": "mile",
        "grams": "gram",
        "kilograms": "kg",
        "kilogram": "kg",
        "ounces": "ounce",
        "pounds": "lb",
        "pound": "lb",
        "lbs": "lb",
        "bytes": "byte",
        "dollars": "dollar",
        "euros": "euro",
    }
    return {aliases.get(match.group(0).lower(), match.group(0).lower()) for match in _UNIT_RE.finditer(text)}


def _month_tokens(text: str) -> set[str]:
    return {match.group(0).lower()[:3] for match in _MONTH_RE.finditer(text)}


def _date_tokens(text: str) -> set[str]:
    return {match.group(0) for match in _DATE_RE.finditer(text)}


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _support_conflict_reason(claim: str, evidence_quote: str, verifier_reason: str) -> str | None:
    """Return a fail-closed reason for common deterministic contradictions."""

    if _REASON_CONFLICT_RE.search(verifier_reason):
        return "Verifier label conflicts with its reason, which does not establish support."

    claim_body = _CITATION_GROUP_RE.sub("", claim)
    if bool(_NEGATION_RE.search(claim_body)) != bool(_NEGATION_RE.search(evidence_quote)):
        return "Claim and quoted evidence have conflicting negation."

    claim_dates = _date_tokens(claim_body)
    quote_dates = _date_tokens(evidence_quote)
    if claim_dates and not claim_dates.issubset(quote_dates):
        return "A date in the claim does not match the quoted evidence."

    claim_numbers = _normalised_numbers(claim_body)
    quote_numbers = _normalised_numbers(evidence_quote)
    if claim_numbers and not claim_numbers.issubset(quote_numbers):
        return "A number or date in the claim is absent from the quoted evidence."

    claim_units = _normalised_units(claim_body)
    quote_units = _normalised_units(evidence_quote)
    if claim_units and not claim_units.issubset(quote_units):
        return "A unit in the claim does not match the quoted evidence."

    claim_months = _month_tokens(claim_body)
    quote_months = _month_tokens(evidence_quote)
    if claim_months and not claim_months.issubset(quote_months):
        return "A date in the claim does not match the quoted evidence."

    claim_words = _word_tokens(claim_body)
    quote_words = _word_tokens(evidence_quote)
    for left, right in _OPPOSITE_TERM_PAIRS:
        if claim_words & left and quote_words & right and not quote_words & left:
            return "Claim and quoted evidence use contradictory directional terms."
        if claim_words & right and quote_words & left and not quote_words & right:
            return "Claim and quoted evidence use contradictory directional terms."
    return None


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

    # A citation check cannot establish semantic entailment. Lexical overlap is
    # especially unsafe for negated claims ("improves" vs "does not improve")
    # and transposed numbers, so deterministic mode always requires review.
    return ClaimAssessment(
        claim_id,
        claim,
        "uncertain",
        cited,
        valid_cited,
        "Citation is valid, but semantic support requires a complete verifier pass or human review.",
    )


def _validated_verifier_assessments(
    rows: Any,
    deterministic: Mapping[str, ClaimAssessment],
    source_by_id: Mapping[str, ResearchSource | Mapping[str, Any]],
) -> dict[str, ClaimAssessment]:
    """Parse one complete verifier judgment per claim or reject the pass."""

    if not isinstance(rows, list) or len(rows) != len(deterministic):
        raise ValueError("Verifier must return exactly one row per claim")

    parsed: dict[str, ClaimAssessment] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every verifier row must be an object")
        claim_id = str(row.get("claim_id", "")).upper()
        if claim_id not in deterministic or claim_id in parsed:
            raise ValueError("Verifier returned an unknown or duplicate claim identifier")

        raw_label = row.get("label")
        if not isinstance(raw_label, str) or raw_label.lower() not in {"supported", "unsupported", "uncertain"}:
            raise ValueError("Verifier returned an invalid label")
        label = raw_label.lower()

        raw_ids = row.get("evidence_source_ids")
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            raise ValueError("evidence_source_ids must be a string list")
        evidence_ids = tuple(item.upper() for item in raw_ids)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Verifier returned duplicate evidence identifiers")

        base = deterministic[claim_id]
        if any(source_id not in source_by_id or source_id not in base.cited_source_ids for source_id in evidence_ids):
            raise ValueError("Verifier cited evidence not cited inline by this claim")

        reason_value = row.get("reason")
        if not isinstance(reason_value, str) or not reason_value.strip():
            raise ValueError("Verifier must provide a non-empty reason")
        reason = reason_value.strip()

        quote_value = row.get("evidence_quote", "")
        if not isinstance(quote_value, str):
            raise ValueError("evidence_quote must be a string")
        evidence_quote = quote_value.strip()

        # Never truncate before semantic checks. A disclaimer or negation in a
        # discarded suffix could otherwise turn an explicitly conflicted row
        # into a supported one. Oversized fields violate the terse verifier
        # contract and invalidate the complete pass rather than being repaired.
        if len(reason) > _MAX_VERIFIER_REASON_CHARS:
            raise ValueError("Verifier reason exceeds the safe length limit")
        if len(evidence_quote) > _MAX_EVIDENCE_QUOTE_CHARS:
            raise ValueError("Verifier evidence_quote exceeds the safe length limit")

        if label == "supported":
            if not evidence_ids or not evidence_quote:
                label = "uncertain"
                reason = "Supported verdict lacked a cited source or verbatim evidence quote."
            else:
                quote_sources = tuple(
                    source_id
                    for source_id in evidence_ids
                    if evidence_quote in _source_evidence(source_by_id[source_id])
                )
                if not quote_sources:
                    label = "uncertain"
                    reason = "Verifier quote is not a verbatim substring of its cited evidence source."
                else:
                    conflict = _support_conflict_reason(base.claim, evidence_quote, reason)
                    if conflict:
                        label = "uncertain"
                        reason = conflict

        if base.label == "unsupported":
            label = "unsupported"
            reason = base.reason

        parsed[claim_id] = ClaimAssessment(
            claim_id=claim_id,
            claim=base.claim,
            label=label,
            cited_source_ids=base.cited_source_ids,
            evidence_source_ids=evidence_ids,
            reason=reason,
            evidence_quote=evidence_quote,
        )

    if set(parsed) != set(deterministic):
        raise ValueError("Verifier omitted one or more claims")
    return parsed


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
            if not isinstance(parsed, Mapping) or set(parsed) != {"claims"}:
                raise ValueError("Verifier response must contain only a claims list")
            assessments = _validated_verifier_assessments(
                parsed["claims"], deterministic, source_by_id
            )
            used_llm = True
        except Exception:
            # Failing closed is intentional.  UI callers can show the uncertain
            # verdict and invite retry/human review rather than leaking an answer.
            used_llm = False

    ordered = tuple(assessments[f"C{index}"] for index in range(1, len(claims) + 1))
    unsupported_count = sum(item.label == "unsupported" for item in ordered)
    uncertain_count = sum(item.label == "uncertain" for item in ordered)
    safe_to_show = (
        bool(ordered)
        and used_llm
        and citation_validation.valid
        and len(ordered) == len(claims)
        and all(item.label == "supported" for item in ordered)
    )
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


def select_supported_excerpt(
    answer: str,
    sources: ResearchPack | Sequence[ResearchSource | Mapping[str, Any]],
    report: VerificationReport,
) -> tuple[str, VerificationReport] | None:
    """Copy only semantically verified claims into a safe partial answer.

    This is a deterministic display fallback, not another verification pass. It
    refuses to operate unless the complete semantic verifier ran successfully and
    the original answer passed citation validation. No prose is generated: claim
    text is copied verbatim from the verified report in its original order.
    """

    if not report.used_llm_verifier or not report.citation_validation.valid:
        return None

    original_claims = extract_claims(answer)
    if len(original_claims) != len(report.claims) or any(
        assessment.claim_id != f"C{index}" or assessment.claim != claim
        for index, (assessment, claim) in enumerate(
            zip(report.claims, original_claims),
            1,
        )
    ):
        return None

    # Re-run deterministic citation validation so a report accidentally paired
    # with a different answer cannot authorize selective display.
    original_validation = validate_citations(answer, sources, require_every_claim=True)
    if not original_validation.valid:
        return None

    source_by_id: dict[str, ResearchSource | Mapping[str, Any]] = {}
    for index, source in enumerate(_source_sequence(sources), 1):
        source_id = _source_value(source, "source_id", f"S{index}").upper() or f"S{index}"
        source_by_id[source_id] = source

    selected: list[ClaimAssessment] = []
    for assessment in report.claims:
        if assessment.label != "supported":
            continue
        if not assessment.evidence_quote or not assessment.evidence_source_ids:
            return None
        cited = extract_citation_ids(assessment.claim)
        if cited != assessment.cited_source_ids or any(
            source_id not in source_by_id or source_id not in cited
            for source_id in assessment.evidence_source_ids
        ):
            return None
        if not any(
            assessment.evidence_quote in _source_evidence(source_by_id[source_id])
            for source_id in assessment.evidence_source_ids
        ):
            return None
        if _support_conflict_reason(
            assessment.claim,
            assessment.evidence_quote,
            assessment.reason,
        ):
            return None
        selected.append(assessment)

    if not selected:
        return None

    filtered_answer = "\n\n".join(assessment.claim for assessment in selected)
    filtered_claims = extract_claims(filtered_answer)
    if tuple(assessment.claim for assessment in selected) != filtered_claims:
        return None
    filtered_validation = validate_citations(
        filtered_answer,
        sources,
        require_every_claim=True,
    )
    if not filtered_validation.valid or any(
        assessment.label != "supported" or not assessment.evidence_quote
        for assessment in selected
    ):
        return None

    filtered_report = VerificationReport(
        citation_validation=filtered_validation,
        claims=tuple(selected),
        used_llm_verifier=True,
        safe_to_show=True,
        summary=(
            f"Narrowed partial answer: {len(selected)} of {len(report.claims)} "
            "claim(s) were directly supported by cited evidence."
        ),
    )
    return filtered_answer, filtered_report


__all__ = [
    "CitationValidation",
    "ClaimAssessment",
    "VerificationReport",
    "build_verification_prompt",
    "extract_citation_ids",
    "extract_claims",
    "select_supported_excerpt",
    "validate_citations",
    "verify_answer",
]
