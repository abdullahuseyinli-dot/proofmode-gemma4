from __future__ import annotations

from email.message import Message
import json
import threading

import pytest

from proofmode.services.research_service import (
    ResearchPack,
    ResearchSource,
    UnsafeURLError,
    build_evidence_prompt,
    prepare_research_pack,
    safe_fetch_text,
    search_web,
)
from proofmode.services.verification_service import (
    build_verification_prompt,
    extract_citation_ids,
    select_supported_excerpt,
    validate_citations,
    verify_answer,
)


def _source(
    source_id: str,
    domain: str,
    text: str,
    *,
    title: str = "Evidence source",
) -> ResearchSource:
    return ResearchSource(
        source_id=source_id,
        title=title,
        url=f"https://{domain}/article",
        snippet=text[:160],
        domain=domain,
        text=text,
    )


def test_search_ranks_authoritative_sources_and_prepares_page_text_without_network() -> None:
    def fake_search(query: str, max_results: int):
        assert "retrieval practice" in query
        assert max_results >= 3
        return [
            {
                "title": "A forum opinion",
                "href": "https://www.reddit.com/r/study/example",
                "body": "A personal opinion about retrieval practice.",
            },
            {
                "title": "Official education evidence",
                "href": "https://education.gov.uk/retrieval",
                "body": "Official documentation about retrieval practice.",
            },
            {
                "title": "University experiment",
                "href": "https://example.edu/research/retrieval",
                "body": "A randomized retrieval practice experiment.",
            },
        ]

    fetched_urls: list[str] = []

    def fake_fetch(url: str, *, timeout: float) -> str:
        fetched_urls.append(url)
        assert timeout == 2.0
        return f"Extracted evidence from {url}"

    pack = prepare_research_pack(
        "retrieval practice evidence",
        max_results=3,
        search_client=fake_search,
        page_fetcher=fake_fetch,
        timeout=2.0,
    )

    assert pack.status == "online"
    assert [source.source_id for source in pack.sources] == ["S1", "S2", "S3"]
    assert pack.sources[0].domain == "education.gov.uk"
    assert pack.sources[-1].domain == "reddit.com"
    assert all(source.text.startswith("Extracted evidence") for source in pack.sources)
    assert len(fetched_urls) == 3
    assert pack.method == "preparation/research pack (no fine-tuning)"


def test_page_fetches_are_bounded_concurrent_ordered_and_failure_isolated() -> None:
    domains = [f"{letter}.edu" for letter in "abcdef"]

    def fake_search(query: str, max_results: int):
        assert max_results >= len(domains)
        return [
            {
                "title": f"Source {domain}",
                "href": f"https://{domain}/article",
                "body": f"Snippet from {domain}",
            }
            for domain in domains
        ]

    lock = threading.Lock()
    first_four_started = threading.Barrier(4, timeout=5)
    active = 0
    max_active = 0

    def slow_fetch(url: str, *, timeout: float) -> str:
        nonlocal active, max_active
        domain = url.split("/", 3)[2]
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            # The first four ranked sources cannot complete until four workers
            # are active, proving overlap without a tight elapsed-time assertion.
            if domain in domains[:4]:
                first_four_started.wait()
            if domain in {"b.edu", "e.edu"}:
                raise OSError("isolated source failure")
            return f"Extracted page from {domain}"
        finally:
            with lock:
                active -= 1

    pack = prepare_research_pack(
        "bounded concurrent evidence",
        max_results=6,
        search_client=fake_search,
        page_fetcher=slow_fetch,
        timeout=2.0,
    )

    assert max_active == 4
    assert [source.source_id for source in pack.sources] == [f"S{i}" for i in range(1, 7)]
    assert [source.domain for source in pack.sources] == domains
    assert [source.text == "" for source in pack.sources] == [False, True, False, False, True, False]
    assert pack.warnings == (
        "S2 could not be fetched; using search snippet (OSError).",
        "S5 could not be fetched; using search snippet (OSError).",
    )
    assert pack.status == "partial"


def test_search_fails_gracefully_offline() -> None:
    def offline_search(query: str, max_results: int):
        raise OSError("network disconnected")

    pack = search_web("spacing effect", search_client=offline_search)

    assert pack.is_offline
    assert pack.sources == ()
    assert "local material only" in pack.warnings[0]


class _FakeResponse:
    def __init__(self, html: bytes) -> None:
        self._html = html
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def geturl(self) -> str:
        return "https://example.org/article"

    def read(self, amount: int) -> bytes:
        return self._html[:amount]


def test_safe_fetch_bounds_and_extracts_html_with_injected_io() -> None:
    called: dict[str, object] = {}

    def fake_resolver(host: str, port: int):
        assert host == "example.org"
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    def fake_opener(request, *, timeout: float):
        called["url"] = request.full_url
        called["timeout"] = timeout
        return _FakeResponse(
            b"<html><head><script>steal()</script></head>"
            b"<body><main><h1>Spacing effect</h1><p>Review after a delay improves retention.</p>"
            b"</main></body></html>"
        )

    text = safe_fetch_text(
        "https://example.org/article",
        timeout=1.25,
        opener=fake_opener,
        resolver=fake_resolver,
    )

    assert called == {"url": "https://example.org/article", "timeout": 1.25}
    assert "Spacing effect" in text
    assert "improves retention" in text
    assert "steal()" not in text


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1:11434/v1/models",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_safe_fetch_rejects_non_public_targets_before_opening(url: str) -> None:
    opened = False

    def should_not_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("unsafe URL reached the opener")

    with pytest.raises(UnsafeURLError):
        safe_fetch_text(url, opener=should_not_open, resolver=None)
    assert opened is False


def test_evidence_prompt_is_explicitly_grounded_and_calls_it_preparation() -> None:
    source = _source(
        "S1",
        "example.edu",
        "Repeated retrieval improved delayed recall in this experiment.",
        title="Retrieval experiment",
    )
    prompt = build_evidence_prompt(
        "How should I revise?",
        ResearchPack(query="revision", sources=(source,), status="online"),
    )

    assert "preparation/research pack" in prompt
    assert "NOT training or fine-tuning" in prompt
    assert "[S1] Retrieval experiment" in prompt
    assert "immediately after every factual claim" in prompt
    assert "If the evidence is missing or insufficient" in prompt


def test_verifier_prompt_includes_only_cited_sources_and_bounds_their_evidence() -> None:
    long_evidence = "start-of-evidence " + ("x" * 4_000) + " beyond-the-bound"
    sources = (
        _source("S7", "used.example.edu", long_evidence, title="Used source"),
        _source(
            "S2",
            "unused.example.edu",
            "unused-source-secret",
            title="Unused source",
        ),
    )

    prompt = build_verification_prompt(
        "The measured effect was stable [S7].",
        sources,
    )
    evidence_section = prompt.split("EVIDENCE\n", 1)[1].split("\n\nCLAIMS\n", 1)[0]
    header, included_evidence = evidence_section.split("\n", 1)

    assert header == "[S7] Used source"
    assert included_evidence == long_evidence[:3_500]
    assert "beyond-the-bound" not in prompt
    assert "Unused source" not in prompt
    assert "unused-source-secret" not in prompt


def test_programmatic_citation_validation_detects_unknown_malformed_and_uncited() -> None:
    sources = (
        _source("S1", "example.edu", "The Earth orbits the Sun."),
        _source("S2", "science.gov", "One orbit takes approximately one year."),
    )
    answer = (
        "The Earth orbits the Sun [S1]. "
        "One orbit takes about a year [S99]. "
        "This statement has no source. "
        "Another statement uses broken syntax [Sx]."
    )

    result = validate_citations(answer, sources)

    assert not result.valid
    assert result.used_source_ids == ("S1", "S99")
    assert result.unknown_source_ids == ("S99",)
    assert result.malformed_citations == ("[Sx]",)
    assert result.uncited_claims == ("This statement has no source.",)
    assert extract_citation_ids("Combined evidence [S1, S2].") == ("S1", "S2")


def test_injected_llm_verifier_labels_claims_and_cannot_invent_evidence() -> None:
    sources = (
        _source("S1", "example.edu", "Repeated retrieval improves delayed recall."),
        _source("S2", "science.gov", "Lunar samples are made of rock and minerals."),
    )
    answer = (
        "Retrieval practice improves delayed recall [S1]. "
        "The Moon is made of cheese [S2]. "
        "Retrieval might help every learner equally [S1]."
    )
    callback_prompts: list[str] = []

    def fake_gemma_verifier(prompt: str):
        callback_prompts.append(prompt)
        return {
            "claims": [
                {
                    "claim_id": "C1",
                    "label": "supported",
                    "evidence_source_ids": ["S1"],
                    "evidence_quote": "Repeated retrieval improves delayed recall.",
                    "reason": "S1 directly states the effect.",
                },
                {
                    "claim_id": "C2",
                    "label": "unsupported",
                    "evidence_source_ids": ["S2"],
                    "evidence_quote": "Lunar samples are made of rock and minerals.",
                    "reason": "S2 describes rock and minerals, not cheese.",
                },
                {
                    "claim_id": "C3",
                    "label": "uncertain",
                    "evidence_source_ids": ["S1"],
                    "evidence_quote": "Repeated retrieval improves delayed recall.",
                    "reason": "S1 does not establish an identical effect for every learner.",
                },
            ]
        }

    report = verify_answer(answer, sources, llm_callback=fake_gemma_verifier)

    assert callback_prompts and "Return JSON only" in callback_prompts[0]
    assert report.used_llm_verifier
    assert [claim.label for claim in report.claims] == ["supported", "unsupported", "uncertain"]
    assert report.supported_count == 1
    assert report.unsupported_count == 1
    assert report.uncertain_count == 1
    assert report.safe_to_show is False


def test_supported_verdict_requires_a_known_source_cited_inline() -> None:
    source = _source("S1", "example.edu", "The evidence supports the factual claim.")
    answer = "The factual claim is true without an inline citation."

    def overconfident_verifier(prompt: str):
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "C1",
                        "label": "supported",
                        "evidence_source_ids": ["S1"],
                        "evidence_quote": "The evidence supports the factual claim.",
                        "reason": "Trust me.",
                    }
                ]
            }
        )

    report = verify_answer(answer, (source,), llm_callback=overconfident_verifier)

    assert report.claims[0].label == "unsupported"
    assert report.safe_to_show is False
    assert not report.citation_validation.valid


def test_deterministic_fallback_never_approves_lexical_overlap_or_negation() -> None:
    source = _source("S1", "example.edu", "Repeated retrieval improves delayed recall.")
    answer = "Repeated retrieval does not improve delayed recall [S1]."

    report = verify_answer(answer, (source,))

    assert report.used_llm_verifier is False
    assert report.claims[0].label == "uncertain"
    assert report.safe_to_show is False


def test_complete_semantic_verifier_with_verbatim_quote_can_release_answer() -> None:
    evidence = "Repeated retrieval improves delayed recall."
    source = _source("S1", "example.edu", evidence)
    answer = "Repeated retrieval improves delayed recall [S1]."

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": evidence,
                "reason": "The quote directly entails the claim.",
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is True
    assert report.claims[0].evidence_quote == evidence
    assert report.claims[0].label == "supported"
    assert report.safe_to_show is True


def test_selective_excerpt_rescues_only_supported_verified_claims() -> None:
    evidence = "Repeated retrieval improves delayed recall."
    source = _source("S1", "example.edu", evidence)
    answer = (
        "Repeated retrieval improves delayed recall [S1]. "
        "The effect is identical for every learner [S1]."
    )

    def verifier(prompt: str):
        return {
            "claims": [
                {
                    "claim_id": "C1",
                    "label": "supported",
                    "evidence_source_ids": ["S1"],
                    "evidence_quote": evidence,
                    "reason": "The quote directly entails the claim.",
                },
                {
                    "claim_id": "C2",
                    "label": "uncertain",
                    "evidence_source_ids": ["S1"],
                    "evidence_quote": evidence,
                    "reason": "The evidence does not cover every learner.",
                },
            ]
        }

    full_report = verify_answer(answer, (source,), llm_callback=verifier)
    selected = select_supported_excerpt(answer, (source,), full_report)

    assert full_report.safe_to_show is False
    assert selected is not None
    excerpt, excerpt_report = selected
    assert excerpt == "Repeated retrieval improves delayed recall [S1]."
    assert "every learner" not in excerpt
    assert excerpt_report.safe_to_show is True
    assert excerpt_report.used_llm_verifier is True
    assert excerpt_report.citation_validation.valid is True
    assert [claim.label for claim in excerpt_report.claims] == ["supported"]


def test_selective_excerpt_refuses_deterministic_fallback_report() -> None:
    evidence = "Repeated retrieval improves delayed recall."
    source = _source("S1", "example.edu", evidence)
    answer = "Repeated retrieval improves delayed recall [S1]."
    fallback_report = verify_answer(answer, (source,))

    assert fallback_report.used_llm_verifier is False
    assert select_supported_excerpt(answer, (source,), fallback_report) is None


def test_selective_excerpt_refuses_when_verifier_supported_no_claims() -> None:
    evidence = "Repeated retrieval improves delayed recall."
    source = _source("S1", "example.edu", evidence)
    answer = "The effect is identical for every learner [S1]."

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "uncertain",
                "evidence_source_ids": ["S1"],
                "evidence_quote": evidence,
                "reason": "The evidence does not establish this universal claim.",
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is True
    assert report.supported_count == 0
    assert select_supported_excerpt(answer, (source,), report) is None


@pytest.mark.parametrize(
    ("source_text", "answer", "reason"),
    [
        (
            "Repeated retrieval improves delayed recall.",
            "Repeated retrieval improves delayed recall [S1].",
            "The evidence does not support this claim.",
        ),
        (
            "Repeated retrieval improves delayed recall.",
            "Repeated retrieval improves delayed recall [S1].",
            "The evidence does not actually state this.",
        ),
        (
            "Repeated retrieval improves delayed recall.",
            "Repeated retrieval improves delayed recall [S1].",
            "The claim is not clearly supported by this evidence.",
        ),
        (
            "Repeated retrieval improves delayed recall.",
            "Repeated retrieval does not improve delayed recall [S1].",
            "The quote directly supports the claim.",
        ),
        (
            "The trial enrolled 21 students.",
            "The trial enrolled 12 students [S1].",
            "The quote directly supports the claim.",
        ),
        (
            "The package weighs 12 kg.",
            "The package weighs 12 lb [S1].",
            "The quote directly supports the claim.",
        ),
        (
            "The examination is in April 2025.",
            "The examination is in March 2025 [S1].",
            "The quote directly supports the claim.",
        ),
        (
            "The deadline is 2025-03-12.",
            "The deadline is 2025-12-03 [S1].",
            "The quote directly supports the claim.",
        ),
        (
            "Scores decreased after practice.",
            "Scores increased after practice [S1].",
            "The quote directly supports the claim.",
        ),
    ],
)
def test_supported_label_is_downgraded_on_deterministic_conflict(
    source_text: str,
    answer: str,
    reason: str,
) -> None:
    source = _source("S1", "example.edu", source_text)

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": source_text,
                "reason": reason,
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is True
    assert report.claims[0].label == "uncertain"
    assert report.safe_to_show is False


def test_supported_quote_must_be_verbatim_substring_of_cited_source() -> None:
    source = _source("S1", "example.edu", "The observed gain was 18 percent.")
    answer = "The observed gain was 18 percent [S1]."

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": "The observed gain was approximately 18 percent.",
                "reason": "The source supports the value.",
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.claims[0].label == "uncertain"
    assert "verbatim substring" in report.claims[0].reason
    assert report.safe_to_show is False


def test_overlong_quote_cannot_hide_negation_in_a_truncated_tail() -> None:
    evidence = "Repeated retrieval " + ("context " * 260) + "does not improve delayed recall."
    assert len(evidence) > 2_000
    source = _source("S1", "example.edu", evidence)
    answer = "Repeated retrieval improves delayed recall [S1]."

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": evidence,
                "reason": "The quote directly entails the claim.",
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is False
    assert report.claims[0].label == "uncertain"
    assert report.claims[0].evidence_quote == ""
    assert report.safe_to_show is False


def test_overlong_reason_cannot_hide_label_conflict_in_a_truncated_tail() -> None:
    evidence = "Repeated retrieval improves delayed recall."
    source = _source("S1", "example.edu", evidence)
    answer = "Repeated retrieval improves delayed recall [S1]."
    reason = (
        "Apparently supported. "
        + ("detail " * 100)
        + "The evidence does not actually state this."
    )
    assert len(reason) > 600

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": evidence,
                "reason": reason,
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is False
    assert report.claims[0].label == "uncertain"
    assert report.claims[0].reason.startswith("Citation is valid")
    assert report.safe_to_show is False


def test_overlong_fabricated_quote_suffix_cannot_pass_via_prefix_truncation() -> None:
    prefix = "Repeated retrieval improves delayed recall. " + ("evidence " * 220)
    source = _source("S1", "example.edu", prefix)
    fabricated_quote = prefix + "This unsupported suffix is not in the cited source."
    answer = "Repeated retrieval improves delayed recall [S1]."
    assert len(fabricated_quote) > 2_000

    def verifier(prompt: str):
        return {
            "claims": [{
                "claim_id": "C1",
                "label": "supported",
                "evidence_source_ids": ["S1"],
                "evidence_quote": fabricated_quote,
                "reason": "The quote directly entails the claim.",
            }]
        }

    report = verify_answer(answer, (source,), llm_callback=verifier)

    assert report.used_llm_verifier is False
    assert report.claims[0].label == "uncertain"
    assert report.safe_to_show is False


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "invalid"])
def test_missing_duplicate_unknown_or_invalid_verifier_rows_fail_closed(failure: str) -> None:
    sources = (
        _source("S1", "example.edu", "Retrieval improves delayed recall."),
        _source("S2", "science.gov", "Practice improves transfer performance."),
    )
    answer = "Retrieval improves delayed recall [S1]. Practice improves transfer performance [S2]."

    def row(claim_id: str, source_id: str, quote: str) -> dict[str, object]:
        return {
            "claim_id": claim_id,
            "label": "supported",
            "evidence_source_ids": [source_id],
            "evidence_quote": quote,
            "reason": "The quote directly entails the claim.",
        }

    rows = [
        row("C1", "S1", "Retrieval improves delayed recall."),
        row("C2", "S2", "Practice improves transfer performance."),
    ]
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[1] = row("C1", "S1", "Retrieval improves delayed recall.")
    elif failure == "unknown":
        rows[1] = row("C3", "S2", "Practice improves transfer performance.")
    else:
        rows[1]["label"] = "probably"

    report = verify_answer(answer, sources, llm_callback=lambda prompt: {"claims": rows})

    assert report.used_llm_verifier is False
    assert [claim.label for claim in report.claims] == ["uncertain", "uncertain"]
    assert report.safe_to_show is False
