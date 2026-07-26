from __future__ import annotations

from email.message import Message
import json

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
    extract_citation_ids,
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
                    "reason": "S1 directly states the effect.",
                },
                {
                    "claim_id": "C2",
                    "label": "unsupported",
                    "evidence_source_ids": ["S2"],
                    "reason": "S2 describes rock and minerals, not cheese.",
                },
                {
                    "claim_id": "C3",
                    "label": "uncertain",
                    "evidence_source_ids": ["S1"],
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
                        "reason": "Trust me.",
                    }
                ]
            }
        )

    report = verify_answer(answer, (source,), llm_callback=overconfident_verifier)

    assert report.claims[0].label == "unsupported"
    assert report.safe_to_show is False
    assert not report.citation_validation.valid
