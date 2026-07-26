"""Safe, optional web research for evidence-backed Gemma answers.

This module deliberately calls the activity *preparation* rather than training.
No model weights are changed: a small, inspectable research pack is retrieved and
placed in the model context for one answer.

The web dependencies are optional.  When ``ddgs`` or ``trafilatura`` is missing,
the module remains importable and either uses the standard-library fallback or
returns an explicit offline research pack.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from html import unescape
from html.parser import HTMLParser
import ipaddress
import re
import socket
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 6.0
DEFAULT_MAX_PAGE_BYTES = 1_000_000
DEFAULT_USER_AGENT = "ProofMode/0.1 evidence-research (+local learning app)"


class UnsafeURLError(ValueError):
    """Raised before a URL that could access a local/private service is fetched."""


class FetchError(RuntimeError):
    """Raised when a public page cannot be safely downloaded or extracted."""


@dataclass(frozen=True)
class ResearchSource:
    """A normalized source made available to Gemma and to the UI."""

    source_id: str
    title: str
    url: str
    snippet: str
    domain: str
    text: str = ""
    authority_score: float = 0.0

    @property
    def evidence(self) -> str:
        """Best evidence text available, preferring fetched page content."""

        return self.text.strip() or self.snippet.strip()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchPack:
    """The complete, transparent preparation context for one question."""

    query: str
    sources: tuple[ResearchSource, ...] = ()
    status: str = "offline"
    warnings: tuple[str, ...] = ()
    method: str = "preparation/research pack (no fine-tuning)"

    @property
    def is_offline(self) -> bool:
        return self.status == "offline"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "warnings": list(self.warnings),
            "method": self.method,
            "sources": [source.as_dict() for source in self.sources],
        }


_HIGH_AUTHORITY_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".edu",
    ".ac.uk",
    ".nhs.uk",
)

_HIGH_AUTHORITY_DOMAINS = {
    "who.int",
    "un.org",
    "europa.eu",
    "oecd.org",
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "docs.python.org",
    "developer.mozilla.org",
    "developers.google.com",
    "ai.google.dev",
}

_LOW_AUTHORITY_DOMAINS = {
    "reddit.com",
    "quora.com",
    "pinterest.com",
    "tiktok.com",
    "facebook.com",
    "x.com",
}


def source_domain(url: str) -> str:
    """Return a stable, lower-case hostname without a leading ``www``."""

    hostname = (urlparse(url).hostname or "").lower().rstrip(".")
    return hostname[4:] if hostname.startswith("www.") else hostname


def authority_score(
    url: str,
    *,
    title: str = "",
    snippet: str = "",
    query: str = "",
) -> float:
    """Heuristically rank primary/official sources ahead of social/SEO results.

    This score is a retrieval ordering signal, not a guarantee that a statement is
    true.  ProofMode still asks the verifier to inspect the evidence for each claim.
    """

    domain = source_domain(url)
    score = 0.0
    if domain in _HIGH_AUTHORITY_DOMAINS or any(
        domain == suffix.lstrip(".") or domain.endswith(suffix)
        for suffix in _HIGH_AUTHORITY_SUFFIXES
    ):
        score += 6.0
    if any(token in domain for token in ("journal", "university", "institute")):
        score += 2.0
    if domain in _LOW_AUTHORITY_DOMAINS or any(
        domain.endswith("." + low) for low in _LOW_AUTHORITY_DOMAINS
    ):
        score -= 5.0

    lowered = f"{title} {snippet}".lower()
    if any(term in lowered for term in ("official", "documentation", "systematic review", "randomized")):
        score += 1.5
    if any(term in lowered for term in ("sponsored", "affiliate", "opinion")):
        score -= 1.0

    query_terms = {term for term in re.findall(r"[a-z0-9]{3,}", query.lower())}
    text_terms = set(re.findall(r"[a-z0-9]{3,}", lowered))
    if query_terms:
        score += min(2.0, 2.0 * len(query_terms & text_terms) / len(query_terms))
    return round(score, 3)


def _is_forbidden_ip(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    return not parsed.is_global


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., Sequence[Any]] | None = socket.getaddrinfo,
) -> str:
    """Validate an HTTP(S) URL and reject loopback/private/link-local targets.

    DNS is resolved before the request to reduce SSRF risk.  A resolver can be
    injected in deterministic tests.  The validated URL is returned unchanged.
    """

    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeURLError("Only http:// and https:// sources are allowed")
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credential-bearing URLs are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError("URL has no hostname")
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        raise UnsafeURLError("Localhost sources are not allowed")

    try:
        literal = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise UnsafeURLError("Private, loopback, and link-local sources are not allowed")

    if resolver is not None and literal is None:
        try:
            records = resolver(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        except OSError as exc:
            raise FetchError(f"Could not resolve source host: {hostname}") from exc
        addresses = {record[4][0] for record in records if len(record) > 4 and record[4]}
        if not addresses:
            raise FetchError(f"Source host resolved to no addresses: {hostname}")
        if any(_is_forbidden_ip(address) for address in addresses):
            raise UnsafeURLError("Source resolves to a non-public network address")
    return url


class _ReadableHTMLParser(HTMLParser):
    """Small dependency-free article-text fallback."""

    _BLOCKED = {"script", "style", "noscript", "svg", "canvas", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocked_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BLOCKED:
            self._blocked_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCKED and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._chunks)


def extract_readable_text(payload: str, *, content_type: str = "text/html") -> str:
    """Extract readable text, using trafilatura only when it is installed."""

    if "html" not in content_type.lower():
        return _clean_text(payload)
    try:
        import trafilatura  # type: ignore[import-not-found]

        extracted = trafilatura.extract(
            payload,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted and extracted.strip():
            return _clean_text(extracted)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    parser = _ReadableHTMLParser()
    parser.feed(payload)
    return _clean_text(parser.text())


def _clean_text(text: str) -> str:
    text = unescape(text).replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_fetch_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    opener: Callable[..., Any] | None = None,
    resolver: Callable[..., Sequence[Any]] | None = socket.getaddrinfo,
) -> str:
    """Download and extract bounded public HTML/text content.

    Requests have a short timeout and byte cap.  Binary content is rejected.  Both
    the initial and final redirect URLs are safety checked.
    """

    validate_public_url(url, resolver=resolver)
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    open_url = opener or urlopen
    try:
        response = open_url(request, timeout=timeout)
        with response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            validate_public_url(final_url, resolver=resolver)
            status = getattr(response, "status", 200)
            if status and int(status) >= 400:
                raise FetchError(f"Source returned HTTP {status}")
            headers = getattr(response, "headers", {})
            content_type = headers.get("Content-Type", "text/html") if headers else "text/html"
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
                raise FetchError(f"Unsupported source content type: {media_type or 'unknown'}")
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise FetchError(f"Source exceeds the {max_bytes}-byte safety limit")
            charset = "utf-8"
            if headers and hasattr(headers, "get_content_charset"):
                charset = headers.get_content_charset() or charset
            decoded = raw.decode(charset, errors="replace")
    except (UnsafeURLError, FetchError):
        raise
    except Exception as exc:  # urllib and injected clients expose varied errors
        raise FetchError(f"Could not fetch source: {source_domain(url) or url}") from exc

    extracted = extract_readable_text(decoded, content_type=content_type)
    if not extracted:
        raise FetchError("Source contained no readable text")
    return extracted


def _default_search(query: str, max_results: int) -> Iterable[Mapping[str, Any]]:
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Web search is unavailable: optional package 'ddgs' is not installed") from exc
    # Google's text index is consistently stronger for technical and academic
    # queries in the current DDGS backends. Fall back to the metasearch route if
    # that provider is unavailable rather than turning research into a hard
    # dependency on one public endpoint.
    try:
        return DDGS().text(query, max_results=max_results, backend="google")
    except Exception:
        return DDGS().text(query, max_results=max_results, backend="auto")


def _call_search_client(
    client: Any,
    query: str,
    max_results: int,
) -> Iterable[Mapping[str, Any]]:
    if callable(client):
        try:
            return client(query, max_results=max_results)
        except TypeError:
            return client(query, max_results)
    if hasattr(client, "text"):
        return client.text(query, max_results=max_results)
    if hasattr(client, "search"):
        return client.search(query, max_results=max_results)
    raise TypeError("search_client must be callable or expose text()/search()")


def _normalize_result(result: Mapping[str, Any], query: str) -> ResearchSource | None:
    url = str(result.get("url") or result.get("href") or result.get("link") or "").strip()
    title = str(result.get("title") or result.get("name") or source_domain(url) or "Untitled source").strip()
    snippet = str(result.get("snippet") or result.get("body") or result.get("description") or "").strip()
    domain = source_domain(url)
    if not url or not domain or urlparse(url).scheme.lower() not in {"http", "https"}:
        return None
    return ResearchSource(
        source_id="",
        title=title[:300],
        url=url,
        snippet=_clean_text(snippet)[:1_500],
        domain=domain,
        authority_score=authority_score(url, title=title, snippet=snippet, query=query),
    )


def search_web(
    query: str,
    *,
    max_results: int = 6,
    search_client: Any | None = None,
) -> ResearchPack:
    """Search and rank sources, returning an offline pack on any search failure."""

    clean_query = " ".join(query.split()).strip()
    if not clean_query:
        return ResearchPack(query="", status="offline", warnings=("No research query was provided.",))
    try:
        raw_results = (
            _default_search(clean_query, max_results * 2)
            if search_client is None
            else _call_search_client(search_client, clean_query, max_results * 2)
        )
        normalized = [source for result in raw_results if (source := _normalize_result(result, clean_query))]
    except Exception as exc:
        return ResearchPack(
            query=clean_query,
            status="offline",
            warnings=(f"Live research unavailable; answer from local material only ({type(exc).__name__}).",),
        )

    deduplicated: list[ResearchSource] = []
    seen_urls: set[str] = set()
    for source in sorted(normalized, key=lambda item: (-item.authority_score, item.domain, item.url)):
        canonical = source.url.rstrip("/").lower()
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        deduplicated.append(source)
        if len(deduplicated) >= max_results:
            break

    sources = tuple(replace(source, source_id=f"S{index}") for index, source in enumerate(deduplicated, 1))
    return ResearchPack(
        query=clean_query,
        sources=sources,
        status="online" if sources else "offline",
        warnings=() if sources else ("Search returned no usable public sources.",),
    )


def _call_page_fetcher(fetcher: Callable[..., str], url: str, timeout: float) -> str:
    try:
        return fetcher(url, timeout=timeout)
    except TypeError:
        return fetcher(url)


def prepare_research_pack(
    query: str,
    *,
    max_results: int = 6,
    search_client: Any | None = None,
    page_fetcher: Callable[..., str] = safe_fetch_text,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_chars_per_source: int = 8_000,
) -> ResearchPack:
    """Search, rank, and fetch a bounded evidence pack for Gemma.

    Fetch failures are isolated per source.  Useful snippets remain available when
    pages block extraction or the device goes offline after search.
    """

    search_pack = search_web(query, max_results=max_results, search_client=search_client)
    if not search_pack.sources:
        return search_pack

    prepared: list[ResearchSource] = []
    warnings = list(search_pack.warnings)
    fetched_count = 0
    # Page downloads are independent and typically dominate research latency.
    # Keep the pool deliberately small, then consume futures in ranked source
    # order so source IDs, result order, and warning order remain deterministic.
    with ThreadPoolExecutor(max_workers=min(4, len(search_pack.sources))) as executor:
        fetches = [
            executor.submit(_call_page_fetcher, page_fetcher, source.url, timeout)
            for source in search_pack.sources
        ]
        for source, fetch in zip(search_pack.sources, fetches):
            try:
                text = _clean_text(fetch.result())[:max_chars_per_source]
                if text:
                    fetched_count += 1
                prepared.append(replace(source, text=text))
            except Exception as exc:
                prepared.append(source)
                warnings.append(
                    f"{source.source_id} could not be fetched; "
                    f"using search snippet ({type(exc).__name__})."
                )

    status = "online" if fetched_count == len(prepared) else "partial"
    if fetched_count == 0 and not any(source.snippet for source in prepared):
        status = "offline"
    return ResearchPack(
        query=search_pack.query,
        sources=tuple(prepared),
        status=status,
        warnings=tuple(warnings),
    )


def build_evidence_prompt(
    question: str,
    research: ResearchPack | Sequence[ResearchSource],
    *,
    max_chars_per_source: int = 6_000,
) -> str:
    """Construct an evidence-only prompt with stable inline source identifiers."""

    if isinstance(research, ResearchPack):
        pack = research
        sources = pack.sources
        status = pack.status
    else:
        sources = tuple(research)
        status = "provided"

    evidence_blocks: list[str] = []
    for index, source in enumerate(sources, 1):
        source_id = source.source_id or f"S{index}"
        evidence = source.evidence[:max_chars_per_source]
        evidence_blocks.append(
            f"[{source_id}] {source.title}\n"
            f"Domain: {source.domain}\n"
            f"URL: {source.url}\n"
            f"Evidence:\n{evidence or '[No extract available]'}"
        )
    evidence_text = "\n\n".join(evidence_blocks) or "[No external evidence was available.]"

    return f"""You are ProofMode's evidence-grounded teaching assistant.

This is a preparation/research pack for this answer. It is retrieval context, NOT training or fine-tuning.
Research status: {status}

NON-NEGOTIABLE RULES
1. Use only the supplied evidence and the student's supplied material for externally verifiable factual claims.
2. Put one or more inline citations such as [S1] immediately after every factual claim they support.
3. Never invent a citation, URL, quotation, statistic, author, or study.
4. A citation supports only what its evidence actually says. If sources disagree, state the disagreement.
5. If the evidence is missing or insufficient, say exactly what cannot be verified; do not fill the gap from memory.
6. Separate established facts from inferences and label inferences explicitly.
7. Explain at the learner's level, but preserve important qualifications and uncertainty.

RESEARCH PACK
{evidence_text}

STUDENT QUESTION
{question.strip()}

Answer with inline [S#] citations. End with a short "Sources used" list containing only sources actually cited."""


def build_evidence_messages(
    question: str,
    research: ResearchPack | Sequence[ResearchSource],
) -> list[dict[str, str]]:
    """Return chat messages suitable for an OpenAI-compatible Gemma endpoint."""

    return [
        {
            "role": "system",
            "content": (
                "Answer from the supplied research pack only. Cite factual claims inline as [S1]. "
                "Say when evidence is insufficient and never invent sources."
            ),
        },
        {"role": "user", "content": build_evidence_prompt(question, research)},
    ]


__all__ = [
    "DEFAULT_MAX_PAGE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "FetchError",
    "ResearchPack",
    "ResearchSource",
    "UnsafeURLError",
    "authority_score",
    "build_evidence_messages",
    "build_evidence_prompt",
    "extract_readable_text",
    "prepare_research_pack",
    "safe_fetch_text",
    "search_web",
    "source_domain",
    "validate_public_url",
]
