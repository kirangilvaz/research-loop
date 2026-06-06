"""DuckDuckGo HTML scraper (async, no Playwright).

Hits `html.duckduckgo.com/html/?q=...` directly via httpx with realistic
browser headers and parses results with the same regex we had before. Falls
back to the `lite.duckduckgo.com/lite/` endpoint when the primary returns
nothing.

No API key required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from .. import config
from ..browser.http_fetch import HttpClient, get_with_backoff, polite_headers
from ..utils import canonicalize_url, host_of, strip_html


@dataclass
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""
    rank: int = 0


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?(?:<a[^>]+class="result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>'
    r'|<div[^>]+class="result__snippet[^"]*"[^>]*>(?P<snippet2>.*?)</div>)',
    re.I | re.S,
)

_LITE_LINK_RE = re.compile(
    r'<a[^>]+rel="nofollow"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)


def _decode_ddg_redirect(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
    except Exception:
        return ""
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        target = (qs.get("uddg") or qs.get("u") or [""])[0]
        if target:
            return canonicalize_url(unquote(target))
        return ""
    return canonicalize_url(href)


def _parse_results(html: str, max_results: int) -> List[SearchHit]:
    if not html:
        return []
    hits: List[SearchHit] = []
    seen: Set[str] = set()
    for i, m in enumerate(_RESULT_RE.finditer(html)):
        if i >= max_results * 3:
            break
        href = m.group("href") or ""
        title_html = m.group("title") or ""
        snippet_html = m.group("snippet") or m.group("snippet2") or ""
        url = _decode_ddg_redirect(href)
        if not url or url in seen:
            continue
        if not host_of(url):
            continue
        seen.add(url)
        hits.append(
            SearchHit(
                url=url,
                title=strip_html(title_html)[:300],
                snippet=strip_html(snippet_html)[:600],
                rank=len(hits),
            )
        )
        if len(hits) >= max_results:
            break
    return hits


def _parse_lite(html: str, max_results: int) -> List[SearchHit]:
    hits: List[SearchHit] = []
    seen: Set[str] = set()
    for i, m in enumerate(_LITE_LINK_RE.finditer(html or "")):
        if len(hits) >= max_results:
            break
        href = m.group("href") or ""
        title = strip_html(m.group("title") or "")
        url = _decode_ddg_redirect(href)
        if not url or url in seen:
            continue
        if not host_of(url):
            continue
        seen.add(url)
        hits.append(SearchHit(url=url, title=title[:300], snippet="", rank=i))
    return hits


_DDG_HEADERS = polite_headers(
    {
        "Referer": "https://duckduckgo.com/",
        "Sec-Fetch-Site": "same-site",
    }
)


_BLOCK_HINTS = (
    "anomaly",
    "captcha",
    "unusual traffic",
    "are you a robot",
    "verify you are human",
    "blocked",
    "rate limit",
)


def _diagnose_empty(label: str, query: str, resp) -> None:
    """Print a one-line diagnosis when DDG returns 200 but parses to 0 hits.

    Helps distinguish 'CAPTCHA / blocked' from 'layout change' from 'truly no
    results'. We grep for known block-page substrings and otherwise sample
    the body head, stripping HTML angle brackets and collapsing whitespace.
    """
    if resp is None:
        return
    body = resp.text or ""
    status = resp.status_code
    length = len(body)
    lower = body.lower()
    hit_marker = next((m for m in _BLOCK_HINTS if m in lower), None)
    sample = re.sub(r"<[^>]+>", " ", body[:600])
    sample = re.sub(r"\s+", " ", sample).strip()[:200]
    if hit_marker:
        diag = f"BLOCKED({hit_marker!r})"
    elif length < 200:
        diag = "TINY_BODY"
    else:
        diag = "PARSE_MISS"
    print(
        f"  [ddg/{label}] empty parse for {query[:40]!r}: status={status} "
        f"len={length} -> {diag}; head={sample!r}"
    )


async def search_duckduckgo(
    client: HttpClient,
    query: str,
    max_results: int = config.DDG_RESULTS_PER_QUERY,
) -> List[SearchHit]:
    """Run a single DuckDuckGo search; returns up to `max_results` hits."""
    if not query or not query.strip():
        return []
    primary_url = f"{config.DDG_HTML_ENDPOINT}?q={quote_plus(query)}"
    try:
        resp = await get_with_backoff(
            client,
            primary_url,
            host="html.duckduckgo.com",
            headers=_DDG_HEADERS,
            max_attempts=2,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [ddg/html] transport error for {query[:40]!r}: {e}")
        resp = None
    primary_ok = resp is not None and resp.status_code < 400
    html = resp.text if primary_ok else ""
    hits = _parse_results(html, max_results) if html else []
    if hits:
        return hits
    if primary_ok:
        _diagnose_empty("html", query, resp)
    elif resp is not None:
        print(f"  [ddg/html] status={resp.status_code} for {query[:40]!r}")

    lite_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    try:
        lite_resp = await get_with_backoff(
            client,
            lite_url,
            host="lite.duckduckgo.com",
            headers=_DDG_HEADERS,
            max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [ddg/lite] transport error for {query[:40]!r}: {e}")
        return []
    if lite_resp is None or lite_resp.status_code >= 400 or not lite_resp.text:
        if lite_resp is not None:
            print(f"  [ddg/lite] status={lite_resp.status_code} for {query[:40]!r}")
        return []
    lite_hits = _parse_lite(lite_resp.text, max_results)
    if not lite_hits:
        _diagnose_empty("lite", query, lite_resp)
    return lite_hits


async def self_test(client: HttpClient) -> str:
    """One-shot ping at startup; returns a human-readable status string.

    DDG's HTML endpoint is famously trigger-happy with anomaly throttling,
    so we expect the test to occasionally show `BLOCKED` even on a fresh
    IP. The point is to make that visible rather than have the user
    discover it after iteration 2.
    """
    url = f"{config.DDG_HTML_ENDPOINT}?q=test"
    try:
        resp = await get_with_backoff(
            client,
            url,
            host="html.duckduckgo.com",
            headers=_DDG_HEADERS,
            max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        return f"unreachable: {e}"
    if resp is None:
        return "unreachable (no response)"
    if resp.status_code == 403:
        return "HTTP 403 -- IP already rate-limited; expect link-harvest + SearXNG to carry the load"
    if resp.status_code >= 400:
        return f"HTTP {resp.status_code}"
    hits = _parse_results(resp.text or "", 5)
    if hits:
        return f"HTML ok ({len(hits)} hits for 'test')"
    lower = (resp.text or "")[:600].lower()
    if any(m in lower for m in ("anomaly", "captcha", "verify you are human")):
        return "HTML 200 but anomaly/CAPTCHA page returned -- effectively blocked"
    return f"HTML 200 but parsed 0 results (layout may have changed)"


__all__ = ["SearchHit", "search_duckduckgo", "self_test"]
