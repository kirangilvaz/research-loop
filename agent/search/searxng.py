"""SearXNG client with JSON-first + HTML fallback + self-test.

Reads `config.SEARXNG_URL` (env: `RESEARCH_SEARXNG_URL`); if unset, every
call returns an empty list and the backend is silently skipped by the
dispatcher.

Why two endpoints?
    Many self-hosted SearXNG instances ship with `format=json` disabled
    (only HTML enabled) or with `limiter: true` (which 403s JSON requests
    while still serving HTML). Our previous version silently returned []
    in both cases, which was indistinguishable from "engines empty". This
    module now:

      1. Tries `format=json` first (clean structured data).
      2. On non-JSON content-type, 4xx status, or empty results, falls
         back to the regular HTML endpoint and parses
         `<article class="result">` blocks.
      3. Emits a one-line diagnosis (`JSON_DISABLED`, `LIMITER_BLOCKED`,
         `BOT_DETECTED`, etc.) so the user can tell *why* a path failed,
         mirroring the diagnostic we added for DuckDuckGo.

`self_test(client)` does a single ping at startup and returns a human
summary of the instance's state -- so misconfiguration is visible
immediately, not after 20 iterations of zero hits.
"""

from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import quote_plus

from .. import config
from ..browser.http_fetch import HttpClient, get_with_backoff, polite_headers
from ..utils import canonicalize_url, host_of, strip_html
from .duckduckgo import SearchHit


_JSON_HEADERS = polite_headers(
    {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
)

_HTML_HEADERS = polite_headers(
    {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)


# SearXNG's "simple" theme wraps each result in <article class="result ...">.
# Older themes use <div class="result-default ...">. The regex matches both
# by anchoring on the class substring "result" instead of the tag name.
_HTML_RESULT_BLOCK_RE = re.compile(
    r'<(?P<tag>article|div)[^>]*class="[^"]*\bresult[\w-]*\b[^"]*"[^>]*>(?P<body>.*?)</(?P=tag)>',
    re.I | re.S,
)
# Inside each block, the title link sits in <h3><a href="..."> or
# <h4 class="result_header"><a href="...">. We accept either.
_HTML_TITLE_LINK_RE = re.compile(
    r'<h[34][^>]*>\s*<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)
_HTML_SNIPPET_RE = re.compile(
    r'<p[^>]*class="[^"]*(?:content|result-content|snippet|result__snippet)[^"]*"[^>]*>(?P<snippet>.*?)</p>',
    re.I | re.S,
)


def is_enabled() -> bool:
    return bool(config.SEARXNG_URL)


def _base() -> str:
    return (config.SEARXNG_URL or "").rstrip("/")


def _host_label() -> str:
    return host_of(_base()) or "searxng"


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _diagnose_empty(label: str, query: str, resp) -> None:
    """Emit a one-line diagnosis (mirrors the DDG `[ddg/html] empty parse` log).

    `label` is "json" or "html"; the diagnosis classifies the response into
    one of a fixed set of buckets so the user can tell at a glance which
    SearXNG setting needs flipping.
    """
    if resp is None:
        return
    status = resp.status_code
    ctype = (resp.headers.get("content-type") or "").lower()
    body = resp.text or ""
    length = len(body)
    lower_body = body.lower()

    if status == 403:
        diag = "LIMITER_BLOCKED (try `limiter: false` in settings.yml)"
    elif status == 429:
        diag = "RATE_LIMITED"
    elif status >= 400:
        diag = f"HTTP_{status}"
    elif label == "json" and "html" in ctype:
        diag = "JSON_DISABLED (got HTML; enable `search.formats: [html, json]` in settings.yml)"
    elif label == "json" and "json" not in ctype:
        diag = f"NON_JSON_CONTENT_TYPE({ctype!r})"
    elif label == "json" and "json" in ctype:
        # 200 + JSON content-type but we still got here -> the JSON parsed
        # but `results` was empty. Engines returned nothing for this query.
        diag = "EMPTY_RESULTS (200 + valid JSON but engines returned nothing; cooldown or no engines configured)"
    elif any(m in lower_body for m in ("too many requests", "rate limit", "captcha", "anomaly")):
        diag = "BOT_DETECTED"
    elif length < 200:
        diag = "TINY_BODY"
    else:
        diag = "PARSE_MISS (HTML returned but no <article class=result>; layout may differ)"

    sample = re.sub(r"<[^>]+>", " ", body[:600])
    sample = re.sub(r"\s+", " ", sample).strip()[:200]
    print(
        f"  [searxng/{label}] empty for {query[:40]!r}: status={status} "
        f"len={length} ct={ctype!r} -> {diag}; head={sample!r}"
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_json(payload: dict, max_results: int) -> List[SearchHit]:
    raw = payload.get("results") or []
    hits: List[SearchHit] = []
    seen: set = set()
    for entry in raw:
        href = (entry.get("url") or "").strip()
        cu = canonicalize_url(href)
        if not cu or cu in seen:
            continue
        if not host_of(cu):
            continue
        seen.add(cu)
        title = strip_html(entry.get("title") or "")[:300]
        snippet = strip_html(entry.get("content") or "")[:600]
        hits.append(SearchHit(url=cu, title=title, snippet=snippet, rank=len(hits)))
        if len(hits) >= max_results:
            break
    return hits


def _parse_html(html: str, max_results: int) -> List[SearchHit]:
    if not html:
        return []
    hits: List[SearchHit] = []
    seen: set = set()
    for block in _HTML_RESULT_BLOCK_RE.finditer(html):
        body = block.group("body")
        link_m = _HTML_TITLE_LINK_RE.search(body)
        if not link_m:
            continue
        href = (link_m.group("href") or "").strip()
        cu = canonicalize_url(href)
        if not cu or cu in seen:
            continue
        if not host_of(cu):
            continue
        # Skip internal SearXNG links (preferences, image previews, etc).
        if _host_label() and host_of(cu) == _host_label():
            continue
        seen.add(cu)
        title = strip_html(link_m.group("title"))[:300]
        snippet_m = _HTML_SNIPPET_RE.search(body)
        snippet = strip_html(snippet_m.group("snippet"))[:600] if snippet_m else ""
        hits.append(SearchHit(url=cu, title=title, snippet=snippet, rank=len(hits)))
        if len(hits) >= max_results:
            break
    return hits


# ---------------------------------------------------------------------------
# Public search entry point
# ---------------------------------------------------------------------------

async def search_searxng(
    client: HttpClient,
    query: str,
    max_results: int = config.DDG_RESULTS_PER_QUERY,
) -> List[SearchHit]:
    """Try SearXNG JSON, fall back to HTML, log diagnostics on every empty path."""
    if not is_enabled() or not query or not query.strip():
        return []

    host = _host_label()
    base = _base()

    # ----- Tier 1: JSON endpoint -----
    json_url = f"{base}/search?q={quote_plus(query)}&format=json&safesearch=0&pageno=1"
    json_resp = None
    try:
        json_resp = await get_with_backoff(
            client, json_url, host=host, headers=_JSON_HEADERS, max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [searxng/json] transport error for {query[:40]!r}: {e}")

    if json_resp is not None and json_resp.status_code < 400:
        ctype = (json_resp.headers.get("content-type") or "").lower()
        if "json" in ctype:
            try:
                data = json_resp.json()
            except Exception:
                data = None
            if isinstance(data, dict):
                hits = _parse_json(data, max_results)
                if hits:
                    return hits
                _diagnose_empty("json", query, json_resp)
            else:
                _diagnose_empty("json", query, json_resp)
        else:
            # 200 OK but JSON disabled -- fall through to HTML.
            _diagnose_empty("json", query, json_resp)
    elif json_resp is not None:
        _diagnose_empty("json", query, json_resp)

    # ----- Tier 2: HTML endpoint fallback -----
    html_url = f"{base}/search?q={quote_plus(query)}&safesearch=0&pageno=1"
    html_resp = None
    try:
        html_resp = await get_with_backoff(
            client, html_url, host=host, headers=_HTML_HEADERS, max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [searxng/html] transport error for {query[:40]!r}: {e}")
        return []
    if html_resp is None or html_resp.status_code >= 400:
        if html_resp is not None:
            _diagnose_empty("html", query, html_resp)
        return []
    hits = _parse_html(html_resp.text or "", max_results)
    if not hits:
        _diagnose_empty("html", query, html_resp)
    return hits


# ---------------------------------------------------------------------------
# Startup self-test
# ---------------------------------------------------------------------------

async def self_test(client: HttpClient) -> str:
    """One-shot ping at startup; returns a human-readable status string.

    The goal is to make misconfiguration obvious immediately. We probe both
    the JSON and HTML endpoints with a trivial query ('test') and report
    exactly which path works.
    """
    if not is_enabled():
        return "skipped (not configured)"

    base = _base()
    host = _host_label()

    # Probe JSON first.
    try:
        json_resp = await get_with_backoff(
            client,
            f"{base}/search?q=test&format=json",
            host=host,
            headers=_JSON_HEADERS,
            max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        return f"unreachable: {e}"
    if json_resp is None:
        return "unreachable (no response from instance)"

    status = json_resp.status_code
    if status == 403:
        return "HTTP 403 -- limiter blocking automated traffic (set `limiter: false` in searxng/settings.yml)"
    if status == 429:
        return "HTTP 429 -- rate limited"
    if status >= 400 and status not in (403, 429):
        return f"HTTP {status} on JSON endpoint"

    ctype = (json_resp.headers.get("content-type") or "").lower()
    json_hits = 0
    if "json" in ctype:
        try:
            data = json_resp.json()
            json_hits = len(data.get("results") or [])
        except Exception:
            json_hits = -1

    if json_hits > 0:
        return f"JSON ok ({json_hits} hits for 'test')"

    # JSON didn't work; probe HTML to know if the fallback can carry us.
    try:
        html_resp = await get_with_backoff(
            client,
            f"{base}/search?q=test",
            host=host,
            headers=_HTML_HEADERS,
            max_attempts=1,
        )
    except Exception:
        html_resp = None

    html_hits = 0
    if html_resp is not None and html_resp.status_code < 400:
        html_hits = len(_parse_html(html_resp.text or "", 5))

    # Now compose a precise summary.
    if json_hits == 0 and "json" in ctype:
        if html_hits > 0:
            return (
                f"JSON returned 0 hits (engines empty/cooldown); "
                f"HTML fallback ok ({html_hits} hits) -- agent will use HTML path"
            )
        return "JSON returned 0 hits and HTML also empty -- check enabled engines in settings.yml"
    # JSON not enabled (HTML content-type)
    if html_hits > 0:
        return (
            f"JSON disabled (content-type={ctype!r}); HTML fallback ok ({html_hits} hits) -- "
            f"agent will use HTML path. To enable JSON, add `search.formats: [html, json]` to settings.yml."
        )
    if html_resp is not None and html_resp.status_code == 403:
        return "JSON disabled AND HTML 403 -- limiter is blocking; set `limiter: false` in settings.yml"
    return (
        f"JSON disabled (content-type={ctype!r}) AND HTML returned 0 hits -- "
        f"instance is reachable but no engines are returning results"
    )


__all__ = ["is_enabled", "search_searxng", "self_test"]
