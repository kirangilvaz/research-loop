"""Brave Search API client.

Reads `config.BRAVE_API_KEY` (env: `RESEARCH_BRAVE_API_KEY`); if unset, every
call returns an empty list and the backend is silently skipped by the
dispatcher.

Endpoint: https://api.search.brave.com/res/v1/web/search
"""

from __future__ import annotations

from typing import List
from urllib.parse import quote_plus

from .. import config
from ..browser.http_fetch import HttpClient, get_with_backoff, polite_headers
from ..utils import canonicalize_url, host_of, strip_html
from .duckduckgo import SearchHit


_HOST = "api.search.brave.com"


def is_enabled() -> bool:
    return bool(config.BRAVE_API_KEY)


def _headers() -> dict:
    h = polite_headers(
        {
            "Accept": "application/json",
            "X-Subscription-Token": config.BRAVE_API_KEY,
        }
    )
    return h


async def search_brave(
    client: HttpClient,
    query: str,
    max_results: int = config.DDG_RESULTS_PER_QUERY,
) -> List[SearchHit]:
    if not is_enabled() or not query or not query.strip():
        return []
    count = max(1, min(max_results, 20))
    url = (
        f"https://{_HOST}/res/v1/web/search?q={quote_plus(query)}"
        f"&count={count}&safesearch=off"
    )
    try:
        resp = await get_with_backoff(
            client,
            url,
            host=_HOST,
            headers=_headers(),
            max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [brave] error: {e}")
        return []
    if resp is None or resp.status_code >= 400:
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    web = (data.get("web") or {}).get("results") or []
    hits: List[SearchHit] = []
    seen = set()
    for entry in web:
        href = (entry.get("url") or "").strip()
        cu = canonicalize_url(href)
        if not cu or cu in seen:
            continue
        if not host_of(cu):
            continue
        seen.add(cu)
        title = strip_html(entry.get("title") or "")[:300]
        snippet = strip_html(entry.get("description") or "")[:600]
        hits.append(SearchHit(url=cu, title=title, snippet=snippet, rank=len(hits)))
        if len(hits) >= max_results:
            break
    return hits


async def self_test(client: HttpClient) -> str:
    """One-shot ping at startup; returns a human-readable status string."""
    if not is_enabled():
        return "skipped (RESEARCH_BRAVE_API_KEY not set)"
    url = f"https://{_HOST}/res/v1/web/search?q=test&count=3"
    try:
        resp = await get_with_backoff(
            client, url, host=_HOST, headers=_headers(), max_attempts=1,
        )
    except Exception as e:  # noqa: BLE001
        return f"unreachable: {e}"
    if resp is None:
        return "unreachable (no response)"
    if resp.status_code == 401:
        return "HTTP 401 -- API key invalid or revoked"
    if resp.status_code == 429:
        return "HTTP 429 -- free tier rate limited (2k/month exceeded?)"
    if resp.status_code >= 400:
        return f"HTTP {resp.status_code}"
    try:
        data = resp.json()
    except Exception:
        return "200 but response is not JSON (Brave API contract changed?)"
    web = (data.get("web") or {}).get("results") or []
    return f"ok ({len(web)} hits for 'test')"


__all__ = ["is_enabled", "search_brave", "self_test"]
