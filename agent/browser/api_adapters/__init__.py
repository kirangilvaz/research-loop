"""Site-specific API adapters: Tier 0 of the fetch ladder.

For well-known hosts (Wikipedia, arXiv, OpenAlex, Crossref, Hacker News, RSS
feeds, sitemap.xml) we bypass HTML entirely and pull clean structured data
from the site's documented API. This both improves data quality (no nav
chrome, no anti-bot HTML noise) and removes anti-bot risk.

The registry is intentionally tiny: each adapter is a coroutine that takes
the URL and a shared `HttpClient`, plus an optional asyncio deadline, and
returns an `AdapterResult` if it can handle the URL (or `None` to let the
ladder fall through to Tier 1).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from ..http_fetch import HttpClient
from . import arxiv, crossref, hackernews, openalex, rss, wikipedia


@dataclass
class AdapterResult:
    text: str
    title: str = ""
    links: List[str] = field(default_factory=list)
    source: str = ""  # e.g. "wikipedia", "arxiv"


Handler = Callable[[str, HttpClient, Optional[float]], Awaitable[Optional[AdapterResult]]]


_HANDLERS: List[Handler] = [
    wikipedia.try_handle,
    arxiv.try_handle,
    openalex.try_handle,
    crossref.try_handle,
    hackernews.try_handle,
    rss.try_handle,
]


async def try_api(
    url: str,
    client: HttpClient,
    *,
    deadline: Optional[float] = None,
) -> Optional[AdapterResult]:
    """Dispatch a URL to the first matching adapter.

    Each adapter inspects the URL and either returns an AdapterResult or
    returns None to indicate "not my host / pattern". Adapters never raise
    out of `try_api`; any failure surfaces as None.
    """
    for handler in _HANDLERS:
        try:
            res = await handler(url, client, deadline)
        except asyncio.CancelledError:
            raise
        except Exception:
            res = None
        if res is not None:
            return res
    return None


__all__ = ["AdapterResult", "try_api"]
