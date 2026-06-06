"""Async tiered fetch ladder.

Per URL we walk:

    Tier 0: api_adapters   (wikipedia/arxiv/openalex/crossref/hn/rss/sitemap)
    Tier 1: httpx + Trafilatura extraction + quality gate
    Tier 2: Playwright pooled stealth context + Trafilatura
    Tier 3: r.jina.ai readability proxy
    Tier 4: Wayback Machine closest snapshot

Each URL gets a strict total wall-clock budget (`config.URL_TOTAL_DEADLINE`)
that is split across tiers. If memory has a `preferred_tier` recorded for the
host, we skip ahead to it (then continue down the ladder from there if it
fails). After every attempt we update memory's per-tier counters so the next
visit gets faster.

A `FetchResult` is returned with the same shape as before so the rest of the
pipeline (research_node, summarize_node, html_node) doesn't change.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from urllib.parse import quote

from .. import config
from ..utils import host_of
from .api_adapters import AdapterResult, try_api
from .extract import ContentExtraction, extract_content, quality_ok
from .http_fetch import HttpClient, get_with_backoff, polite_headers
from .stealth import AsyncStealthBrowser, RobotsCache, humanize, settle_async


class FetchStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    ERROR = "error"
    DISALLOWED = "disallowed"


@dataclass
class FetchResult:
    url: str
    status: FetchStatus = FetchStatus.ERROR
    route: str = ""
    title: str = ""
    text: str = ""
    html: str = ""
    links: List[str] = field(default_factory=list)
    reason: str = ""
    elapsed_seconds: float = 0.0


_TIER_ORDER = ("api", "http", "browser", "jina", "wayback")


def _now_loop() -> float:
    return asyncio.get_event_loop().time()


def _remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return deadline - _now_loop()


def _is_blocked_text(text: str) -> bool:
    """Quick anti-bot smell-check. Real quality is gated by `quality_ok`."""
    if not text:
        return True
    return not quality_ok(text)


class AsyncFetcher:
    """Coordinates the tier ladder for a single URL.

    Bind it to a shared `HttpClient`, an optional `AsyncStealthBrowser` (only
    instantiated for tier 2), a `RobotsCache`, and the `TopicMemory` we use
    to record per-host tier-success learning.
    """

    def __init__(
        self,
        http_client: HttpClient,
        browser: Optional[AsyncStealthBrowser] = None,
        robots: Optional[RobotsCache] = None,
        memory: Optional["object"] = None,
    ) -> None:
        self.http_client = http_client
        self.browser = browser
        self.robots = robots or RobotsCache()
        self.memory = memory

    async def fetch(self, url: str) -> FetchResult:
        started = time.monotonic()
        deadline = _now_loop() + config.URL_TOTAL_DEADLINE

        allowed = await self.robots.allowed_async(url)
        if not allowed:
            return FetchResult(
                url=url,
                status=FetchStatus.DISALLOWED,
                route="robots",
                reason="disallowed by robots.txt",
                elapsed_seconds=time.monotonic() - started,
            )

        host = host_of(url)
        preferred = self._preferred_tier(host)
        tiers = _reorder(_TIER_ORDER, preferred)

        last_attempt: Optional[FetchResult] = None
        notes: List[str] = []

        for tier in tiers:
            if _remaining(deadline) is not None and _remaining(deadline) <= 0:
                notes.append(f"{tier}=deadline")
                break
            try:
                result = await self._run_tier(tier, url, deadline)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                result = FetchResult(url=url, status=FetchStatus.ERROR, route=tier, reason=f"{tier} crashed: {e}")
            self._record_tier(host, tier, result.status == FetchStatus.OK)
            if result.status == FetchStatus.OK:
                result.elapsed_seconds = time.monotonic() - started
                return result
            last_attempt = result
            notes.append(f"{tier}={result.status.value}" + (f" ({result.reason[:60]})" if result.reason else ""))

        # All tiers exhausted; return the best information we have.
        if last_attempt is None:
            return FetchResult(
                url=url,
                status=FetchStatus.ERROR,
                route="none",
                reason="no tier ran (deadline / config)",
                elapsed_seconds=time.monotonic() - started,
            )
        last_attempt.reason = "all tiers failed: " + "; ".join(notes)
        last_attempt.elapsed_seconds = time.monotonic() - started
        if last_attempt.status == FetchStatus.OK:
            last_attempt.status = FetchStatus.BLOCKED
        return last_attempt

    def _preferred_tier(self, host: str) -> Optional[str]:
        if self.memory is None or not host:
            return None
        try:
            return self.memory.preferred_tier(host)
        except Exception:
            return None

    def _record_tier(self, host: str, tier: str, success: bool) -> None:
        if self.memory is None or not host:
            return
        try:
            self.memory.record_tier(host, tier, success)
        except Exception:
            pass

    async def _run_tier(self, tier: str, url: str, deadline: float) -> FetchResult:
        if tier == "api":
            return await self._tier_api(url, deadline)
        if tier == "http":
            return await self._tier_http(url, deadline)
        if tier == "browser":
            return await self._tier_browser(url, deadline)
        if tier == "jina":
            return await self._tier_jina(url, deadline)
        if tier == "wayback":
            return await self._tier_wayback(url, deadline)
        return FetchResult(url=url, status=FetchStatus.ERROR, route=tier, reason="unknown tier")

    async def _tier_api(self, url: str, deadline: float) -> FetchResult:
        result = FetchResult(url=url, route="api")
        try:
            adapter: Optional[AdapterResult] = await asyncio.wait_for(
                try_api(url, self.http_client, deadline=deadline),
                timeout=max(0.5, _remaining(deadline) or config.HTTP_TIMEOUT),
            )
        except asyncio.TimeoutError:
            result.reason = "api timeout"
            return result
        if adapter is None:
            result.reason = "no api adapter"
            return result
        result.route = f"api:{adapter.source or 'unknown'}"
        result.title = adapter.title
        result.text = adapter.text
        result.links = adapter.links
        if adapter.text and len(adapter.text) >= 80:
            result.status = FetchStatus.OK
        else:
            result.status = FetchStatus.BLOCKED
            result.reason = "api returned empty"
        return result

    async def _tier_http(self, url: str, deadline: float) -> FetchResult:
        result = FetchResult(url=url, route="http")
        host = host_of(url)
        try:
            resp = await get_with_backoff(
                self.http_client,
                url,
                host=host,
                max_attempts=2,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result.reason = f"http error: {e}"
            return result
        if resp is None:
            result.reason = "http transport failure"
            return result
        if resp.status_code >= 400:
            result.reason = f"http {resp.status_code}"
            if resp.status_code in (403, 429) or resp.status_code in range(500, 600):
                result.status = FetchStatus.BLOCKED
            return result
        ctype = resp.headers.get("content-type", "").lower()
        if "text/" not in ctype and "html" not in ctype and "xml" not in ctype and "json" not in ctype:
            result.reason = f"non-text content-type {ctype!r}"
            return result
        html = resp.text or ""
        result.html = html
        extracted = extract_content(html, str(resp.url))
        result.title = extracted.title
        result.text = extracted.text
        result.links = extracted.links
        if quality_ok(result.text):
            result.status = FetchStatus.OK
        else:
            result.status = FetchStatus.BLOCKED
            result.reason = f"quality gate failed ({extracted.extractor or 'no-extractor'}, {len(result.text)} chars)"
        return result

    async def _tier_browser(self, url: str, deadline: float) -> FetchResult:
        result = FetchResult(url=url, route="browser")
        if config.BROWSER_DEFAULT == "off" or self.browser is None:
            result.reason = "browser tier disabled"
            return result
        remaining = _remaining(deadline)
        if remaining is None or remaining < 2.0:
            result.reason = "no browser budget"
            return result
        nav_timeout = min(max(int(remaining * 1000) - 500, 2000), config.PAGE_NAV_TIMEOUT_MS)
        try:
            async with self.browser.acquire_context() as ctx:
                page = await ctx.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
                    await settle_async()
                    await humanize(page)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=min(4000, nav_timeout))
                    except Exception:
                        pass
                    html = await page.content()
                finally:
                    try:
                        await page.close()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result.reason = f"browser error: {e}"
            return result

        result.html = html
        extracted = extract_content(html, url)
        result.title = extracted.title
        result.text = extracted.text
        result.links = extracted.links
        if quality_ok(result.text):
            result.status = FetchStatus.OK
        else:
            result.status = FetchStatus.BLOCKED
            result.reason = f"browser content failed quality gate ({len(result.text)} chars)"
        return result

    async def _tier_jina(self, url: str, deadline: float) -> FetchResult:
        result = FetchResult(url=url, route="jina")
        proxy_url = f"{config.JINA_READER_PREFIX}{url}"
        headers = polite_headers(
            {
                "Accept": "text/plain, text/markdown",
                "X-Return-Format": "markdown",
            }
        )
        try:
            resp = await get_with_backoff(
                self.http_client,
                proxy_url,
                host="r.jina.ai",
                headers=headers,
                max_attempts=1,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result.reason = f"jina error: {e}"
            return result
        if resp is None or resp.status_code >= 400 or not resp.text:
            result.reason = f"jina http {(resp.status_code if resp else 'no-response')}"
            return result
        result.text = resp.text
        first_line = (resp.text.splitlines() or [""])[0].strip()
        result.title = first_line[:200]
        if quality_ok(result.text, min_len=200):
            result.status = FetchStatus.OK
        else:
            result.status = FetchStatus.BLOCKED
            result.reason = "jina text failed quality gate"
        return result

    async def _tier_wayback(self, url: str, deadline: float) -> FetchResult:
        result = FetchResult(url=url, route="wayback")
        api = f"https://archive.org/wayback/available?url={quote(url, safe='')}"
        try:
            avail = await get_with_backoff(
                self.http_client,
                api,
                host="archive.org",
                max_attempts=1,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result.reason = f"wayback availability error: {e}"
            return result
        if avail is None or avail.status_code >= 400:
            result.reason = "wayback availability fail"
            return result
        try:
            data = avail.json()
        except Exception:
            result.reason = "wayback non-json"
            return result
        snapshot = ((data.get("archived_snapshots") or {}).get("closest") or {})
        target = snapshot.get("url")
        if not target:
            result.reason = "no wayback snapshot"
            return result
        try:
            page_resp = await get_with_backoff(
                self.http_client,
                target,
                host="web.archive.org",
                max_attempts=1,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result.reason = f"wayback fetch error: {e}"
            return result
        if page_resp is None or page_resp.status_code >= 400 or not page_resp.text:
            result.reason = f"wayback http {(page_resp.status_code if page_resp else 'no-response')}"
            return result
        result.html = page_resp.text
        extracted = extract_content(page_resp.text, target)
        result.title = extracted.title
        result.text = extracted.text
        result.links = extracted.links
        if quality_ok(result.text):
            result.status = FetchStatus.OK
        else:
            result.status = FetchStatus.BLOCKED
            result.reason = "wayback content failed quality gate"
        return result


def _reorder(tiers: Tuple[str, ...], preferred: Optional[str]) -> List[str]:
    """If `preferred` is known, try it first then continue from where it sits."""
    if not preferred or preferred not in tiers:
        return list(tiers)
    idx = tiers.index(preferred)
    return [preferred] + [t for i, t in enumerate(tiers) if i != idx]


__all__ = [
    "AsyncFetcher",
    "FetchResult",
    "FetchStatus",
]
