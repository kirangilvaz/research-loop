"""Search backends and a multi-backend async dispatcher.

The dispatcher runs every available backend in parallel for a single query,
merges their hits (dedup by canonical URL, first-seen wins for title/snippet,
rank stays sequential), and returns the merged list plus per-backend stats.

Backends that aren't configured (no env var for SearXNG / Brave) are silently
skipped at dispatch time but are explicitly announced at startup via
`print_backend_status()` so the user can confirm what's actually active.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Tuple

from .. import config
from ..browser.http_fetch import HttpClient
from . import brave as _brave_mod
from . import duckduckgo as _ddg_mod
from . import searxng as _searxng_mod
from .brave import is_enabled as brave_enabled, search_brave
from .duckduckgo import SearchHit, search_duckduckgo
from .searxng import is_enabled as searxng_enabled, search_searxng


Backend = Callable[[HttpClient, str, int], Awaitable[List[SearchHit]]]


@dataclass
class SearchStats:
    """Per-backend outcome for a single query."""
    name: str
    enabled: bool
    hits: int = 0
    status: str = "skipped"  # "ok", "empty", "error", "skipped"
    error: str = ""


@dataclass
class SearchResult:
    hits: List[SearchHit] = field(default_factory=list)
    per_backend: List[SearchStats] = field(default_factory=list)

    def summary_line(self) -> str:
        parts = []
        for s in self.per_backend:
            if not s.enabled:
                parts.append(f"{s.name}=skipped")
            elif s.status == "ok":
                parts.append(f"{s.name}={s.hits}")
            elif s.status == "empty":
                parts.append(f"{s.name}=0")
            elif s.status == "error":
                parts.append(f"{s.name}=err")
            else:
                parts.append(f"{s.name}={s.status}")
        return " ".join(parts) or "(no backends)"


def _enabled_backends() -> List[Tuple[str, Backend, bool]]:
    """Return [(name, fn, enabled_flag)] for ALL backends, in priority order.

    Optional backends (SearXNG, Brave) are listed FIRST so their results get
    the first rank slots in the merged output. They're also marked
    `enabled=False` when env vars are missing, so callers can log them as
    skipped instead of silently dropping them.
    """
    return [
        ("searxng", search_searxng, searxng_enabled()),
        ("brave", search_brave, brave_enabled()),
        ("ddg", search_duckduckgo, True),
    ]


def print_backend_status() -> None:
    """Print which search backends are enabled. Call once at startup."""
    backends = _enabled_backends()
    active = [name for name, _, enabled in backends if enabled]
    skipped = [name for name, _, enabled in backends if not enabled]
    print("Search backends:")
    for name, _, enabled in backends:
        if name == "searxng":
            detail = f"({config.SEARXNG_URL})" if enabled else "(set RESEARCH_SEARXNG_URL to enable)"
        elif name == "brave":
            detail = "(env: RESEARCH_BRAVE_API_KEY)" if enabled else "(set RESEARCH_BRAVE_API_KEY to enable)"
        else:
            detail = "(html.duckduckgo.com, no key required)"
        flag = "ENABLED " if enabled else "skipped "
        print(f"  [{flag}] {name:<8} {detail}")
    if not active:
        print("  WARNING: no search backends are enabled; discovery will rely on link harvest + LLM suggestions only.")
    elif active == ["ddg"]:
        print("  Only DuckDuckGo is active. For long runs, consider self-hosted SearXNG "
              "(docker run -d -p 8888:8080 searxng/searxng) and export RESEARCH_SEARXNG_URL=http://localhost:8888.")


async def search(
    client: HttpClient,
    query: str,
    max_results: int = config.DDG_RESULTS_PER_QUERY,
) -> SearchResult:
    """Run every enabled backend concurrently and return a SearchResult.

    The merged hits are deduplicated by canonical URL (first-seen wins).
    `per_backend` records what every backend returned, including disabled ones
    (as `enabled=False, status="skipped"`), so the caller can log diagnostics
    without having to consult the env separately.
    """
    backends = _enabled_backends()
    if not query or not query.strip():
        return SearchResult(
            hits=[],
            per_backend=[
                SearchStats(name=name, enabled=enabled)
                for name, _, enabled in backends
            ],
        )

    async def _one(name: str, fn: Backend, enabled: bool) -> Tuple[str, SearchStats, List[SearchHit]]:
        stats = SearchStats(name=name, enabled=enabled)
        if not enabled:
            return name, stats, []
        try:
            hits = await fn(client, query, max_results)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            stats.status = "error"
            stats.error = str(e)[:160]
            print(f"  [{name}] backend error: {e}")
            return name, stats, []
        stats.hits = len(hits)
        stats.status = "ok" if hits else "empty"
        return name, stats, hits

    coros = [_one(name, fn, enabled) for name, fn, enabled in backends]
    runs = await asyncio.gather(*coros, return_exceptions=False)

    per_backend: List[SearchStats] = []
    merged: List[SearchHit] = []
    seen: set = set()
    backend_cap = max_results * max(1, len([s for s in runs if s[1].hits > 0]))
    for _, stats, hits in runs:
        per_backend.append(stats)
        for hit in hits:
            if hit.url in seen:
                continue
            seen.add(hit.url)
            merged.append(SearchHit(url=hit.url, title=hit.title, snippet=hit.snippet, rank=len(merged)))
            if len(merged) >= backend_cap:
                break
    return SearchResult(hits=merged, per_backend=per_backend)


def print_backend_self_tests() -> None:
    """Run a one-shot ping against each enabled backend and print results.

    Called once at startup right after `print_backend_status()`. Each
    backend module exposes an async `self_test(client)` that returns a
    one-line human-readable status string. We collect them in parallel
    inside a single asyncio.run so the startup delay is roughly one
    HTTP RTT, not three.
    """
    backends = [
        ("searxng", _searxng_mod.self_test, _searxng_mod.is_enabled()),
        ("brave", _brave_mod.self_test, _brave_mod.is_enabled()),
        ("ddg", _ddg_mod.self_test, True),
    ]
    print("Search backend self-tests:")
    if not any(enabled for _, _, enabled in backends):
        print("  (no backends enabled)")
        return

    async def _run_all() -> List[Tuple[str, str]]:
        async with HttpClient() as client:
            async def _one(name: str, fn, enabled: bool) -> Tuple[str, str]:
                if not enabled:
                    return name, "skipped (not configured)"
                try:
                    return name, await fn(client)
                except Exception as e:  # noqa: BLE001
                    return name, f"self-test crashed: {type(e).__name__}: {e}"
            return await asyncio.gather(*(_one(*b) for b in backends))

    try:
        results = asyncio.run(_run_all())
    except Exception as e:  # noqa: BLE001
        print(f"  self-test runner failed: {e}")
        return

    for name, msg in results:
        low = msg.lower()
        # Classify into ok / warn buckets for at-a-glance scanning.
        if msg.startswith("skipped"):
            marker = "[skip]"
        elif "ok (" in low or low.startswith("ok ") or " ok " in low:
            marker = "[ok]  "
        else:
            marker = "[WARN]"
        print(f"  {marker} {name:<8} -> {msg}")


__all__ = [
    "SearchHit",
    "SearchResult",
    "SearchStats",
    "print_backend_status",
    "print_backend_self_tests",
    "search",
    "search_duckduckgo",
    "search_searxng",
    "search_brave",
]
