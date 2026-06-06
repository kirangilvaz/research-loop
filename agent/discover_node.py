"""Discover node: continuously expand the candidate frontier.

Each iteration we pull candidates from four streams:
    1. Multi-backend web search (DDG always; SearXNG + Brave when configured)
       on queries derived from `state.missing` + facets, fanned out in parallel.
    2. Outbound links harvested from previously-visited pages (memory).
    3. LLM-suggested URLs (still useful, especially for niche/primary sources).
    4. Memory replay: previously-trusted domains for this topic.

All candidates are scored by `ranking.py`, deduped, capped per host, and pushed
into the persistent frontier.
"""

from __future__ import annotations

import asyncio
import random
from typing import List, Optional, Sequence

from . import config
from .browser.http_fetch import HttpClient
from .browser.stealth import HostRateLimiter, AsyncStealthBrowser
from .health import bucket_source
from .llm import is_placeholder, llm_chat, llm_chat_json
from .ranking import (
    CandidateInput,
    push_to_frontier,
    score_candidates,
    llm_rerank_topk,
)
from .search import search as multi_search
from .state import State
from .utils import canonicalize_url, host_of, parse_urls


_LLM_URLS_PER_CALL = 6
_MAX_QUERIES_PER_DISCOVER = 5


def _build_query_batch(state: State) -> List[str]:
    missing = [m for m in (state.missing or []) if m]
    facets = [f for f in (state.facets or []) if f]
    base_queries: List[str] = []

    if missing:
        for m in missing[: _MAX_QUERIES_PER_DISCOVER]:
            base_queries.append(m)

    if state.iteration <= 1 and state.initial_queries:
        for q in state.initial_queries:
            if q not in base_queries:
                base_queries.append(q)

    if len(base_queries) < _MAX_QUERIES_PER_DISCOVER and facets:
        for f in facets:
            q = f"{state.task} {f}"
            if q not in base_queries:
                base_queries.append(q)
            if len(base_queries) >= _MAX_QUERIES_PER_DISCOVER:
                break

    if not base_queries:
        base_queries = [state.task]

    out = []
    seen = set()
    for q in base_queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    random.shuffle(out)
    return out[: _MAX_QUERIES_PER_DISCOVER]


async def _gather_search_candidates(
    state: State,
    client: HttpClient,
) -> List[CandidateInput]:
    queries = _build_query_batch(state)
    print(f"  Search queries ({len(queries)}):")
    for q in queries:
        print(f"    ? {q}")
    if not queries:
        return []

    async def _one(q: str) -> List[CandidateInput]:
        try:
            result = await multi_search(client, q)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"    [search err] {q[:60]!r}: {e}")
            return []
        # Per-query backend breakdown so user can see who returned what.
        print(f"    [{q[:40]:<40}] {result.summary_line()} -> merged={len(result.hits)}")
        return [
            CandidateInput(
                url=h.url,
                title=h.title,
                snippet=h.snippet,
                source=f"search:{q[:60]}",
            )
            for h in result.hits
        ]

    batches = await asyncio.gather(*(_one(q) for q in queries), return_exceptions=False)
    out: List[CandidateInput] = []
    for batch in batches:
        out.extend(batch)
    return out


def _harvest_link_candidates(state: State) -> List[CandidateInput]:
    if state.memory is None:
        return []
    pages = state.memory.all_pages()
    seen = set()
    out: List[CandidateInput] = []
    for page in pages:
        for link in page.get("links") or []:
            link = canonicalize_url(link)
            if not link or link in seen:
                continue
            seen.add(link)
            out.append(
                CandidateInput(
                    url=link,
                    title="",
                    snippet=f"linked from {page.get('url', '')}",
                    source="link-harvest",
                )
            )
    return out


def _llm_suggested_candidates(state: State) -> List[CandidateInput]:
    facets_block = "\n".join(f"- {f}" for f in state.facets) or "(none)"
    missing_block = "\n".join(f"- {m}" for m in state.missing) or "(no specific gaps)"
    visited_block = (
        "\n".join(
            f"- {h}" for h in (state.memory.unique_supporting_domains() if state.memory else [])
        )
        or "(none yet)"
    )
    prompt = (
        f"Research task: {state.task}\n"
        f"Iteration: {state.iteration}\n\n"
        "Facets to cover:\n"
        f"{facets_block}\n\n"
        "Currently missing / would strengthen the answer:\n"
        f"{missing_block}\n\n"
        "Domains we've already used:\n"
        f"{visited_block}\n\n"
        f"Suggest {_LLM_URLS_PER_CALL} NEW, reliable URLs (full https links) "
        "that would best help fill the gaps. Prefer primary sources, official "
        "data portals, reputable analyses, and niche outlets. Avoid duplicating "
        "the domains above. Return ONLY a plain list of URLs, one per line, no commentary."
    )
    raw = llm_chat(prompt, temperature=0.5)
    if is_placeholder(raw):
        return []
    out: List[CandidateInput] = []
    seen = set()
    for u in parse_urls(raw):
        cu = canonicalize_url(u)
        if not cu or cu in seen:
            continue
        seen.add(cu)
        out.append(CandidateInput(url=cu, source="llm-suggest"))
    return out


def _memory_replay_candidates(state: State) -> List[CandidateInput]:
    if state.memory is None:
        return []
    out: List[CandidateInput] = []
    for u in state.memory.replay_good_urls(min_score=0.5):
        if state.memory.is_visited(u):
            continue
        out.append(CandidateInput(url=u, source="memory-replay"))
    return out


def _summarize_sources(cands: Sequence[CandidateInput]) -> str:
    by_source: dict = {}
    for c in cands:
        key = c.source.split(":", 1)[0] if c.source else "?"
        by_source[key] = by_source.get(key, 0) + 1
    if not by_source:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))


async def _discover_async(state: State) -> List[CandidateInput]:
    async with HttpClient() as client:
        pool: List[CandidateInput] = []
        search_cands = await _gather_search_candidates(state, client)
        pool.extend(search_cands)
    pool.extend(_harvest_link_candidates(state))
    if state.memory is None or state.memory.frontier_size() < config.FRONTIER_PUSH_TOP_K:
        pool.extend(_llm_suggested_candidates(state))
    pool.extend(_memory_replay_candidates(state))
    return pool


def discover_sources(
    state: State,
    browser: Optional[AsyncStealthBrowser] = None,
    rate_limiter: Optional[HostRateLimiter] = None,
) -> int:
    """Expand the persistent frontier. Returns number of NEW candidates pushed.

    `browser` and `rate_limiter` are accepted for backwards compatibility with
    the orchestrator but are no longer needed for search (which now runs over
    httpx). They stay in the signature so external callers don't break.
    """
    del browser, rate_limiter  # unused; kept for signature compatibility
    print(f"Discovering sources (iteration {state.iteration})...")

    pool = asyncio.run(_discover_async(state))

    # Record source mix for health monitoring (search-stalled detector etc).
    mix: dict = {}
    for c in pool:
        b = bucket_source(c.source)
        mix[b] = mix.get(b, 0) + 1
    state.source_mix_history.append(mix)
    if len(state.source_mix_history) > 16:
        state.source_mix_history = state.source_mix_history[-16:]

    print(f"  Raw candidates: {len(pool)} ({_summarize_sources(pool)})")
    if not pool or state.memory is None:
        print(f"  Frontier size after discovery: "
              f"{state.memory.frontier_size() if state.memory else 0}\n")
        return 0

    scored = score_candidates(
        pool, state.task, state.facets, state.missing, state.memory
    )
    if not scored:
        print(f"  After scoring: 0 viable candidates.\n")
        return 0

    if config.LLM_RERANK_TOP_K > 0:
        scored = llm_rerank_topk(
            scored, state.task, state.facets, state.missing, top_k=config.LLM_RERANK_TOP_K
        )

    pushed = push_to_frontier(scored, state.memory)
    print(f"  Pushed {len(pushed)} new candidate(s) to frontier "
          f"(top score={pushed[0].score:.3f})." if pushed
          else "  No new candidates passed dedupe.")
    if pushed:
        for s in pushed[:6]:
            print(f"    + {s.score:.3f} {s.url}")

    print(f"  Frontier size: {state.memory.frontier_size()}, "
          f"visited: {state.memory.visited_count()}.\n")
    return len(pushed)


def creativity_boost(state: State) -> None:
    """Generate radically different angles when the agent is stuck."""
    if state.memory is None:
        return
    print(f"Creativity boost #{state.creativity_boosts + 1}: searching new angles...")

    prompt = (
        "The research agent is stuck and needs RADICALLY different angles. "
        "Output STRICT JSON only.\n\n"
        f"Task: {state.task}\n"
        f"Facets: {', '.join(state.facets) or '(none)'}\n"
        f"Already-used domains: {', '.join(state.memory.unique_supporting_domains()) or '(none)'}\n\n"
        "Generate fresh ideas that depart from the current approach:\n"
        "- contrarian framings (what if the opposite is true?)\n"
        "- different time windows (older archives, future projections)\n"
        "- adjacent sub-topics or related fields\n"
        "- niche outlets, academic / government / industry sources\n"
        "- non-English angles (translated terms)\n"
        "- methodological angles (data sources, datasets, methodology papers)\n\n"
        "Schema:\n"
        '{ "queries": ["6-10 brand-new search queries"],\n'
        '  "missing": ["3-6 phrases describing what we still need"] }'
    )
    parsed = llm_chat_json(prompt, temperature=0.85)
    if not isinstance(parsed, dict):
        print("  creativity LLM call failed; nudging missing list manually.")
        state.missing = (state.missing or []) + [
            f"contrarian view on {state.task}",
            f"primary data sources for {state.task}",
            f"academic research on {state.task}",
        ]
    else:
        new_queries = [str(x).strip() for x in (parsed.get("queries") or []) if str(x).strip()]
        new_missing = [str(x).strip() for x in (parsed.get("missing") or []) if str(x).strip()]
        if new_queries:
            for q in new_queries:
                if q not in state.initial_queries:
                    state.initial_queries.append(q)
            print(f"  Added {len(new_queries)} new angle queries.")
        if new_missing:
            for m in new_missing:
                if m not in state.missing:
                    state.missing.append(m)
            print(f"  Added {len(new_missing)} new missing-info phrases.")

    state.creativity_boosts += 1
    state.stagnant_streak = 0
