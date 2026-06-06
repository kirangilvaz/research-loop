"""Research node: pop top candidates, fetch with the async ladder, harvest links.

The orchestrator stays synchronous; each iteration wraps an asyncio.run() over
a coroutine that fans out the AsyncFetcher across the popped candidates using
asyncio.as_completed and per-host semaphores from the shared HttpClient.

Disk writes (memory.mark_visited, record_page, run-log append) go through
asyncio.to_thread so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Tuple

from . import config
from .browser.fetcher import AsyncFetcher, FetchResult, FetchStatus
from .browser.http_fetch import HttpClient
from .browser.stealth import AsyncStealthBrowser, HostRateLimiter, RobotsCache
from .memory import Candidate, TopicMemory
from .ranking import CandidateInput, push_to_frontier, score_candidates
from .state import State
from .utils import canonicalize_url, host_of, retry_on_transient_oserror


def init_research_file(state: State) -> str:
    """Create a fresh JSONL log of the current run for debugging / inspection."""
    if state.memory is not None:
        run_id = state.ensure_run_id()
        run_dir = state.memory.begin_run(run_id)
        path = str(run_dir / "events.jsonl")
    else:
        fd, path = tempfile.mkstemp(prefix="research_", suffix=".jsonl")
        os.close(fd)
    state.research_file = path
    _append_event(path, {
        "type": "meta",
        "task": state.task,
        "topic_slug": state.topic_slug,
        "run_id": state.run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"  Run event log: {path}\n")
    return path


def _append_event(path: str, record: dict) -> None:
    if not path:
        return
    line = json.dumps(record, ensure_ascii=False) + "\n"

    def _do_append() -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    retry_on_transient_oserror(_do_append)


def _record_result(
    state: State,
    cand: Candidate,
    result: FetchResult,
) -> Optional[dict]:
    """Persist a fetch result to memory + run log. Returns the page dict if OK."""
    memory = state.memory
    host = host_of(cand.url)
    extra = {
        "iteration": state.iteration,
        "route": result.route,
        "elapsed": round(result.elapsed_seconds, 2),
        "host": host,
        "score": cand.score,
        "source": cand.source,
    }

    if result.status == FetchStatus.OK:
        if memory is not None:
            memory.mark_visited(cand.url, "ok", **extra)
            memory.bump_domain(host, "ok")
            memory.record_page(
                url=cand.url,
                title=result.title,
                text=result.text,
                links=result.links,
                snippet_chars=config.SNIPPET_CHARS,
            )
        state.successful_sources.append(cand.url)
        _append_event(state.research_file, {
            "type": "page",
            "status": "ok",
            "url": cand.url,
            "title": result.title[:200],
            "length": len(result.text),
            "links": len(result.links),
            "route": result.route,
            "iteration": state.iteration,
        })
        return {
            "url": cand.url,
            "title": result.title,
            "text": result.text,
            "links": result.links,
        }

    kind = "blocked" if result.status == FetchStatus.BLOCKED else "fail"
    if memory is not None:
        memory.mark_visited(cand.url, result.status.value, reason=result.reason[:240], **extra)
        memory.bump_domain(host, "blocked" if kind == "blocked" else "fail")
    state.failed_sources.append(cand.url)
    _append_event(state.research_file, {
        "type": "page",
        "status": result.status.value,
        "url": cand.url,
        "reason": result.reason[:240],
        "route": result.route,
        "iteration": state.iteration,
    })
    return None


def backflow_links(state: State, productive_pages: List[dict]) -> int:
    """Score outbound links from productive pages and push the best to frontier.

    `productive_pages` must already be filtered to pages that contributed at
    least one claim. This is enforced by the caller (synthesize_node) rather
    than here so the link-harvest logic stays focused on URL handling.
    """
    if state.memory is None or not productive_pages:
        return 0
    inputs: List[CandidateInput] = []
    seen: set = set()
    for page in productive_pages:
        for link in page.get("links") or []:
            cu = canonicalize_url(link)
            if not cu or cu in seen:
                continue
            seen.add(cu)
            inputs.append(
                CandidateInput(
                    url=cu,
                    title="",
                    snippet=f"linked from {page.get('url', '')}",
                    source="link-harvest:backflow",
                )
            )
    if not inputs:
        return 0
    scored = score_candidates(inputs, state.task, state.facets, state.missing, state.memory)
    pushed = push_to_frontier(
        scored,
        state.memory,
        top_k=config.FRONTIER_PUSH_TOP_K,
        per_host_cap=config.FRONTIER_PER_HOST_CAP,
    )
    return len(pushed)


async def _fetch_one(
    fetcher: AsyncFetcher,
    cand: Candidate,
    sem: asyncio.Semaphore,
    rate_limiter: HostRateLimiter,
) -> Tuple[Candidate, FetchResult]:
    async with sem:
        try:
            await rate_limiter.wait_async(host_of(cand.url))
        except Exception:
            pass
        try:
            result = await fetcher.fetch(cand.url)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            result = FetchResult(
                url=cand.url,
                status=FetchStatus.ERROR,
                route="error",
                reason=str(e),
            )
        return cand, result


async def _research_async(
    state: State,
    candidates: List[Candidate],
    rate_limiter: HostRateLimiter,
    workers: int,
) -> Tuple[List[dict], int]:
    successful_pages: List[dict] = []
    new_success = 0

    browser: Optional[AsyncStealthBrowser] = None
    if config.BROWSER_DEFAULT != "off":
        browser = AsyncStealthBrowser()

    sem = asyncio.Semaphore(max(1, workers))
    robots = RobotsCache()
    try:
        async with HttpClient() as http_client:
            fetcher = AsyncFetcher(
                http_client=http_client,
                browser=browser,
                robots=robots,
                memory=state.memory,
            )
            tasks = [
                asyncio.create_task(_fetch_one(fetcher, cand, sem, rate_limiter))
                for cand in candidates
            ]
            try:
                for fut in asyncio.as_completed(tasks):
                    time_left = state.time_left()
                    if time_left is not None and time_left <= 0:
                        print("  Deadline hit while fetching; cancelling remaining.")
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        break
                    try:
                        cand, result = await fut
                    except asyncio.CancelledError:
                        continue
                    except Exception as e:  # noqa: BLE001
                        print(f"  worker error: {e}")
                        continue
                    page = await asyncio.to_thread(_record_result, state, cand, result)
                    if page:
                        successful_pages.append(page)
                        new_success += 1
                        print(f"    ok ({result.route}, {len(result.text)} chars, "
                              f"{len(result.links)} links): {cand.url}")
                    else:
                        print(f"    {result.status.value} ({result.route}): {cand.url} -- "
                              f"{(result.reason or '')[:120]}")
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if browser is not None and browser.started:
            try:
                await browser.close()
            except Exception:
                pass
    return successful_pages, new_success


def research(
    state: State,
    rate_limiter: HostRateLimiter,
    workers: int = config.WORKERS_DEFAULT,
) -> List[dict]:
    """Pop top candidates and fetch them via the async tiered ladder.

    Returns the list of successful page records (each {url,title,text,links}).
    The browser is constructed inside the iteration's asyncio.run scope so
    Playwright objects remain bound to a live event loop, and only ever
    spawned on demand by the tier-2 escalation path.
    """
    if state.memory is None:
        print("  No memory configured; skipping research.")
        return []

    pop_n = max(1, state.queue_pop_size)
    candidates = state.memory.pop_top_n(pop_n, per_host_cap=config.FRONTIER_PER_HOST_CAP)
    if not candidates:
        print("  Frontier is empty; nothing to research this iteration.\n")
        return []

    workers = max(1, min(workers, len(candidates)))
    print(f"Researching (iteration {state.iteration}): "
          f"fetching {len(candidates)} URLs (workers={workers}, "
          f"per-URL budget={config.URL_TOTAL_DEADLINE:.1f}s)...")
    for c in candidates:
        print(f"    -> {c.score:.3f} {c.url}")

    successful_pages, new_success = asyncio.run(
        _research_async(state, candidates, rate_limiter, workers)
    )

    # Backflow is now deferred to synthesize_node so we can filter to pages
    # that actually produced claims (avoids off-topic seeds dragging the
    # frontier into rabbit holes).
    print(
        f"  Iteration {state.iteration}: {new_success} OK / "
        f"{len(candidates) - new_success} fail. "
        f"Frontier size now: {state.memory.frontier_size()} "
        f"(backflow runs in synthesize).\n"
    )
    return successful_pages
