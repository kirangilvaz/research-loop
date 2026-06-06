"""Synthesize node: turn fetched pages into atomic claims and next-iteration queries.

For each page collected this iteration, we run claim extraction (which dedupes
+ corroborates against persistent memory). We then ask the LLM to look at the
facet-coverage map and emit *new search queries* targeted at uncovered facets;
those land in `state.missing` and feed `discover` next pass.

When the open web is thin, we also append a model-knowledge note to the run
log so the summarizer has supplementary context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence

from . import config
from .claims import facet_coverage, ingest_page_into_claims
from .llm import is_placeholder, llm_chat, llm_chat_json
from .memory import TopicMemory
from .research_node import backflow_links
from .state import State
from .utils import shorten


def _coverage_summary(coverage: Dict[str, Dict[str, object]]) -> List[str]:
    lines = []
    for facet, info in coverage.items():
        domains = info.get("domains") or []
        support = info.get("support_count", len(info.get("supporting_claims") or []))
        marker = "OK " if isinstance(domains, list) and len(domains) >= 2 else "thin"
        lines.append(
            f"  [{marker}] {facet}: {support} claims / "
            f"{len(domains) if isinstance(domains, list) else 0} domains"
        )
    return lines


def _emit_next_queries(state: State) -> None:
    """Ask the LLM for next-iteration search queries targeting uncovered facets."""
    coverage = facet_coverage(state.memory, state.facets) if state.memory else {}
    weak_facets = [
        f for f, info in coverage.items()
        if not info.get("domains") or len(info.get("domains") or []) < 2
    ]
    if not weak_facets:
        return

    domains_used = ", ".join((state.memory.unique_supporting_domains() if state.memory else [])[:20])
    prompt = (
        "You are guiding the next round of web research. Output STRICT JSON only.\n\n"
        f"Task: {state.task}\n"
        f"Weakly covered facets:\n" + "\n".join(f"- {f}" for f in weak_facets) + "\n\n"
        f"Domains already used: {domains_used or '(none)'}\n\n"
        "Generate concrete, search-friendly queries that would close those gaps. "
        "Prefer specific phrasings (named entities, dates, metrics, document types).\n\n"
        "Schema:\n"
        '{ "missing_phrases": ["short phrases describing what info we still need"],\n'
        '  "queries": ["6-10 search queries"] }'
    )
    parsed = llm_chat_json(prompt, temperature=0.5)
    if not isinstance(parsed, dict):
        return

    missing_phrases = [str(x).strip() for x in (parsed.get("missing_phrases") or []) if str(x).strip()]
    queries = [str(x).strip() for x in (parsed.get("queries") or []) if str(x).strip()]

    if missing_phrases:
        state.missing = list(dict.fromkeys((state.missing or []) + missing_phrases))[:24]
    if queries:
        for q in queries:
            if q not in state.initial_queries:
                state.initial_queries.append(q)


def _maybe_append_model_note(state: State) -> None:
    """When the corpus is thin, add a synthesized note from the LLM's own knowledge."""
    if state.memory is None:
        return
    page_count = len(state.memory.all_pages())
    if page_count >= 5:
        return

    missing_block = "\n".join(f"- {m}" for m in (state.missing or [])) or "(no specific gaps)"
    prompt = (
        f"User's task: {state.task}\n"
        f"Iteration: {state.iteration}\n"
        f"Pages collected so far: {page_count}\n\n"
        "The web corpus is thin. Write detailed factual research notes from your "
        "own knowledge that would help answer the task and address the missing "
        "info below. Be concrete (names, dates, numbers, events) and mark "
        "uncertain figures with '(approx)'.\n\n"
        "Missing info:\n"
        f"{missing_block}\n\n"
        "Output prose only, no lists of URLs."
    )
    note = llm_chat(prompt, temperature=0.5)
    if is_placeholder(note) or len(note) < 200:
        return
    note_url = f"model-note:{state.run_id}:{state.iteration}"
    counts = ingest_page_into_claims(
        memory=state.memory,
        task=state.task,
        facets=state.facets,
        page_url=note_url,
        page_title=f"Model synthesis note (iter {state.iteration})",
        page_text=note,
    )
    print(f"  Model-note claims: extracted={counts['extracted']}, "
          f"added={counts['added']}, merged={counts['merged']}.")


def synthesize_knowledge(state: State, fetched_pages: Sequence[dict]) -> int:
    """Run claim extraction over freshly fetched pages. Returns # of new claims added.

    After all pages are ingested we backflow outbound links ONLY from pages
    that contributed >=1 atomic claim. This prevents off-topic seeds (e.g. a
    semiconductor query that lands on a CISA cybersecurity page) from
    flooding the frontier with their navigation chrome -- those pages would
    succeed the fetch but produce no claims, so we no longer harvest their
    links.
    """
    print(f"Synthesizing knowledge (iteration {state.iteration})...")

    if state.memory is None:
        return 0

    total_added = 0
    total_merged = 0
    productive_pages: List[dict] = []
    unproductive = 0
    for page in fetched_pages:
        url = page.get("url") or ""
        title = page.get("title") or ""
        text = page.get("text") or ""
        if not text:
            continue
        counts = ingest_page_into_claims(
            memory=state.memory,
            task=state.task,
            facets=state.facets,
            page_url=url,
            page_title=title,
            page_text=text,
        )
        total_added += counts["added"]
        total_merged += counts["merged"]
        # Productive = produced at least one claim (new or merged into existing).
        # We accept "merged" because corroborating an existing claim is still
        # a signal that the page is on-topic; it just wasn't novel.
        if counts.get("extracted", 0) > 0 or counts.get("merged", 0) > 0:
            productive_pages.append(page)
        else:
            unproductive += 1
        print(f"  {url}: extracted={counts['extracted']}, "
              f"added={counts['added']}, merged={counts['merged']}.")

    state.new_claims_history.append(total_added)
    if len(state.new_claims_history) > 32:
        state.new_claims_history = state.new_claims_history[-32:]

    # Record productivity for health monitoring (rabbit-hole/low-productivity detectors).
    state.productivity_history.append({
        "fetched": int(len(fetched_pages)),
        "productive": int(len(productive_pages)),
    })
    if len(state.productivity_history) > 16:
        state.productivity_history = state.productivity_history[-16:]

    coverage = facet_coverage(state.memory, state.facets) if state.facets else {}
    if coverage:
        print("  Facet coverage:")
        for line in _coverage_summary(coverage):
            print(line)

    _emit_next_queries(state)
    _maybe_append_model_note(state)

    pushed = backflow_links(state, productive_pages)
    print(
        f"  Iteration {state.iteration}: +{total_added} new claims "
        f"(merged into existing: {total_merged}). "
        f"Total claims in memory: {state.memory.claim_count()}.\n"
        f"  Backflow: harvested links from {len(productive_pages)} productive "
        f"page(s), skipped {unproductive} unproductive; "
        f"pushed {pushed} new candidate(s).\n"
    )
    return total_added
