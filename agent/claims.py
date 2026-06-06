"""Atomic claim extraction + dedupe.

For each freshly-fetched page we ask the LLM to pull out the most important,
self-contained, factual statements (with supporting quotes). Each claim is
embedded; near-duplicates merge in `TopicMemory`, contradictions are flagged.

Claims are the unit conviction is built on: one source = 1 support, two
independent domains = corroborated, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config
from .embeddings import cosine_matrix, embed
from .llm import is_placeholder, llm_chat_json
from .memory import TopicMemory
from .utils import host_of, shorten


@dataclass
class ExtractedClaim:
    text: str
    quote: str = ""


def _build_extraction_prompt(
    task: str,
    facets: Sequence[str],
    page_url: str,
    page_title: str,
    page_text: str,
) -> str:
    facets_block = ", ".join(f for f in (facets or []) if f) or "(no specific facets)"
    return (
        "You are extracting atomic, self-contained, FACTUAL claims relevant to the user's research task. "
        "Output STRICT JSON only (no markdown).\n\n"
        f"User task: {task}\n"
        f"Facets to cover: {facets_block}\n"
        f"Source URL: {page_url}\n"
        f"Source title: {page_title}\n\n"
        "Guidelines:\n"
        "- Each claim must be a single declarative sentence that could be cited on its own.\n"
        "- Prefer concrete numbers, names, dates, events, comparisons over vague generalities.\n"
        "- Skip opinions, marketing fluff, navigation text.\n"
        "- Provide a short verbatim supporting quote for each claim (<= 240 chars).\n"
        f"- Limit to at most {config.MAX_CLAIMS_PER_PAGE} highest-value claims.\n"
        "- If the page is irrelevant, return an empty list.\n\n"
        "Schema:\n"
        '{ "claims": [ {"text": "...", "quote": "..."} ] }\n\n'
        f"--- PAGE TEXT (truncated) ---\n{page_text[: config.SNIPPET_CHARS * 2]}\n--- END ---"
    )


def extract_claims_from_page(
    task: str,
    facets: Sequence[str],
    page_url: str,
    page_title: str,
    page_text: str,
) -> List[ExtractedClaim]:
    if not page_text or len(page_text) < 200:
        return []
    prompt = _build_extraction_prompt(task, facets, page_url, page_title, page_text)
    parsed = llm_chat_json(prompt, temperature=0.2)
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("claims") or []
    if not isinstance(raw, list):
        return []
    out: List[ExtractedClaim] = []
    for c in raw[: config.MAX_CLAIMS_PER_PAGE]:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        quote = (c.get("quote") or "").strip()
        out.append(ExtractedClaim(text=text, quote=quote))
    return out


def ingest_page_into_claims(
    memory: TopicMemory,
    task: str,
    facets: Sequence[str],
    page_url: str,
    page_title: str,
    page_text: str,
) -> Dict[str, int]:
    """Extract claims from a page and merge them into memory.

    Returns counts: {extracted, added, merged, contradicts}.
    """
    extracted = extract_claims_from_page(task, facets, page_url, page_title, page_text)
    counts = {"extracted": len(extracted), "added": 0, "merged": 0, "contradicts": 0}
    if not extracted:
        return counts
    vecs = embed([c.text for c in extracted]) or [None] * len(extracted)
    for c, v in zip(extracted, vecs):
        rec, action = memory.add_claim(
            text=c.text,
            source_url=page_url,
            embedding=v,
            quote=c.quote,
        )
        if action == "added":
            counts["added"] += 1
        elif action == "merged":
            counts["merged"] += 1
        elif action == "contradicts":
            counts["contradicts"] += 1
    return counts


def facet_coverage(
    memory: TopicMemory,
    facets: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    """For each facet, find supporting claims by embedding similarity."""
    facets = [f for f in (facets or []) if f]
    if not facets:
        return {}
    claims = memory.all_claims()
    if not claims:
        return {f: {"supporting_claims": [], "domains": []} for f in facets}

    facet_vecs = embed(facets)
    claim_texts = [c.get("text", "") for c in claims]
    claim_vecs = embed(claim_texts)
    if not claim_vecs:
        return {f: {"supporting_claims": [], "domains": []} for f in facets}
    mat = np.vstack(claim_vecs)
    out: Dict[str, Dict[str, object]] = {}
    for facet, fv in zip(facets, facet_vecs):
        sims = cosine_matrix(fv, mat)
        if sims.size == 0:
            out[facet] = {"supporting_claims": [], "domains": []}
            continue
        idxs = np.where(sims >= 0.45)[0]
        supports = [claims[int(i)] for i in idxs]
        domains = set()
        for c in supports:
            for u in c.get("sources") or []:
                h = host_of(u)
                if h:
                    domains.add(h)
        out[facet] = {
            "supporting_claims": [shorten(c.get("text", ""), 160) for c in supports[:10]],
            "domains": sorted(domains),
            "support_count": len(supports),
        }
    return out
