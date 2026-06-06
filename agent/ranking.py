"""Candidate ranker for the discovery frontier.

Each candidate URL gets a composite score:

    score = w_sim * cosine(task_embedding, slug+title+snippet)
          + w_trust * domain_trust(host)
          + w_novelty * novelty(host_seen, length, slug)

Optionally, the top-K candidates are batch-reranked by the LLM for high
precision tie-breaking. The frontier itself lives in `TopicMemory`; this
module only computes scores and applies the per-host cap.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .embeddings import cosine, embed, embed_one
from .llm import is_placeholder, llm_chat_json
from .memory import TopicMemory
from .utils import canonicalize_url, host_of


@dataclass
class CandidateInput:
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.url = canonicalize_url(self.url)


@dataclass
class ScoredCandidate:
    url: str
    score: float
    title: str = ""
    snippet: str = ""
    source: str = ""
    breakdown: Dict[str, float] = field(default_factory=dict)


_SLUG_CHUNK_RE = re.compile(r"[a-zA-Z0-9]+")


def _slug_text(url: str) -> str:
    parts = _SLUG_CHUNK_RE.findall(url or "")
    return " ".join(parts[-12:])


def _topic_text(task: str, facets: Sequence[str], missing: Sequence[str]) -> str:
    parts = [task or ""]
    if facets:
        parts.append("Facets: " + "; ".join(f for f in facets if f))
    if missing:
        parts.append("Missing: " + "; ".join(m for m in missing if m))
    return " \n".join(parts)


def _novelty(
    host: str,
    title: str,
    host_counts: Dict[str, int],
    visited_hosts: Counter,
) -> float:
    saturation = host_counts.get(host, 0) + visited_hosts.get(host, 0)
    novelty = math.exp(-0.35 * saturation)
    if title:
        novelty = min(1.0, novelty + 0.05)
    return float(novelty)


def score_candidates(
    inputs: Sequence[CandidateInput],
    task: str,
    facets: Sequence[str],
    missing: Sequence[str],
    memory: TopicMemory,
) -> List[ScoredCandidate]:
    """Score and dedupe candidates without mutating memory's frontier."""
    if not inputs:
        return []

    deduped: Dict[str, CandidateInput] = {}
    for c in inputs:
        if not c.url:
            continue
        existing = deduped.get(c.url)
        if existing is None:
            deduped[c.url] = c
        else:
            if not existing.title and c.title:
                existing.title = c.title
            if not existing.snippet and c.snippet:
                existing.snippet = c.snippet
    cands = [c for c in deduped.values() if not memory.is_visited(c.url)]
    if not cands:
        return []

    topic_vec = embed_one(_topic_text(task, facets, missing))
    descriptions = [
        " ".join([c.title, c.snippet, _slug_text(c.url)]).strip() or c.url
        for c in cands
    ]
    cand_vecs = embed(descriptions)
    if not cand_vecs:
        cand_vecs = [np.zeros_like(topic_vec) for _ in cands]

    visited_hosts = Counter(host_of(u) for u in memory.successful_urls())
    visited_hosts.update(host_of(u) for u in memory.failed_urls())

    seen_hosts: Dict[str, int] = {}
    scored: List[ScoredCandidate] = []
    for c, vec in zip(cands, cand_vecs):
        host = host_of(c.url)
        sim = cosine(topic_vec, vec)
        trust = memory.domain_trust_score(host)
        novelty = _novelty(host, c.title, seen_hosts, visited_hosts)
        seen_hosts[host] = seen_hosts.get(host, 0) + 1

        sim01 = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        score = (
            config.RANK_WEIGHT_SIM * sim01
            + config.RANK_WEIGHT_TRUST * trust
            + config.RANK_WEIGHT_NOVELTY * novelty
        )
        scored.append(
            ScoredCandidate(
                url=c.url,
                score=score,
                title=c.title,
                snippet=c.snippet,
                source=c.source,
                breakdown={
                    "sim": sim01,
                    "trust": trust,
                    "novelty": novelty,
                },
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def llm_rerank_topk(
    scored: List[ScoredCandidate],
    task: str,
    facets: Sequence[str],
    missing: Sequence[str],
    top_k: int = config.LLM_RERANK_TOP_K,
) -> List[ScoredCandidate]:
    """Ask the LLM to pick the most useful URLs from the heuristic top-K."""
    if not scored or top_k <= 0:
        return scored
    head = scored[:top_k]
    if len(head) <= 2:
        return scored

    listing = "\n".join(
        f"{i+1}. {s.url}\n   title: {s.title}\n   snippet: {s.snippet[:200]}"
        for i, s in enumerate(head)
    )
    prompt = (
        "You are scoring candidate web URLs for usefulness on the research task below. "
        "Return STRICT JSON only.\n\n"
        f"Task: {task}\n"
        f"Facets to cover: {', '.join(facets) or '(none)'}\n"
        f"Currently missing: {', '.join(missing) or '(nothing explicit)'}\n\n"
        "For each URL, give a 0-100 usefulness score and a 1-line reason.\n"
        "Schema:\n"
        '{ "rankings": [ {"index": 1, "score": 0-100, "reason": "..."}, ... ] }\n\n'
        f"Candidates:\n{listing}"
    )
    parsed = llm_chat_json(prompt, temperature=0.2)
    if not isinstance(parsed, dict):
        return scored
    rankings = parsed.get("rankings") or []
    if not isinstance(rankings, list):
        return scored
    boosts: Dict[int, float] = {}
    for r in rankings:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("index", 0)) - 1
            sc = float(r.get("score", 0))
        except Exception:
            continue
        if 0 <= idx < len(head):
            boosts[idx] = max(0.0, min(100.0, sc)) / 100.0
    if not boosts:
        return scored
    for idx, b in boosts.items():
        head[idx].score = 0.6 * head[idx].score + 0.4 * b
        head[idx].breakdown["llm_rerank"] = b
    new_head_sorted = sorted(head, key=lambda s: s.score, reverse=True)
    tail = scored[top_k:]
    return new_head_sorted + tail


def push_to_frontier(
    scored: List[ScoredCandidate],
    memory: TopicMemory,
    top_k: int = config.FRONTIER_PUSH_TOP_K,
    per_host_cap: int = config.FRONTIER_PER_HOST_CAP,
) -> List[ScoredCandidate]:
    """Insert top-K (with per-host cap) into the persistent frontier."""
    pushed: List[ScoredCandidate] = []
    host_counts: Dict[str, int] = {}
    for s in scored:
        if len(pushed) >= top_k:
            break
        host = host_of(s.url)
        if host_counts.get(host, 0) >= per_host_cap:
            continue
        cand = memory.add_candidate(
            url=s.url,
            score=s.score,
            source=s.source,
            title=s.title,
            snippet=s.snippet,
        )
        if cand:
            host_counts[host] = host_counts.get(host, 0) + 1
            pushed.append(s)
    return pushed
