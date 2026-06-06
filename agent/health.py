"""End-of-iteration health summary + actionable alerts.

The goal is plain-English visibility into "is the run going well?" without
having to read every line of the verbose per-node logs. After each iteration
the orchestrator calls `print_health(state)` which prints one compact block:

    [Health iter 5/500] conviction=0.27 (+0.07) coverage=2/10 frontier=86 visited=30 claims=29
      sources(last 3): link=68% llm=24% search=4% replay=4%
      top hosts: cisa.gov=12(40%) ti.com=8(27%) semianalysis.com=4(13%)
      productivity: 6/8 pages produced claims this iter (75%)
      [ok] no issues detected

When something is wrong, one or more alert lines are added:

      [WARN] SEARCH STALLED: 0 search-source candidates in last 3 iterations -- DDG/SearXNG likely blocked or empty
      [WARN] RABBIT HOLE: top 2 hosts (cisa.gov=12, ti.com=8) hold 67% of visited URLs -- frontier likely off-topic
      [WARN] LOW PRODUCTIVITY: only 3/8 (37%) recent pages produced claims -- seeds may be off-topic
      [WARN] COVERAGE SKEW: 7/10 facets have 0 claims while top facet has 14
      [WARN] CONVICTION FLAT: +0.01 over last 4 iterations -- expect creativity_boost soon

The alerts intentionally tell the user *what to do* (lower workers, switch
search backend, reset slug, etc.) rather than just naming the symptom.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .claims import facet_coverage
from .state import State
from .utils import host_of


_ALERT_PREFIX = "[WARN]"
_OK_PREFIX = "[ok]  "


# ---------------------------------------------------------------------------
# Source-mix bucketing
# ---------------------------------------------------------------------------

def bucket_source(source: str) -> str:
    """Map a raw candidate `source` string to a coarse bucket name.

    `source` strings look like `search:semiconductor etf...`, `link-harvest`,
    `link-harvest:backflow`, `llm-suggest`, `memory-replay`. We bucket them
    down to 4 categories that are useful for diagnosing where seeds came from.
    """
    if not source:
        return "?"
    head = source.split(":", 1)[0]
    if head == "search":
        return "search"
    if head == "link-harvest":
        return "link"
    if head == "llm-suggest":
        return "llm"
    if head == "memory-replay":
        return "replay"
    return head[:8] or "?"


# ---------------------------------------------------------------------------
# Alert detectors
# ---------------------------------------------------------------------------

def _alert_search_stalled(state: State) -> Optional[str]:
    """Fires when the search backends have returned 0 hits for >= 3 iterations."""
    history = state.source_mix_history[-3:]
    if len(history) < 3:
        return None
    search_hits = sum(m.get("search", 0) for m in history)
    if search_hits > 0:
        return None
    return (
        f"{_ALERT_PREFIX} SEARCH STALLED: 0 'search'-source candidates in last "
        f"{len(history)} iterations -- DDG/SearXNG likely blocked or returning empty. "
        f"Check the per-query [ddg/html] log lines above; consider self-hosting "
        f"SearXNG (RESEARCH_SEARXNG_URL) or setting RESEARCH_BRAVE_API_KEY."
    )


def _alert_rabbit_hole(state: State, min_visited: int = 15, threshold: float = 0.6) -> Optional[str]:
    """Fires when the top 2 hosts account for >threshold of successful fetches."""
    if state.memory is None:
        return None
    visited = state.memory.successful_urls()
    if len(visited) < min_visited:
        return None
    host_counts: Dict[str, int] = defaultdict(int)
    for u in visited:
        h = host_of(u)
        if h:
            host_counts[h] += 1
    if not host_counts:
        return None
    total = sum(host_counts.values())
    top = sorted(host_counts.items(), key=lambda kv: -kv[1])[:2]
    top2 = sum(c for _, c in top)
    share = top2 / total
    if share < threshold:
        return None
    names = ", ".join(f"{h}={c}" for h, c in top)
    return (
        f"{_ALERT_PREFIX} RABBIT HOLE: top 2 hosts ({names}) hold "
        f"{share:.0%} of {total} visited URLs -- frontier likely off-topic. "
        f"Try a creativity_boost (it should fire automatically soon), or "
        f"restart with a different --topic-slug if the seeds are wrong."
    )


def _alert_low_productivity(state: State, window: int = 2, threshold: float = 0.4) -> Optional[str]:
    """Fires when recent fetches mostly failed to extract claims."""
    history = state.productivity_history[-window:]
    if len(history) < window:
        return None
    fetched = sum(p.get("fetched", 0) for p in history)
    productive = sum(p.get("productive", 0) for p in history)
    if fetched < 6:  # too small a sample to be meaningful
        return None
    ratio = productive / fetched
    if ratio >= threshold:
        return None
    return (
        f"{_ALERT_PREFIX} LOW PRODUCTIVITY: only {productive}/{fetched} ({ratio:.0%}) "
        f"recent pages produced claims -- frontier may be drifting to off-topic seeds. "
        f"The backflow filter (synthesize_node) will limit further damage; "
        f"a creativity_boost should fire if this persists."
    )


def _alert_coverage_skew(state: State, min_total_claims: int = 12, min_empty_share: float = 0.5) -> Optional[str]:
    """Fires when many facets are empty while others have lots of claims."""
    if state.memory is None or not state.facets:
        return None
    coverage = facet_coverage(state.memory, state.facets)
    if not coverage:
        return None
    counts: List[int] = []
    for info in coverage.values():
        if isinstance(info, dict):
            counts.append(int(info.get("support_count", len(info.get("supporting_claims") or []))))
    if not counts:
        return None
    if sum(counts) < min_total_claims:
        return None
    empty = sum(1 for c in counts if c == 0)
    if empty / len(counts) < min_empty_share:
        return None
    return (
        f"{_ALERT_PREFIX} COVERAGE SKEW: {empty}/{len(counts)} facets have 0 claims while "
        f"top facet has {max(counts)}. The agent is over-indexed on a few angles; "
        f"check the 'Facet coverage' block above to see which facets are starving."
    )


def _alert_conviction_flat(state: State, window: int = 4, min_delta: float = 0.02) -> Optional[str]:
    """Fires when conviction hasn't meaningfully moved in `window` iterations."""
    hist = state.conviction_history[-window:]
    if len(hist) < window:
        return None
    delta = hist[-1] - hist[0]
    if delta >= min_delta:
        return None
    return (
        f"{_ALERT_PREFIX} CONVICTION FLAT: only {delta:+.2f} change in last {len(hist)} "
        f"iterations (curve: {' -> '.join(f'{x:.2f}' for x in hist)}). "
        f"Plateau-detection will trigger a creativity_boost shortly if this persists."
    )


_DETECTORS = (
    _alert_search_stalled,
    _alert_rabbit_hole,
    _alert_low_productivity,
    _alert_coverage_skew,
    _alert_conviction_flat,
)


def detect_alerts(state: State) -> List[str]:
    out: List[str] = []
    for fn in _DETECTORS:
        try:
            msg = fn(state)
        except Exception:  # noqa: BLE001 - never crash the run from health checks
            msg = None
        if msg:
            out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _format_conviction(state: State) -> str:
    if not state.conviction_history:
        return "conviction=n/a"
    last = state.conviction_history[-1]
    if len(state.conviction_history) >= 2:
        delta = last - state.conviction_history[-2]
        sign = "+" if delta >= 0 else ""
        return f"conviction={last:.2f} ({sign}{delta:.2f})"
    return f"conviction={last:.2f}"


def _format_coverage(state: State) -> Tuple[str, int, int]:
    if state.memory is None or not state.facets:
        return "coverage=n/a", 0, 0
    coverage = facet_coverage(state.memory, state.facets)
    ok = 0
    total = len(state.facets)
    for info in coverage.values():
        if isinstance(info, dict):
            domains = info.get("domains") or []
            if isinstance(domains, list) and len(domains) >= 2:
                ok += 1
    return f"coverage={ok}/{total}", ok, total


def _format_source_mix(state: State, window: int = 3) -> Optional[str]:
    history = state.source_mix_history[-window:]
    if not history:
        return None
    totals: Dict[str, int] = defaultdict(int)
    for m in history:
        for k, v in m.items():
            totals[k] += v
    total = sum(totals.values())
    if total == 0:
        return f"sources(last {len(history)}): (none)"
    pairs = sorted(totals.items(), key=lambda kv: -kv[1])
    body = " ".join(f"{k}={v/total:.0%}" for k, v in pairs)
    return f"sources(last {len(history)}): {body}"


def _format_top_hosts(state: State, n: int = 3) -> Optional[str]:
    if state.memory is None or state.memory.visited_count() == 0:
        return None
    counts: Dict[str, int] = defaultdict(int)
    for u in state.memory.successful_urls():
        h = host_of(u)
        if h:
            counts[h] += 1
    if not counts:
        return None
    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
    body = " ".join(f"{h}={c}({c/total:.0%})" for h, c in top)
    return f"top hosts: {body}"


def _format_productivity(state: State) -> Optional[str]:
    if not state.productivity_history:
        return None
    last = state.productivity_history[-1]
    fetched = int(last.get("fetched", 0))
    productive = int(last.get("productive", 0))
    if fetched == 0:
        return "productivity: no pages fetched this iter"
    return (
        f"productivity: {productive}/{fetched} pages produced claims this iter "
        f"({productive/fetched:.0%})"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def print_health(state: State) -> None:
    """Print a one-block health summary for the current iteration.

    Safe to call even on iteration 1 (fields gracefully degrade to 'n/a' when
    the history sliding windows aren't full yet).
    """
    frontier = state.memory.frontier_size() if state.memory is not None else 0
    visited = state.memory.visited_count() if state.memory is not None else 0
    claims = state.memory.claim_count() if state.memory is not None else 0

    conv_str = _format_conviction(state)
    cov_str, _ok, _total = _format_coverage(state)

    print(
        f"[Health iter {state.iteration}] "
        f"{conv_str} {cov_str} "
        f"frontier={frontier} visited={visited} claims={claims}"
    )

    for line in filter(None, (
        _format_source_mix(state),
        _format_top_hosts(state),
        _format_productivity(state),
    )):
        print(f"  {line}")

    alerts = detect_alerts(state)
    if alerts:
        for a in alerts:
            print(f"  {a}")
    else:
        print(f"  {_OK_PREFIX} no issues detected")
    print()


__all__ = ["bucket_source", "detect_alerts", "print_health"]
