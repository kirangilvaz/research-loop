"""Plan node: interpret the user task, decompose into facets, and seed queries.

Produces (on `state`):
    interpretation       - what the user is really asking
    facets[]             - concrete sub-questions to cover
    initial_queries[]    - varied DDG queries to seed discovery
    presentation_hint    - early guess of layout / output schema
    plan (string)        - human-readable short plan for the report
"""

from __future__ import annotations

from typing import Any, Dict, List

from .llm import is_placeholder, llm_chat, llm_chat_json
from .state import State


_DEFAULT_FACETS = [
    "Background and context",
    "Key entities and definitions",
    "Recent developments",
    "Risks and counterpoints",
    "Practical implications",
]


def _parse_planning_response(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("interpretation", "headline"):
        v = parsed.get(key)
        if isinstance(v, str):
            out[key] = v.strip()
    facets = parsed.get("facets")
    if isinstance(facets, list):
        out["facets"] = [str(x).strip() for x in facets if str(x).strip()]
    queries = parsed.get("initial_queries")
    if isinstance(queries, list):
        out["initial_queries"] = [str(x).strip() for x in queries if str(x).strip()]
    pres = parsed.get("presentation")
    if isinstance(pres, dict):
        out["presentation"] = pres
    plan_steps = parsed.get("plan_steps")
    if isinstance(plan_steps, list):
        out["plan_steps"] = [str(x).strip() for x in plan_steps if str(x).strip()]
    return out


def _fallback_plan(task: str) -> Dict[str, Any]:
    return {
        "interpretation": task,
        "facets": list(_DEFAULT_FACETS),
        "initial_queries": [
            task,
            f"{task} overview",
            f"{task} explained",
            f"{task} latest news",
            f"{task} pros and cons",
            f"{task} expert analysis",
        ],
        "presentation": {
            "layout": "narrative",
            "title": task[:80] or "Research Results",
            "headline": "",
        },
        "plan_steps": [
            "Decompose the task into specific sub-questions.",
            "Search across diverse sources, harvest links, and crawl deeper as evidence accumulates.",
            "Synthesize findings, weigh contradictions, and present an honest answer with citations.",
        ],
    }


def plan(state: State) -> None:
    print("Planning...")
    prompt = (
        "Plan a research session for the user's task. Output STRICT JSON only.\n\n"
        f"Task: {state.task}\n\n"
        "Schema:\n"
        "{\n"
        '  "interpretation": "one paragraph: what the user is really asking and any ambiguity",\n'
        '  "facets": ["6-10 concrete sub-questions whose answers, together, would satisfy the task"],\n'
        '  "initial_queries": ["6-10 varied DuckDuckGo-style search queries that cover those facets"],\n'
        '  "presentation": {\n'
        '    "layout": "cards|table|ranked_list|comparison|narrative|timeline",\n'
        '    "title": "tailored page title for the final report",\n'
        '    "headline": "one-line take (<=140 chars)",\n'
        '    "columns": ["only for table/comparison"],\n'
        '    "sections": ["optional per-item subsections"]\n'
        "  },\n"
        '  "plan_steps": ["3-5 short numbered steps describing the research approach"]\n'
        "}\n\n"
        "Guidelines:\n"
        "- Pick the layout that genuinely fits the task. Use 'narrative' for explanatory topics, "
        "'cards' for sets of items, 'table' for comparable rows, 'ranked_list' for ordering, "
        "'comparison' for 2-3 things side by side, 'timeline' for chronologies.\n"
        "- Queries must be specific enough to retrieve high-signal pages on the open web. "
        "Vary phrasing, time windows ('2024', 'latest'), source qualifiers ('site:gov', 'pdf'), "
        "and angles (positive, contrarian, technical, plain-language).\n"
    )
    parsed = llm_chat_json(prompt, temperature=0.3)
    data = _parse_planning_response(parsed) if parsed else {}
    if not data:
        print("  Plan LLM call failed or empty; using deterministic fallback.")
        data = _fallback_plan(state.task)

    state.interpretation = data.get("interpretation", "") or state.task
    state.facets = data.get("facets") or list(_DEFAULT_FACETS)
    state.initial_queries = data.get("initial_queries") or [state.task]
    state.presentation_hint = data.get("presentation") or {}

    plan_steps: List[str] = data.get("plan_steps") or []
    if plan_steps:
        state.plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan_steps))
    else:
        state.plan = (
            "1. Decompose the task into concrete facets.\n"
            "2. Search broadly, follow links, and crawl until conviction is high.\n"
            "3. Synthesize findings with citations and present them in the right layout."
        )

    print(f"  Interpretation: {state.interpretation[:200]}")
    print(f"  Facets ({len(state.facets)}):")
    for f in state.facets:
        print(f"    - {f}")
    print(f"  Initial queries ({len(state.initial_queries)}):")
    for q in state.initial_queries[:8]:
        print(f"    > {q}")
    print()
