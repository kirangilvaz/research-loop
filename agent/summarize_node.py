"""Final summarize node: turn the persistent corpus into a structured answer.

This is the inference step. It consumes the claims, page snippets, and
conviction signals stored in `TopicMemory` and produces:
    - status (ready / need_more_info)
    - missing items
    - presentation hint (layout/title/headline/columns/sections)
    - generic items (no ticker/conviction forced)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import config
from .conviction import ConvictionSignals
from .llm import llm_chat_json
from .memory import TopicMemory
from .state import State
from .utils import host_of, shorten


# Report types drive both how much detail the summarizer captures and which
# HTML template renders the result (see agent/html_node.REPORT_TYPES). The
# "general" types map back to the original six flat layouts.
VALID_REPORT_TYPES = {
    "qa_list",
    "financial",
    "analysis",
    "comparison",
    "entity_list",
    "ranked",
    "timeline",
    "narrative",
    "general",
}

_REPORT_TYPE_MENU = (
    "- qa_list: a set of questions/answers (e.g. interview questions). Group by topic; "
    "include difficulty, frequency, and company tags when available.\n"
    "- financial: a company / financial assessment. KPIs, fundamentals, segments, risks, outlook.\n"
    "- analysis: a deep dive on a concept, method, or phenomenon. Narrative sections + key findings + caveats.\n"
    "- comparison: a head-to-head of 2+ options. Per-option pros/cons + an attribute matrix + a verdict.\n"
    "- entity_list: a set of distinct entities/items (companies, tools, papers, events).\n"
    "- ranked: an ordered ranking where position matters.\n"
    "- timeline: chronologically ordered events.\n"
    "- narrative: a single long-form explanatory answer.\n"
    "- general: anything else; the layout hint decides the rendering."
)


def _claim_block(claims: Sequence[dict], limit: int = config.SUMMARY_MAX_CLAIMS) -> str:
    sorted_claims = sorted(
        claims,
        key=lambda c: (-int(c.get("support", 0)), c.get("created_at", "")),
    )
    lines = []
    for c in sorted_claims[:limit]:
        sources = c.get("sources") or []
        domain_str = ", ".join({host_of(u) for u in sources if host_of(u)}) or "n/a"
        lines.append(
            f"- {c.get('text', '').strip()} "
            f"[support={c.get('support', 0)}; domains={domain_str}]"
        )
    return "\n".join(lines) or "(no claims accumulated)"


def _page_excerpt_block(
    memory: TopicMemory,
    limit: int = config.SUMMARY_MAX_PAGE_EXCERPTS,
    chars: int = config.SUMMARY_EXCERPT_CHARS,
) -> str:
    pages = sorted(memory.all_pages(), key=lambda p: -int(p.get("text_len", 0)))
    chunks = []
    for p in pages[:limit]:
        snippet = (p.get("snippet") or "").strip()[:chars]
        chunks.append(
            f"--- {p.get('url')} (title: {p.get('title') or '(no title)'}) ---\n{snippet}"
        )
    return "\n\n".join(chunks) or "(no pages collected)"


def _ok_sources(memory: TopicMemory) -> List[str]:
    return memory.successful_urls()


def _build_prompt(
    state: State,
    memory: TopicMemory,
    signals: ConvictionSignals,
    force_ready: bool,
) -> str:
    ok_urls = _ok_sources(memory)
    sources_block = "\n".join(f"- {u}" for u in ok_urls) or "(none)"
    facets_block = "\n".join(f"- {f}" for f in state.facets) or "(none)"
    coverage_lines = []
    for f, info in (signals.facet_coverage or {}).items():
        domains = info.get("domains") or []
        support = info.get("support_count", len(info.get("supporting_claims") or []))
        coverage_lines.append(
            f"- {f}: {support} claims, {len(domains) if isinstance(domains, list) else 0} domains"
        )
    presentation_hint = state.presentation_hint or {}
    layout_hint = presentation_hint.get("layout") or "narrative"

    force_clause = ""
    if force_ready:
        force_clause = (
            "\nThis is the FINAL pass. You MUST set status to \"ready\" and "
            "return your best-effort items even if data is thin.\n"
        )

    head = (
        f"User's task: {state.task}\n"
        f"Topic interpretation: {state.interpretation}\n"
        f"Facets:\n{facets_block}\n\n"
        f"Confidence threshold for 'ready': {config.CONVICTION_TARGET}\n"
        f"Conviction signals so far:\n"
        f"  overall={signals.overall:.2f} coverage={signals.coverage:.2f} "
        f"diversity={signals.diversity:.2f} plateau={signals.plateau:.2f} "
        f"contradiction={signals.contradiction:.2f} llm_conf={signals.llm_confidence:.2f}\n"
        f"  domains={signals.distinct_domains} new_recent={signals.new_claims_recent}\n"
        f"Facet coverage:\n" + "\n".join(coverage_lines or ["(none)"]) + "\n"
        f"{force_clause}\n"
        "You have the corpus below: atomic claims with support counts and source domains, "
        "plus excerpts from collected pages. Your job is to produce a DETAILED, well-structured "
        "report that lets the reader consume everything efficiently.\n\n"
        "STEP 1 - Classify the report. Pick the single best `report_type` for this task:\n"
        f"{_REPORT_TYPE_MENU}\n\n"
        "STEP 2 - Decide whether the answer is naturally CATEGORIZED (e.g. interview questions by "
        "topic, findings by theme, companies by sector). If so, return `groups` (each with a "
        "`name`, optional `summary`, and its own `items`) so the report can show category tabs. "
        "If the answer is a single flat list or a single narrative, leave `groups` empty and use "
        "`items` directly.\n\n"
        "STEP 3 - Fill items with as much grounded detail as the corpus supports. Items must cite "
        "URLs from the OK sources list (or, if no relevant URL exists, omit `sources`).\n\n"
        "Return STRICT JSON only. Schema:\n"
        "{\n"
        '  "status": "ready" | "need_more_info",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "report_type": "one of the report types listed above",\n'
        '  "missing": ["concrete info still needed; each a short search-friendly phrase"],\n'
        '  "presentation": {\n'
        '    "layout": "cards" | "table" | "ranked_list" | "comparison" | "narrative" | "timeline",\n'
        '    "title": "tailored page title",\n'
        '    "headline": "one-line overall take (<=160 chars)",\n'
        '    "columns": ["only for table/comparison"],\n'
        '    "sections": ["per-item subsections, e.g. Background, Key Points, Risks"]\n'
        "  },\n"
        '  "groups": [\n'
        "    {\n"
        '      "name": "Category / tab name",\n'
        '      "summary": "1-line description of this category (optional)",\n'
        '      "items": [ "...same item shape as below..." ]\n'
        "    }\n"
        "  ],\n"
        '  "items": [\n'
        "    {\n"
        '      "name": "Display name of the item, person, event, question, or section",\n'
        '      "headline": "1-line summary of this item",\n'
        '      "question": "for qa_list: the actual question text",\n'
        '      "answer": "for qa_list: a thorough answer / approach (can be multi-paragraph)",\n'
        '      "company_tags": ["for qa_list: companies known to ask this"],\n'
        '      "frequency": "for qa_list: how often asked, e.g. Very common / Occasional",\n'
        '      "difficulty": "for qa_list: Easy | Medium | Hard",\n'
        '      "key_points": ["3-7 concise bullets"],\n'
        '      "metrics": {"for financial/analysis: numeric facts, e.g. Revenue: $60.9B, P/E: 65"},\n'
        '      "pros": ["for comparison: strengths of this option"],\n'
        '      "cons": ["for comparison: weaknesses of this option"],\n'
        '      "verdict": "for comparison: when to prefer this option",\n'
        '      "details": {"free-form key/value facts, e.g. Date: 2024-05, Region: APAC"},\n'
        '      "evidence": ["for analysis: supporting findings / data points"],\n'
        '      "body": "for narrative/analysis/timeline: 1-3 paragraphs of prose. Optional otherwise.",\n'
        '      "when": "for timeline only: ISO-ish date or year",\n'
        '      "sources": ["urls from the OK sources list that support this item"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        f"- Layout hint from the planning step: {layout_hint}. Override if the corpus suggests another fits better.\n"
        "- Populate ONLY the item fields that fit the chosen report_type; omit the rest. Renderers ignore unknown/empty fields.\n"
        "- Be generous with detail: prefer many well-supported items over a few thin ones. Pull concrete numbers, names, dates, and quotes from the corpus.\n"
        "- For qa_list: group by topic, set difficulty/frequency/company_tags when the corpus supports them, and write a genuinely useful answer.\n"
        "- For financial: surface KPIs in `metrics`, and organize items as report sections (Valuation, Fundamentals, Segments, Risks, Outlook).\n"
        "- For comparison: one item per option, fill pros/cons/verdict, and set presentation.columns to the compared attributes.\n"
        "- Be honest: if data is thin or contradictory, set status to 'need_more_info' and put concrete missing phrases into `missing`.\n"
        "- Prefer claims with high `support` and multiple domains; avoid single-source assertions for important items.\n"
        "- Do NOT invent URLs.\n\n"
        f"OK sources:\n{sources_block}\n\n"
    )

    claims_block = _claim_block(memory.all_claims())
    excerpt_block = _page_excerpt_block(memory)
    corpus = (
        f"--- ACCUMULATED CLAIMS ---\n{claims_block}\n--- END CLAIMS ---\n\n"
        f"--- TOP PAGE EXCERPTS ---\n{excerpt_block}\n--- END EXCERPTS ---"
    )

    # Hard safety cap so a huge corpus can't blow past the model context window.
    budget = max(2000, config.SUMMARY_PROMPT_MAX_CHARS - len(head))
    if len(corpus) > budget:
        corpus = corpus[:budget] + "\n--- (corpus truncated to fit prompt budget) ---"
    return head + corpus


def summarize(
    state: State,
    signals: ConvictionSignals,
    force_ready: bool = False,
) -> None:
    print(
        f"Summarizing (iteration {state.iteration}"
        f"{', FORCED FINAL' if force_ready else ''})..."
    )

    if state.memory is None:
        print("  No memory configured; cannot summarize.")
        return

    prompt = _build_prompt(state, state.memory, signals, force_ready)
    parsed = llm_chat_json(prompt, temperature=0.25)
    state.summary_raw = "" if parsed is None else str(parsed)[:8000]

    if not isinstance(parsed, dict):
        print("  Could not parse JSON from model. Treating as insufficient.")
        state.sufficient = force_ready
        state.confidence = signals.overall
        return

    status = str(parsed.get("status", "")).strip().lower()
    try:
        confidence = float(parsed.get("confidence", 0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    missing = parsed.get("missing") or []
    if isinstance(missing, list):
        state.missing = [str(m) for m in missing if m]

    presentation = parsed.get("presentation") or {}
    if isinstance(presentation, dict):
        state.presentation = presentation
        state.headline = str(presentation.get("headline", "")).strip()

    report_type = str(parsed.get("report_type", "")).strip().lower()
    state.report_type = report_type if report_type in VALID_REPORT_TYPES else ""

    groups = parsed.get("groups") or []
    clean_groups: List[Dict[str, Any]] = []
    flattened: List[Dict[str, Any]] = []
    if isinstance(groups, list):
        for g in groups:
            if not isinstance(g, dict):
                continue
            g_items = [it for it in (g.get("items") or []) if isinstance(it, dict)]
            if not g_items:
                continue
            clean_groups.append(
                {
                    "name": str(g.get("name", "")).strip() or "Group",
                    "summary": str(g.get("summary", "")).strip(),
                    "items": g_items,
                }
            )
            flattened.extend(g_items)
    state.groups = clean_groups

    items = parsed.get("items") or []
    parsed_items = [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []
    # Keep `state.items` populated even when the model only returned groups, so
    # the side panels, run stats, and snapshots keep working as before.
    state.items = parsed_items or flattened

    state.confidence = confidence
    state.sufficient = force_ready or (
        status == "ready"
        and confidence >= config.CONVICTION_TARGET
        and bool(state.items)
    )

    print(
        f"  status={status!r} confidence={state.confidence:.2f} "
        f"report_type={state.report_type or '(none)'} "
        f"items={len(state.items)} groups={len(state.groups)} "
        f"sufficient={state.sufficient}"
    )
    if not state.sufficient and state.missing:
        print("  Missing info flagged for next iteration:")
        for m in state.missing[:8]:
            print(f"    - {m}")
    print()
