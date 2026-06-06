"""HTML report renderer.

Six layouts are supported:
    cards        - rich per-item overview, responsive card grid (default)
    table        - comparable columns, good for metrics
    ranked_list  - ordered list, when the answer is a ranking
    comparison   - pivoted side-by-side table for 2-3 items with many attributes
    narrative    - long-form sections of prose with a few highlighted callouts
    timeline     - chronologically ordered events
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .state import State
from .utils import esc, host_of, safe_write_text


DEFAULT_SECTIONS = ["Key Points", "Background", "Risks"]
DEFAULT_TABLE_COLUMNS = ["Name", "Headline", "Sources"]


def _item_get(item: Dict[str, Any], key: str) -> Any:
    if not isinstance(item, dict):
        return ""
    if key in item:
        return item[key]
    kl = key.lower().strip()
    for k, v in item.items():
        if str(k).lower().strip() == kl:
            return v
    details = item.get("details") or {}
    if isinstance(details, dict):
        for k, v in details.items():
            if str(k).lower().strip() == kl:
                return v
    synonyms = {
        "title": "name",
        "summary": "headline",
        "rationale": "headline",
        "points": "key_points",
        "bullets": "key_points",
        "body": "body",
        "description": "body",
    }
    alt = synonyms.get(kl)
    if alt and alt in item:
        return item[alt]
    return ""


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(esc(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{esc(k)}: {esc(v)}" for k, v in value.items())
    return esc(value)


def _render_sources_chips(item: Dict[str, Any]) -> str:
    srcs = item.get("sources") or []
    if not isinstance(srcs, list) or not srcs:
        return '<span class="muted">no sources cited</span>'
    parts = []
    for u in srcs:
        host = host_of(str(u)) or re.sub(r"^https?://(www\.)?", "", str(u)).split("/")[0]
        parts.append(
            f'<a class="chip" href="{esc(u)}" target="_blank" rel="noopener">{esc(host)}</a>'
        )
    return "".join(parts)


def _render_details(item: Dict[str, Any]) -> str:
    details = item.get("details") or {}
    if not isinstance(details, dict) or not details:
        return ""
    rows = "".join(
        f'<div class="metric"><span class="k">{esc(k)}</span>'
        f'<span class="v">{_format_value(v)}</span></div>'
        for k, v in details.items()
    )
    return f'<div class="metrics">{rows}</div>'


def _render_bullets(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"


def _render_extra_sections(item: Dict[str, Any], sections: List[str]) -> str:
    out = []
    for sec in sections or []:
        val = _item_get(item, sec)
        if not val:
            continue
        body = _render_bullets(val) if isinstance(val, list) else f"<p>{esc(val)}</p>"
        if body:
            out.append(f"<section><h4>{esc(sec)}</h4>{body}</section>")
    return "".join(out)


def _render_card(item: Dict[str, Any], sections: List[str]) -> str:
    name = esc(item.get("name", "")) or "&mdash;"
    headline = esc(item.get("headline", ""))
    key_points = _render_bullets(item.get("key_points"))
    details = _render_details(item)
    extras = _render_extra_sections(item, sections)
    sources = _render_sources_chips(item)
    return f"""
    <article class="card">
      <header class="card-head">
        <div class="ticker">{name}</div>
      </header>
      {f'<p class="thesis">{headline}</p>' if headline else ''}
      {('<section><h4>Key Points</h4>' + key_points + '</section>') if key_points else ''}
      {details}
      {extras}
      <footer class="sources">{sources}</footer>
    </article>
    """


def _render_cards(items: List[Dict[str, Any]], sections: List[str]) -> str:
    sections = sections or DEFAULT_SECTIONS
    cards = "\n".join(_render_card(it, sections) for it in items)
    return f'<div class="grid">{cards}</div>'


def _render_table(items: List[Dict[str, Any]], columns: List[str]) -> str:
    columns = columns or DEFAULT_TABLE_COLUMNS
    thead = "".join(f"<th>{esc(c)}</th>" for c in columns)
    rows = []
    for it in items:
        cells = []
        for c in columns:
            val = _item_get(it, c)
            if c.lower() == "sources":
                cells.append(f"<td>{_render_sources_chips(it)}</td>")
            else:
                cells.append(f"<td>{_format_value(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<div class="tablewrap"><table class="results-table">'
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_ranked_list(items: List[Dict[str, Any]], sections: List[str]) -> str:
    sections = sections or DEFAULT_SECTIONS
    rows = []
    for i, it in enumerate(items, 1):
        name = esc(it.get("name", "")) or "&mdash;"
        headline = esc(it.get("headline", ""))
        rows.append(
            f"""
            <li class="ranked-item">
              <div class="rank">#{i}</div>
              <div class="body">
                <div class="head"><span class="ticker">{name}</span></div>
                {f'<p class="thesis">{headline}</p>' if headline else ''}
                {_render_bullets(it.get('key_points'))}
                {_render_details(it)}
                {_render_extra_sections(it, sections)}
                <div class="sources">{_render_sources_chips(it)}</div>
              </div>
            </li>
            """
        )
    return f'<ol class="ranked">{"".join(rows)}</ol>'


def _render_comparison(items: List[Dict[str, Any]], columns: List[str], sections: List[str]) -> str:
    attrs = list(columns) if columns else []
    if not attrs:
        attrs = ["headline"]
        for it in items:
            for k in (it.get("details") or {}).keys():
                if k not in attrs:
                    attrs.append(k)
        for s in (sections or DEFAULT_SECTIONS):
            if s not in attrs:
                attrs.append(s)
    header_cells = "".join(
        f'<th>{esc(it.get("name") or f"Item {i+1}")}</th>'
        for i, it in enumerate(items)
    )
    body_rows = []
    for attr in attrs:
        cells = []
        for it in items:
            val = _item_get(it, attr)
            if isinstance(val, list):
                cells.append(f"<td>{_render_bullets(val)}</td>")
            else:
                cells.append(f"<td>{_format_value(val)}</td>")
        body_rows.append(f"<tr><th class='attr'>{esc(attr)}</th>{''.join(cells)}</tr>")
    return (
        '<div class="tablewrap"><table class="compare-table">'
        f"<thead><tr><th></th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _render_narrative(items: List[Dict[str, Any]], sections: List[str]) -> str:
    """Long-form report. Each item becomes an H2 section."""
    blocks = []
    for it in items:
        name = esc(it.get("name", "")) or "Section"
        headline = esc(it.get("headline", ""))
        body = it.get("body") or ""
        body_html = ""
        if isinstance(body, str) and body.strip():
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
            body_html = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
        key_points = _render_bullets(it.get("key_points"))
        details = _render_details(it)
        extras = _render_extra_sections(it, sections or [])
        sources = _render_sources_chips(it)
        blocks.append(
            f"""
            <section class="narrative-section">
              <h2>{name}</h2>
              {f'<p class="lede">{headline}</p>' if headline else ''}
              {body_html}
              {('<div class="kp"><h4>Key Points</h4>' + key_points + '</div>') if key_points else ''}
              {details}
              {extras}
              <footer class="sources">{sources}</footer>
            </section>
            """
        )
    return '<div class="narrative">' + "".join(blocks) + "</div>"


def _render_timeline(items: List[Dict[str, Any]], sections: List[str]) -> str:
    def _key(it):
        return str(it.get("when") or it.get("date") or "")

    sorted_items = sorted(items, key=_key)
    blocks = []
    for it in sorted_items:
        when = esc(it.get("when") or it.get("date") or "")
        name = esc(it.get("name", "")) or "Event"
        headline = esc(it.get("headline", ""))
        body = it.get("body") or ""
        body_html = ""
        if isinstance(body, str) and body.strip():
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
            body_html = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
        sources = _render_sources_chips(it)
        blocks.append(
            f"""
            <li class="tl-item">
              <div class="tl-when">{when or '&middot;'}</div>
              <div class="tl-body">
                <h3>{name}</h3>
                {f'<p class="lede">{headline}</p>' if headline else ''}
                {body_html}
                {_render_bullets(it.get('key_points'))}
                <footer class="sources">{sources}</footer>
              </div>
            </li>
            """
        )
    return f'<ol class="timeline">{"".join(blocks)}</ol>'


LAYOUTS = {
    "cards": lambda items, cols, secs: _render_cards(items, secs),
    "table": lambda items, cols, secs: _render_table(items, cols),
    "ranked_list": lambda items, cols, secs: _render_ranked_list(items, secs),
    "comparison": lambda items, cols, secs: _render_comparison(items, cols, secs),
    "narrative": lambda items, cols, secs: _render_narrative(items, secs),
    "timeline": lambda items, cols, secs: _render_timeline(items, secs),
}


# ---------------------------------------------------------------------------
# Report-type renderers (richer templates + CSS-only tabbed grouping)
# ---------------------------------------------------------------------------

_TAB_GROUP_SEQ = 0


def _render_tabs(groups: List[Dict[str, Any]], render_group_body) -> str:
    """Render a CSS-only tab strip from `groups`.

    Each group is one radio input + label (the tab) + a panel. The first tab is
    checked by default. No JavaScript, so the report stays self-contained and
    works when opened directly from disk. `render_group_body(group)` returns the
    HTML for a single group's content.
    """
    global _TAB_GROUP_SEQ
    _TAB_GROUP_SEQ += 1
    name = f"tabs{_TAB_GROUP_SEQ}"
    radios, labels, panels = [], [], []
    for i, g in enumerate(groups):
        tab_id = f"{name}-{i}"
        checked = " checked" if i == 0 else ""
        count = len(g.get("items") or [])
        radios.append(
            f'<input type="radio" name="{name}" id="{tab_id}" class="tab-radio"{checked} />'
        )
        labels.append(
            f'<label for="{tab_id}" class="tab-label">{esc(g.get("name") or f"Group {i+1}")}'
            f'<span class="tab-count">{count}</span></label>'
        )
        summary = esc(g.get("summary") or "")
        summary_html = f'<p class="tab-summary">{summary}</p>' if summary else ""
        panels.append(
            f'<section class="tab-panel" data-tab="{tab_id}">{summary_html}'
            f"{render_group_body(g)}</section>"
        )
    return (
        '<div class="tabs">'
        + "".join(radios)
        + '<div class="tab-strip">'
        + "".join(labels)
        + "</div>"
        + "".join(panels)
        + "</div>"
    )


def _maybe_tabbed(
    state: State,
    render_items,
) -> str:
    """If `state.groups` exists, render tabs; otherwise render the flat items."""
    groups = state.groups or []
    if groups:
        return _render_tabs(groups, lambda g: render_items(g.get("items") or []))
    return render_items(state.items)


def _render_qa_chips(item: Dict[str, Any]) -> str:
    chips = []
    difficulty = str(_item_get(item, "difficulty") or "").strip()
    if difficulty:
        cls = "diff-" + re.sub(r"[^a-z]", "", difficulty.lower()) if difficulty else ""
        chips.append(f'<span class="badge {cls}">{esc(difficulty)}</span>')
    frequency = str(_item_get(item, "frequency") or "").strip()
    if frequency:
        chips.append(f'<span class="badge freq">{esc(frequency)}</span>')
    tags = _item_get(item, "company_tags")
    if isinstance(tags, list):
        for t in tags:
            if str(t).strip():
                chips.append(f'<span class="badge tag">{esc(t)}</span>')
    elif isinstance(tags, str) and tags.strip():
        chips.append(f'<span class="badge tag">{esc(tags)}</span>')
    return f'<div class="badges">{"".join(chips)}</div>' if chips else ""


def _render_qa_item(item: Dict[str, Any]) -> str:
    question = esc(_item_get(item, "question") or item.get("name", "")) or "&mdash;"
    headline = esc(item.get("headline", ""))
    answer = _item_get(item, "answer") or item.get("body") or ""
    answer_html = ""
    if isinstance(answer, str) and answer.strip():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
        answer_html = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    chips = _render_qa_chips(item)
    key_points = _render_bullets(item.get("key_points"))
    sources = _render_sources_chips(item)
    lede_html = f'<p class="lede">{headline}</p>' if headline else ""
    kp_html = f'<div class="kp"><h4>Key Points</h4>{key_points}</div>' if key_points else ""
    body = (
        f"{chips}"
        f"{lede_html}"
        f"{answer_html}"
        f"{kp_html}"
        f'<footer class="sources">{sources}</footer>'
    )
    return (
        '<details class="qa-item">'
        f'<summary class="qa-question">{question}</summary>'
        f'<div class="qa-answer">{body}</div>'
        "</details>"
    )


def _render_qa_items(items: List[Dict[str, Any]]) -> str:
    return '<div class="qa-list">' + "".join(_render_qa_item(it) for it in items) + "</div>"


def _render_qa_list(state: State) -> str:
    return _maybe_tabbed(state, _render_qa_items)


def _render_kpi_callouts(items: List[Dict[str, Any]]) -> str:
    """Pull `metrics` from the first item(s) into a KPI strip for financial reports."""
    kpis: List[tuple] = []
    for it in items:
        metrics = _item_get(it, "metrics")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                kpis.append((k, v))
        if len(kpis) >= 8:
            break
    if not kpis:
        return ""
    cards = "".join(
        f'<div class="kpi"><span class="kpi-v">{_format_value(v)}</span>'
        f'<span class="kpi-k">{esc(k)}</span></div>'
        for k, v in kpis[:8]
    )
    return f'<div class="kpi-strip">{cards}</div>'


def _render_financial_section(item: Dict[str, Any]) -> str:
    name = esc(item.get("name", "")) or "Section"
    headline = esc(item.get("headline", ""))
    body = item.get("body") or ""
    body_html = ""
    if isinstance(body, str) and body.strip():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
        body_html = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    metrics = _item_get(item, "metrics")
    metrics_html = ""
    if isinstance(metrics, dict) and metrics:
        rows = "".join(
            f'<div class="metric"><span class="k">{esc(k)}</span>'
            f'<span class="v">{_format_value(v)}</span></div>'
            for k, v in metrics.items()
        )
        metrics_html = f'<div class="metrics">{rows}</div>'
    key_points = _render_bullets(item.get("key_points"))
    details = _render_details(item)
    sources = _render_sources_chips(item)
    lede_html = f'<p class="lede">{headline}</p>' if headline else ""
    kp_html = f'<div class="kp"><h4>Key Points</h4>{key_points}</div>' if key_points else ""
    return (
        '<section class="fin-section">'
        f"<h2>{name}</h2>"
        f"{lede_html}"
        f"{metrics_html}"
        f"{body_html}"
        f"{kp_html}"
        f"{details}"
        f'<footer class="sources">{sources}</footer>'
        "</section>"
    )


def _render_financial_items(items: List[Dict[str, Any]]) -> str:
    callouts = _render_kpi_callouts(items)
    sections = "".join(_render_financial_section(it) for it in items)
    return f'{callouts}<div class="financial">{sections}</div>'


def _render_financial(state: State) -> str:
    if state.groups:
        callouts = _render_kpi_callouts(state.items)
        return callouts + _maybe_tabbed(state, _render_financial_items)
    return _render_financial_items(state.items)


def _render_analysis_section(item: Dict[str, Any]) -> str:
    name = esc(item.get("name", "")) or "Finding"
    headline = esc(item.get("headline", ""))
    body = item.get("body") or ""
    body_html = ""
    if isinstance(body, str) and body.strip():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
        body_html = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    evidence = _item_get(item, "evidence")
    evidence_html = ""
    if isinstance(evidence, list) and evidence:
        evidence_html = (
            '<div class="kp"><h4>Evidence</h4>' + _render_bullets(evidence) + "</div>"
        )
    key_points = _render_bullets(item.get("key_points"))
    details = _render_details(item)
    sources = _render_sources_chips(item)
    lede_html = f'<p class="lede">{headline}</p>' if headline else ""
    kp_html = f'<div class="kp"><h4>Key Points</h4>{key_points}</div>' if key_points else ""
    return (
        '<section class="analysis-section">'
        f"<h2>{name}</h2>"
        f"{lede_html}"
        f"{body_html}"
        f"{kp_html}"
        f"{evidence_html}"
        f"{details}"
        f'<footer class="sources">{sources}</footer>'
        "</section>"
    )


def _render_analysis_items(items: List[Dict[str, Any]]) -> str:
    return '<div class="analysis">' + "".join(_render_analysis_section(it) for it in items) + "</div>"


def _render_analysis(state: State) -> str:
    return _maybe_tabbed(state, _render_analysis_items)


def _render_comparison_v2(state: State) -> str:
    """Side-by-side comparison: attribute matrix + per-option pros/cons + verdict."""
    items = state.items
    if not items:
        return _render_comparison(items, [], [])
    columns = (state.presentation or {}).get("columns") or []

    matrix = _render_comparison(items, columns, (state.presentation or {}).get("sections") or [])

    cards = []
    for it in items:
        name = esc(it.get("name", "")) or "Option"
        headline = esc(it.get("headline", ""))
        pros = _item_get(it, "pros")
        cons = _item_get(it, "cons")
        verdict = esc(_item_get(it, "verdict") or "")
        pros_html = (
            '<div class="pros"><h4>Pros</h4>' + _render_bullets(pros) + "</div>"
            if isinstance(pros, list) and pros
            else ""
        )
        cons_html = (
            '<div class="cons"><h4>Cons</h4>' + _render_bullets(cons) + "</div>"
            if isinstance(cons, list) and cons
            else ""
        )
        verdict_html = f'<p class="verdict"><strong>Verdict:</strong> {verdict}</p>' if verdict else ""
        thesis_html = f'<p class="thesis">{headline}</p>' if headline else ""
        sources = _render_sources_chips(it)
        cards.append(
            '<article class="cmp-card">'
            f'<header class="card-head"><div class="ticker">{name}</div></header>'
            f"{thesis_html}"
            f'<div class="proscons">{pros_html}{cons_html}</div>'
            f"{verdict_html}"
            f'<footer class="sources">{sources}</footer>'
            "</article>"
        )
    cards_html = f'<div class="grid cmp-grid">{"".join(cards)}</div>'
    return matrix + cards_html


# report_type -> renderer(state) -> html. Each renderer reads state.items and
# state.groups directly so it can decide whether to use tabbed grouping.
REPORT_TYPES = {
    "qa_list": _render_qa_list,
    "financial": _render_financial,
    "analysis": _render_analysis,
    "comparison": _render_comparison_v2,
}


def _render_results_section(state: State) -> str:
    if not state.items and not state.groups:
        return (
            '<div class="fallback"><h3>No structured items produced</h3>'
            f'<pre>{esc(state.summary_raw)}</pre></div>'
        )

    # Prefer a dedicated report-type template when the summarizer picked one.
    report_type = (state.report_type or "").strip().lower()
    if report_type in REPORT_TYPES:
        return REPORT_TYPES[report_type](state)

    layout = (state.presentation or {}).get("layout", "narrative")
    layout = layout if layout in LAYOUTS else "narrative"
    columns = (state.presentation or {}).get("columns") or []
    sections = (state.presentation or {}).get("sections") or []
    return LAYOUTS[layout](state.items, columns, sections)


def _render_side_panels(state: State) -> str:
    ok_sources = (
        "".join(
            f'<li><a href="{esc(u)}" target="_blank" rel="noopener">{esc(u)}</a></li>'
            for u in state.successful_sources
        )
        or "<li class='muted'>(none)</li>"
    )
    failed_sources = (
        "".join(f"<li>{esc(u)}</li>" for u in state.failed_sources[:30])
        or "<li class='muted'>(none)</li>"
    )
    missing_block = (
        "".join(f"<li>{esc(m)}</li>" for m in state.missing[:30])
        or "<li class='muted'>(none)</li>"
    )
    facets_block = (
        "".join(f"<li>{esc(f)}</li>" for f in state.facets)
        or "<li class='muted'>(none)</li>"
    )
    signals = state.last_signals or {}
    signals_html = "".join(
        f"<li><strong>{esc(k)}</strong>: {esc(v)}</li>"
        for k, v in signals.items()
        if not isinstance(v, (list, dict))
    ) or "<li class='muted'>(no signals)</li>"

    return f"""
      <div class="row">
        <div class="panel plan">
          <h3>Research plan</h3>
          <pre>{esc(state.plan) or "(none)"}</pre>
        </div>
        <div class="panel">
          <h3>Facets</h3>
          <ul>{facets_block}</ul>
        </div>
      </div>
      <div class="row">
        <div class="panel">
          <h3>Sources used</h3>
          <ul>{ok_sources}</ul>
        </div>
        <div class="panel">
          <h3>Skipped / blocked</h3>
          <ul>{failed_sources}</ul>
        </div>
      </div>
      <div class="row">
        <div class="panel">
          <h3>Run stats</h3>
          <ul>
            <li>Topic slug: <span class="mono">{esc(state.topic_slug)}</span></li>
            <li>Run id: <span class="mono">{esc(state.run_id)}</span></li>
            <li>Iterations: {state.iteration}</li>
            <li>Time elapsed: {state.time_elapsed():.0f}s</li>
            <li>Successful: {len(state.successful_sources)}</li>
            <li>Failed / blocked: {len(state.failed_sources)}</li>
            <li>Items produced: {len(state.items)}</li>
            <li>Final confidence: {state.confidence:.2f}</li>
            <li>Report type: {esc(state.report_type or "general")}</li>
            <li>Layout: {esc((state.presentation or {}).get("layout", "narrative"))}</li>
            <li>Groups / tabs: {len(state.groups)}</li>
            <li>Creativity boosts: {state.creativity_boosts}</li>
          </ul>
        </div>
        <div class="panel">
          <h3>Conviction signals</h3>
          <ul>{signals_html}</ul>
        </div>
      </div>
      <div class="row">
        <div class="panel">
          <h3>Still-missing info (last pass)</h3>
          <ul>{missing_block}</ul>
        </div>
        <div class="panel">
          <h3>Run log</h3>
          <p class="muted mono">{esc(state.research_file) or "(none)"}</p>
        </div>
      </div>
    """


def _page_css() -> str:
    return """
    :root {
      --bg: #0b1020; --panel: #141a30; --panel-2: #1b2340;
      --text: #e7ecf5; --muted: #8b95ad; --accent: #7aa2ff;
      --border: #253056;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, #1a2350 0%, transparent 60%),
        radial-gradient(800px 500px at 110% 10%, #2a1a4a 0%, transparent 55%),
        var(--bg);
      color: var(--text); min-height: 100vh;
    }
    .container { max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; }
    header.page { margin-bottom: 24px; }
    .eyebrow { color: var(--accent); letter-spacing: .14em; font-size: 12px; text-transform: uppercase; }
    h1 { margin: 8px 0 4px; font-size: 32px; line-height: 1.2; }
    h2 { margin: 24px 0 8px; font-size: 22px; }
    h3 { margin: 14px 0 6px; font-size: 18px; }
    .task { color: var(--muted); margin: 0 0 14px; }
    .interpretation {
      background: linear-gradient(135deg, #1d2550, #161b3a);
      border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 14px; margin: 12px 0; font-size: 14px; color: #cdd5f0;
    }
    .headline {
      background: linear-gradient(135deg, #2a3568, #1a2350);
      border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; margin: 16px 0 0; font-size: 15px;
    }
    .grid {
      display: grid; gap: 18px; margin-top: 24px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 14px; padding: 18px 18px 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,.25);
      display: flex; flex-direction: column; gap: 10px;
    }
    .card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
    .ticker { font-size: 22px; font-weight: 700; letter-spacing: .02em; }
    .thesis { margin: 4px 0 2px; line-height: 1.5; }
    .card section h4, .ranked-item section h4, .narrative-section h4 {
      margin: 6px 0 4px; font-size: 12px; text-transform: uppercase;
      letter-spacing: .1em; color: var(--muted);
    }
    .card section ul, .ranked-item ul, .narrative-section ul, .tl-body ul { margin: 0; padding-left: 18px; }
    .card section li, .ranked-item li, .narrative-section li, .tl-body li { margin: 2px 0; font-size: 14px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 6px; }
    .metric {
      display: flex; flex-direction: column; padding: 6px 8px;
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
    }
    .metric .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .metric .v { font-size: 14px; font-weight: 600; }
    .sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .chip {
      background: var(--panel-2); color: var(--text); text-decoration: none;
      font-size: 12px; padding: 4px 9px; border-radius: 999px;
      border: 1px solid var(--border);
    }
    .chip:hover { border-color: var(--accent); color: var(--accent); }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }

    .tablewrap { margin-top: 24px; overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
    table.results-table, table.compare-table {
      width: 100%; border-collapse: collapse; background: var(--panel);
    }
    table.results-table th, table.results-table td,
    table.compare-table th, table.compare-table td {
      padding: 10px 12px; border-bottom: 1px solid var(--border);
      text-align: left; vertical-align: top; font-size: 14px;
    }
    table.results-table thead th, table.compare-table thead th {
      background: var(--panel-2); color: var(--accent);
      font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
    }
    table.compare-table th.attr {
      color: var(--muted); font-weight: 600;
      text-transform: uppercase; letter-spacing: .08em; font-size: 12px; width: 180px;
    }
    table.compare-table tbody tr:hover td, table.results-table tbody tr:hover td {
      background: rgba(122,162,255,.06);
    }

    ol.ranked { list-style: none; padding: 0; margin: 24px 0 0; display: flex; flex-direction: column; gap: 12px; }
    .ranked-item {
      display: grid; grid-template-columns: 64px 1fr; gap: 14px;
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    }
    .ranked-item .rank {
      font-size: 28px; font-weight: 800; color: var(--accent); line-height: 1;
    }
    .ranked-item .head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
    .ranked-item .head .ticker { font-size: 20px; }

    .narrative { margin-top: 20px; }
    .narrative-section {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 18px 22px; margin: 16px 0;
    }
    .narrative-section .lede { color: var(--accent); font-size: 16px; }
    .narrative-section p { line-height: 1.65; font-size: 15px; }
    .narrative-section .kp { margin-top: 10px; }

    ol.timeline { list-style: none; padding: 0; margin: 24px 0 0;
      border-left: 2px solid var(--border); padding-left: 20px;
    }
    .tl-item { position: relative; margin-bottom: 18px; }
    .tl-item::before {
      content: ''; position: absolute; left: -27px; top: 8px;
      width: 12px; height: 12px; border-radius: 50%;
      background: var(--accent); border: 2px solid var(--bg);
    }
    .tl-when { color: var(--accent); font-size: 13px; text-transform: uppercase; letter-spacing: .1em; }
    .tl-body { background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 12px 16px; margin-top: 4px;
    }
    .tl-body h3 { margin-top: 0; }
    .tl-body .lede { color: var(--accent); }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 24px; }
    .panel {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px;
    }
    .panel h3 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
    .panel ul { margin: 0; padding-left: 18px; }
    .panel li { margin: 3px 0; font-size: 14px; word-break: break-all; }
    .panel a { color: var(--accent); text-decoration: none; }
    .panel a:hover { text-decoration: underline; }
    .plan pre { white-space: pre-wrap; margin: 0; font-family: inherit; font-size: 14px; color: var(--text); }
    .fallback pre {
      white-space: pre-wrap; background: var(--panel); border: 1px solid var(--border);
      padding: 14px; border-radius: 12px;
    }
    footer.page { margin-top: 36px; color: var(--muted); font-size: 12px; text-align: center; }
    .disclaimer {
      margin-top: 24px; padding: 12px 14px; border-radius: 10px;
      background: rgba(241,196,15,.08); border: 1px solid rgba(241,196,15,.25);
      color: #f1c40f; font-size: 13px;
    }
    /* ---- CSS-only tabs (report grouping) ---- */
    .tabs { margin-top: 24px; }
    .tabs > .tab-radio { position: absolute; opacity: 0; pointer-events: none; }
    .tab-strip {
      display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px;
      border-bottom: 1px solid var(--border); padding-bottom: 8px;
    }
    .tab-label {
      cursor: pointer; user-select: none; padding: 8px 14px; border-radius: 999px;
      background: var(--panel); border: 1px solid var(--border); color: var(--muted);
      font-size: 14px; display: inline-flex; align-items: center; gap: 8px;
    }
    .tab-label:hover { color: var(--text); border-color: var(--accent); }
    .tab-count {
      background: var(--panel-2); color: var(--muted); font-size: 11px;
      padding: 1px 7px; border-radius: 999px; border: 1px solid var(--border);
    }
    .tab-panel { display: none; }
    .tab-summary { color: var(--muted); margin: 0 0 14px; font-size: 14px; }
    /* Show the panel whose preceding radio is checked. The radios are emitted
       before the strip+panels, so we target subsequent siblings by order. */
    .tabs > .tab-radio:nth-of-type(1):checked ~ .tab-panel:nth-of-type(1),
    .tabs > .tab-radio:nth-of-type(2):checked ~ .tab-panel:nth-of-type(2),
    .tabs > .tab-radio:nth-of-type(3):checked ~ .tab-panel:nth-of-type(3),
    .tabs > .tab-radio:nth-of-type(4):checked ~ .tab-panel:nth-of-type(4),
    .tabs > .tab-radio:nth-of-type(5):checked ~ .tab-panel:nth-of-type(5),
    .tabs > .tab-radio:nth-of-type(6):checked ~ .tab-panel:nth-of-type(6),
    .tabs > .tab-radio:nth-of-type(7):checked ~ .tab-panel:nth-of-type(7),
    .tabs > .tab-radio:nth-of-type(8):checked ~ .tab-panel:nth-of-type(8),
    .tabs > .tab-radio:nth-of-type(9):checked ~ .tab-panel:nth-of-type(9),
    .tabs > .tab-radio:nth-of-type(10):checked ~ .tab-panel:nth-of-type(10),
    .tabs > .tab-radio:nth-of-type(11):checked ~ .tab-panel:nth-of-type(11),
    .tabs > .tab-radio:nth-of-type(12):checked ~ .tab-panel:nth-of-type(12),
    .tabs > .tab-radio:nth-of-type(13):checked ~ .tab-panel:nth-of-type(13),
    .tabs > .tab-radio:nth-of-type(14):checked ~ .tab-panel:nth-of-type(14),
    .tabs > .tab-radio:nth-of-type(15):checked ~ .tab-panel:nth-of-type(15),
    .tabs > .tab-radio:nth-of-type(16):checked ~ .tab-panel:nth-of-type(16) { display: block; }
    .tabs > .tab-radio:nth-of-type(1):checked ~ .tab-strip label[for$="-0"],
    .tabs > .tab-radio:nth-of-type(2):checked ~ .tab-strip label[for$="-1"],
    .tabs > .tab-radio:nth-of-type(3):checked ~ .tab-strip label[for$="-2"],
    .tabs > .tab-radio:nth-of-type(4):checked ~ .tab-strip label[for$="-3"],
    .tabs > .tab-radio:nth-of-type(5):checked ~ .tab-strip label[for$="-4"],
    .tabs > .tab-radio:nth-of-type(6):checked ~ .tab-strip label[for$="-5"],
    .tabs > .tab-radio:nth-of-type(7):checked ~ .tab-strip label[for$="-6"],
    .tabs > .tab-radio:nth-of-type(8):checked ~ .tab-strip label[for$="-7"] {
      background: var(--panel-2); color: var(--accent); border-color: var(--accent);
    }

    /* ---- Badges / chips (qa difficulty, frequency, company tags) ---- */
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px; }
    .badge {
      font-size: 11px; padding: 3px 9px; border-radius: 999px;
      border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
      text-transform: uppercase; letter-spacing: .05em;
    }
    .badge.tag { color: var(--accent); }
    .badge.freq { color: #f1c40f; border-color: rgba(241,196,15,.35); }
    .badge.diff-easy { color: #2ecc71; border-color: rgba(46,204,113,.35); }
    .badge.diff-medium { color: #f39c12; border-color: rgba(243,156,18,.35); }
    .badge.diff-hard { color: #e74c3c; border-color: rgba(231,76,60,.35); }

    /* ---- Q&A list ---- */
    .qa-list { display: flex; flex-direction: column; gap: 10px; }
    .qa-item {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 4px 16px;
    }
    .qa-item[open] { padding-bottom: 14px; }
    .qa-question {
      cursor: pointer; font-size: 16px; font-weight: 600; padding: 12px 0;
      list-style: none; display: flex; align-items: center; gap: 8px;
    }
    .qa-question::-webkit-details-marker { display: none; }
    .qa-question::before { content: '+'; color: var(--accent); font-weight: 700; }
    .qa-item[open] .qa-question::before { content: '\\2212'; }
    .qa-answer { padding-left: 18px; }
    .qa-answer p { line-height: 1.6; font-size: 14px; }
    .qa-answer .lede { color: var(--accent); }

    /* ---- KPI strip (financial) ---- */
    .kpi-strip {
      display: grid; gap: 12px; margin: 24px 0 8px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .kpi {
      background: linear-gradient(135deg, #1d2550, #161b3a);
      border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
      display: flex; flex-direction: column; gap: 4px;
    }
    .kpi-v { font-size: 22px; font-weight: 700; color: var(--accent); }
    .kpi-k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

    /* ---- Financial / analysis sections ---- */
    .fin-section, .analysis-section {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 18px 22px; margin: 16px 0;
    }
    .fin-section .lede, .analysis-section .lede { color: var(--accent); font-size: 16px; }
    .fin-section p, .analysis-section p { line-height: 1.65; font-size: 15px; }

    /* ---- Comparison v2 (pros/cons + verdict) ---- */
    .cmp-grid { margin-top: 18px; }
    .cmp-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 14px; padding: 18px; display: flex; flex-direction: column; gap: 10px;
    }
    .proscons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .proscons .pros h4 { color: #2ecc71; }
    .proscons .cons h4 { color: #e74c3c; }
    .proscons h4 { margin: 0 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .proscons ul { margin: 0; padding-left: 18px; }
    .proscons li { font-size: 14px; margin: 3px 0; }
    .verdict {
      margin: 4px 0 0; padding: 10px 12px; border-radius: 10px;
      background: var(--panel-2); border: 1px solid var(--border); font-size: 14px;
    }

    @media (max-width: 820px) {
      .row { grid-template-columns: 1fr; }
      .proscons { grid-template-columns: 1fr; }
    }
    """


def _parse_run_id(run_id: str) -> tuple:
    """Pull (YYYY-MM-DD, HHMM-SS) out of a `YYYYMMDD-HHMMSS-<hex>` run id.

    Falls back to today's date and 0000-00 if the run id is malformed.
    """
    today = datetime.now()
    date_part = today.strftime("%Y-%m-%d")
    time_part = today.strftime("%H%M-%S")
    if run_id and len(run_id) >= 15 and run_id[8] == "-":
        ymd = run_id[:8]
        hms = run_id[9:15]
        if ymd.isdigit() and hms.isdigit():
            date_part = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
            time_part = f"{hms[:4]}-{hms[4:6]}"
    return date_part, time_part


def _resolve_output_path(state: State) -> Path:
    """Resolve the output HTML path based on `RESEARCH_OUTPUT_LAYOUT`.

    `flat_by_date` (default):
        output/<YYYY-MM-DD>/<topic-slug>--<HHMM-SS>.html

    `per_run_dir` (legacy):
        output/<topic-slug>/<run_id>/index.html
    """
    config.ensure_dirs()
    slug = state.topic_slug or "topic"
    run_id = state.ensure_run_id()
    if (config.OUTPUT_LAYOUT or "flat_by_date") == "per_run_dir":
        out_dir = config.OUTPUT_ROOT / slug / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "index.html"
    date_part, time_part = _parse_run_id(run_id)
    out_dir = config.OUTPUT_ROOT / date_part
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{slug}--{time_part}.html"


def _resolve_output_dir(state: State) -> Path:
    """Backwards-compatible helper (returns the parent dir of the report)."""
    return _resolve_output_path(state).parent


def generate_html(state: State, output_path: Optional[Path] = None) -> Path:
    print("Generating HTML...")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    presentation = state.presentation or {}
    title = esc(presentation.get("title") or state.task[:80] or "Research Results")
    headline = esc(presentation.get("headline") or state.headline)
    interpretation = esc(state.interpretation)

    results_section = _render_results_section(state)
    side_panels = _render_side_panels(state)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title} \u2014 {esc(state.task)}</title>
  <style>{_page_css()}</style>
</head>
<body>
  <div class="container">
    <header class="page">
      <div class="eyebrow">AI Research Agent</div>
      <h1>{title}</h1>
      <p class="task">Task: <strong>{esc(state.task)}</strong></p>
      {f'<p class="interpretation">{interpretation}</p>' if interpretation else ''}
      {f'<p class="headline">{headline}</p>' if headline else ''}
    </header>

    <main>
      {results_section}
      {side_panels}
      <div class="disclaimer">
        Informational only. Items are generated by an LLM from public web content
        and may be inaccurate. Verify critical facts via the cited sources.
      </div>
    </main>

    <footer class="page">Generated {esc(generated_at)} by research_agent.py</footer>
  </div>
</body>
</html>
"""

    out_file = output_path or _resolve_output_path(state)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_write_text(out_file, html)
    state.output_dir = str(out_dir)
    state.results_html = html

    if state.memory is not None:
        try:
            state.memory.snapshot_run(state.run_id, state.to_snapshot())
        except Exception:
            pass

    print(f"  Wrote {out_file} (layout={presentation.get('layout', 'narrative')})\n")
    return out_file
