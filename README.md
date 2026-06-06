# LangResearch — Generic AI Research Agent

A topic-agnostic, time-bounded research agent that browses the open web,
discovers sources, extracts atomic claims, builds conviction across multiple
signals, and produces a self-contained HTML report.

It will keep researching as long as you give it time — pulling new sources
from search engines, harvesting links from pages it has already crawled,
asking the LLM for niche suggestions, and replaying domains that worked well
on the same topic in the past. When it gets stuck, it deliberately changes
angle (contrarian framings, different time windows, niche outlets) instead of
giving up.

---

## Highlights

- **Topic-agnostic.** Works for any subject — stocks, science, sports, history,
  policy, recipes — with no hard-coded source lists.
- **HTTP-first fetching.** Most pages are pulled with `httpx` + Trafilatura
  (fast, low-memory, hard to detect). Playwright stealth Chromium is kept as
  an automatic escalation tier — it only spins up when a site really does
  need JS rendering or anti-bot stealth.
- **Site-specific API adapters.** Wikipedia, arXiv, OpenAlex, Crossref, Hacker
  News, RSS/Atom, and `sitemap.xml` are queried as APIs (clean JSON/XML),
  not scraped — zero anti-bot risk, no extra config.
- **Per-host tier learning.** Each host remembers which fetch tier worked
  (api / http / browser / jina / wayback) and starts there next time, so
  long-running jobs get faster and more reliable over time.
- **Strict per-URL deadlines.** A single URL cannot stall the agent — there
  is a hard wall-clock budget across all tiers.
- **Unlimited discovery.** Search + outbound-link harvesting +
  LLM suggestions + memory replay, all combined and ranked into a persistent
  frontier.
- **No paid services required.** Runs end-to-end on one LLM API key
  (MiniMax recommended). Self-hosted SearXNG and Brave Search are *optional*
  upgrades.
- **Persistent memory per topic.** Each topic gets its own folder under
  `memory/<topic_slug>/` so subsequent runs on the same topic start warm —
  trusted domains, prior claims, and known-good URLs are all reused.
- **Multi-signal conviction.** Coverage, source diversity, plateau in new
  claims, contradiction load, and an LLM judgement combine into a single 0–1
  score. The agent stops when conviction is high *and* coverage is good — not
  on a single LLM "I'm done" call.
- **Time budget aware.** Pass `--time 2h` and the agent paces itself to use the
  whole budget without giving up early.
- **Generic output schema** with six layouts: `cards`, `table`, `ranked_list`,
  `comparison`, `narrative`, `timeline`. The LLM picks the right one for the task.

---

## Quick start (basic — MiniMax only)

The agent runs end-to-end with **just one environment variable**: your LLM key.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install the headless browser (only used when the HTTP tier escalates)
playwright install chromium

# 3. Set your LLM key — this is the ONLY required env var
export MINIMAX_API_KEY=your-key-here

# 4. Run
python research_agent.py "Compare LFP vs NMC battery chemistries" --time 30m
```

That's it. With nothing else configured you get:

- HTTP-first fetching with Trafilatura article extraction (Tier 1)
- API adapters for Wikipedia / arXiv / OpenAlex / Crossref / HN / RSS / sitemap
- DuckDuckGo search (keyless)
- Playwright stealth Chromium as an automatic escalation tier
- `r.jina.ai` and Wayback Machine fallbacks
- Local sentence-transformers embeddings (~80 MB, downloaded on first run)

> **Note on `MINIMAX_MODEL`:** the default model name is `MiniMax-M2.7`. Override
> via `export MINIMAX_MODEL=...` if you have access to a different model.

---

## Watching a run live

The agent prints a one-block **health snapshot** at the end of every iteration
so you can tell at a glance whether the run is healthy without reading every
log line:

```
[Health iter 5] conviction=0.27 (+0.07) coverage=2/10 frontier=86 visited=30 claims=29
  sources(last 3): link=68% llm=24% search=4% replay=4%
  top hosts: cisa.gov=12(40%) ti.com=8(27%) semianalysis.com=4(13%)
  productivity: 6/8 pages produced claims this iter (75%)
  [ok]  no issues detected
```

The block is followed by `[WARN]` lines when something is going wrong, each
telling you *what to do* — not just the symptom:

| Alert | What it means | What to try |
| --- | --- | --- |
| `SEARCH STALLED` | No `search`-bucket candidates in 3+ iterations | Self-host SearXNG (`RESEARCH_SEARXNG_URL`) or set `RESEARCH_BRAVE_API_KEY`. Inspect the per-query `[ddg/html]` log lines just above for the actual reason (CAPTCHA / parse miss / etc.) |
| `RABBIT HOLE` | Top 2 hosts hold >60% of visited URLs | Wait for the automatic creativity boost, or restart with a fresh `--topic-slug` if the LLM's initial seeds were wrong |
| `LOW PRODUCTIVITY` | <40% of recent pages produced any claims | The backflow filter prevents further damage; check whether the seeds are actually on-topic |
| `COVERAGE SKEW` | Some facets have many claims while ≥50% have zero | The agent is over-indexed on a few angles; the creativity boost will diversify, or you can edit `state.missing` |
| `CONVICTION FLAT` | <0.02 change in last 4 iterations | Plateau detection will trigger a creativity boost shortly |

Health is purely diagnostic — alerts don't change behaviour. The pipeline still
relies on its built-in `conviction` + `is_stagnating` signals to decide when to
stop or boost creativity. The block exists so *you* can spot trouble without
having to scroll.

The block is also keyed off small sliding windows in `State`
(`conviction_history`, `source_mix_history`, `productivity_history`), so all
the cost is a handful of integer appends per iteration.

---

## CLI

```
python research_agent.py "<your research task>" [options]
```

| Option | Description |
| --- | --- |
| `--time SPEC` | Wall-clock budget. Accepts `90s`, `15m`, `2h`, `1h30m`, etc. |
| `--minutes N` | Wall-clock budget in minutes (alternative to `--time`). |
| `--topic-slug NAME` | Override the auto-generated slug (used for `memory/<slug>/` and `output/<slug>/`). Useful if you want to keep growing a memory across reworded prompts. |
| `--workers N` | Parallel workers per iteration (default `4`). |
| `--max-iterations N` | Hard safety cap on iterations (default `500`). |
| `--no-llm-judge` | Skip the LLM-based conviction judge (faster, cheaper). |

Examples:

```bash
# Long, deep research session
python research_agent.py "High conviction tech stocks for 2026" --time 2h --workers 6

# Quick exploration
python research_agent.py "What is GLP-1 and what are its known side effects?" --minutes 10

# Continue building memory under a stable slug across multiple reworded prompts
python research_agent.py "tesla q4 earnings" --topic-slug tesla-2026 --time 30m
python research_agent.py "Tesla 2026 deliveries"  --topic-slug tesla-2026 --time 30m
```

---

## What you get

After a run finishes, your report is a single self-contained HTML file
stored under `output/<YYYY-MM-DD>/`, named with the topic subject and
start time so you can identify it at a glance:

```
output/
  2026-05-24/
    compare-lfp-vs-nmc-battery-chemistries--1209-23.html
    high-conviction-tech-stocks-2026--1430-05.html
  2026-05-25/
    what-is-glp-1-side-effects--1015-44.html
```

Just double-click the `.html` file to open it in your browser — no
`index.html` navigation, no per-run subfolders. Filename format:

```
<topic-slug>--<HHMM-SS>.html
```

The report adapts to the **type of question** you asked. The summarizer first
classifies a `report_type` and then renders a tailored template:

- `qa_list` — question/answer sets (e.g. interview questions) as collapsible
  entries with difficulty / frequency / company-tag badges, split into
  category **tabs** when the answer is naturally grouped.
- `financial` — a company / financial assessment with a KPI strip and
  sectioned report bodies (Valuation, Fundamentals, Segments, Risks, Outlook).
- `analysis` — a deep dive on a concept or method, with narrative sections and
  an explicit Evidence block per finding.
- `comparison` — a head-to-head with a side-by-side attribute matrix plus
  per-option pros/cons and a verdict.

Anything that doesn't fit a special type falls back to one of the six base
layouts (`cards` / `table` / `ranked_list` / `comparison` / `narrative` /
`timeline`). Tabs are pure CSS, so the report stays a single self-contained
file that opens offline. Side panels show: the plan, facets, sources used,
blocked sources, conviction signals, run stats (report type + tab count),
still-missing info, and the run-log path. The report cites only URLs the
agent successfully fetched.

### Where intermediate data lives

All intermediate state — frontier, visited URLs, extracted pages, claims,
embeddings, domain trust, per-run event log — lives **separately** under
`memory/<topic-slug>/` (one folder per topic, persists across runs). The
output folder is read-only artifacts; the memory folder is the agent's
working state. See [How memory grows](#how-memory-grows) below for the
full layout.

> Legacy layout: if you have tooling that depended on the old
> `output/<topic_slug>/<run_id>/index.html` shape, set
> `RESEARCH_OUTPUT_LAYOUT=per_run_dir` to switch back.

---

## Optional enhancements

Everything below is **opt-in**. Set only what you need. The agent works
without any of these, but each one improves a specific dimension.

### 1. Better search coverage (recommended for long runs)

Out of the box, the agent queries only DuckDuckGo. Adding a second search
backend reduces single-engine bans and diversifies the candidate frontier —
important on multi-day jobs.

**Option A: SearXNG (self-hosted, recommended).** Free, unlimited, queries
many engines (Google, Bing, Brave, Mojeek, Wikipedia, etc.) at once.

```bash
# One-time: run SearXNG locally via Docker
docker run -d --name searxng -p 8888:8080 searxng/searxng

# Point the agent at it
export RESEARCH_SEARXNG_URL=http://localhost:8888
```

> Public SearXNG instances (e.g. `searx.be`) work too, but they rate-limit
> automated traffic aggressively. Self-host for multi-day runs.

**Option B: Brave Search API.** Generous free tier (2,000 queries/month).
Sign up at <https://search.brave.com/api/>.

```bash
export RESEARCH_BRAVE_API_KEY=your-key-here
```

If both SearXNG and Brave are set, the agent queries all available backends
in parallel and uses whichever returns results first per query.

### 2. Tuning the fetch ladder (multi-day workloads)

The defaults are tuned for typical research sessions. These knobs help when
running for hours or days:

| Variable | Default | Effect |
| --- | --- | --- |
| `RESEARCH_URL_TOTAL_DEADLINE` | `12.0` | Hard wall-clock budget per URL across **all** tiers. Lower to `8.0` if you keep seeing stuck pages. |
| `RESEARCH_HTTP_TIMEOUT` | `8.0` | Per-request timeout for the httpx (Tier 1) fetch. |
| `RESEARCH_HTTP_PER_HOST_CONCURRENCY` | `2` | Max parallel requests to the same host. |
| `RESEARCH_HTTP_MAX_CONNECTIONS` | `32` | Global httpx connection pool size. |
| `RESEARCH_BROWSER_DEFAULT` | `escalation_only` | When Playwright runs: `escalation_only` (default), `off` (never), or `first` (use it eagerly — not recommended). |
| `RESEARCH_BROWSER_POOL_SIZE` | `4` | Number of stealth contexts reused across pages (replaces the old per-URL context creation). |
| `RESEARCH_BROWSER_RECYCLE_EVERY` | `50` | Pages per stealth context before recycling. Prevents Chromium memory bloat over long runs. |

### 3. Other LLM providers

The pipeline runs end-to-end on **any single** provider. MiniMax is the
documented default; alternatives:

```bash
# OpenAI
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini      # optional

# Anthropic
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-3-5-sonnet-latest   # optional

# Local Ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1         # optional

# Force a specific provider regardless of which keys are set
export RESEARCH_LLM_PROVIDER=minimax
```

Provider auto-detection priority (when not forced):
**MiniMax → OpenAI → Anthropic → Ollama**.

### 4. OpenAI embeddings (optional)

Embeddings default to a local sentence-transformers model (~80 MB, downloaded
on first run, no key). If you'd rather use OpenAI embeddings:

```bash
export RESEARCH_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small   # optional
```

---

## How memory grows

Each **topic** has its own folder at `memory/<topic_slug>/`. The folder
persists across runs on the same topic, so the second run starts warm and
each subsequent run gets faster and more reliable.

```
memory/
  <topic-slug>/
    meta.json                  # task variants, first_seen, last_run
    frontier.jsonl             # pending URL candidates with scores
    visited.jsonl              # every URL we've tried + final status + which tier worked
    pages/<sha1>.json          # extracted text + outbound links per URL
    claims.jsonl               # atomic claims with support/contradicts/sources
    domain_trust.json          # host -> {ok, fail, blocked, tier_success, preferred_tier}
    embeddings.npy             # claim embeddings for dedupe / similarity
    embeddings_index.jsonl     # per-row metadata for embeddings.npy
    runs/
      <run-id>/                # YYYYMMDD-HHMMSS-<hex>
        events.jsonl           # live event log (every fetch, every claim, every tier transition)
        state.json             # end-of-run snapshot
```

What "warm start" buys you on a repeat run:

- **Skips already-visited URLs** so each run pushes the corpus forward.
- **Replays known-good URLs** from trusted domains (memory's
  `domain_trust.json` records which hosts have historically returned good
  content).
- **Picks the right fetch tier per host**: each host's
  `preferred_tier` is updated after every success so revisits jump
  straight to the tier that worked before (api / http / browser / jina /
  wayback) instead of climbing the ladder from scratch.
- **Reuses extracted claims** — repeated facts across runs aren't
  duplicated, they accumulate as `support++` on the same claim.

Memory is intentionally separate from `output/`. The output folder holds
the **final reports** (what you read); the memory folder holds the
**working state** (what the agent reads/writes during a run). You can wipe
`output/` freely without losing the agent's accumulated knowledge.

---

## Reference: all environment variables

Most knobs have sensible defaults. The only env var you need to set for a
basic run is **`MINIMAX_API_KEY`** (or another LLM provider key).

### LLM (set exactly one provider)

| Variable | Effect |
| --- | --- |
| `MINIMAX_API_KEY`, `MINIMAX_MODEL` | Use MiniMax. Default model `MiniMax-M2.7`. **Recommended.** |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Use OpenAI. Default model `gpt-4o-mini`. |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Use Anthropic. Default `claude-3-5-sonnet-latest`. |
| `OLLAMA_HOST`, `OLLAMA_MODEL` | Use a local Ollama. Default model `llama3.1`. |
| `RESEARCH_LLM_PROVIDER` | Force a specific provider (`minimax` / `openai` / `anthropic` / `ollama`). |
| `RESEARCH_LLM_TEMPERATURE` | Default chat temperature (default `0.4`). |
| `RESEARCH_LLM_TIMEOUT` | LLM request timeout in seconds (default `180`). |

### Search (all optional)

| Variable | Effect |
| --- | --- |
| `RESEARCH_SEARXNG_URL` | URL of a SearXNG instance (e.g. `http://localhost:8888`). |
| `RESEARCH_BRAVE_API_KEY` | Enables Brave Search API as an additional backend. |
| `RESEARCH_DDG_RESULTS` | Results requested per DuckDuckGo query (default `12`). |

### Embeddings (optional)

| Variable | Effect |
| --- | --- |
| `RESEARCH_EMBEDDING_PROVIDER` | `local` (default) or `openai` (requires `OPENAI_API_KEY`). |
| `RESEARCH_EMBEDDING_MODEL` | Local sentence-transformers model name. |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small`. |

### Crawling / fetch ladder

| Variable | Effect |
| --- | --- |
| `RESPECT_ROBOTS` | `1` (default) to honor robots.txt, `0` to ignore. |
| `RESEARCH_URL_TOTAL_DEADLINE` | Hard per-URL wall-clock budget across all tiers (default `12.0` seconds). |
| `RESEARCH_HTTP_TIMEOUT` | Tier 1 httpx request timeout (default `8.0`). |
| `RESEARCH_HTTP_PER_HOST_CONCURRENCY` | Per-host parallel httpx requests (default `2`). |
| `RESEARCH_HTTP_MAX_CONNECTIONS` | Global httpx connection pool size (default `32`). |
| `RESEARCH_BROWSER_DEFAULT` | `escalation_only` (default), `off`, or `first`. |
| `RESEARCH_BROWSER_POOL_SIZE` | Number of pooled stealth contexts (default `4`). |
| `RESEARCH_BROWSER_RECYCLE_EVERY` | Pages per stealth context before recycle (default `50`). |
| `RESEARCH_PER_HOST_MIN_INTERVAL` | Min seconds between hits on the same host (default `1.5`). |
| `RESEARCH_PAGE_NAV_TIMEOUT_MS` | Playwright page nav timeout (default `30000`). |
| `RESEARCH_PAGE_SETTLE_SECONDS` | Sleep after Playwright navigation (default `2.0`). |
| `RESEARCH_MIN_CONTENT_LEN` | Treat shorter pages as blocked (default `500`). |
| `RESEARCH_SNIPPET_CHARS` | Page snippet stored in memory (default `4000`). |
| `RESEARCH_MAX_LINKS_PER_PAGE` | Outbound links kept per page (default `80`). |
| `RESEARCH_WORKERS` | Default parallel workers (default `4`). |

### Conviction & frontier

| Variable | Effect |
| --- | --- |
| `RESEARCH_CONVICTION_TARGET` | Stop threshold on overall conviction (default `0.75`). |
| `RESEARCH_COVERAGE_TARGET` | Required facet coverage to be "ready" (default `0.7`). |
| `RESEARCH_PLATEAU_NEW_CLAIMS` | Plateau threshold of new claims/iter (default `2`). |
| `RESEARCH_PLATEAU_WINDOW` | Iterations averaged for plateau (default `3`). |
| `RESEARCH_STAGNATION_ITERATIONS` | Iterations of near-zero progress before creativity boost (default `3`). |
| `RESEARCH_MIN_DOMAINS_READY` | Distinct supporting domains needed (default `3`). |
| `RESEARCH_FRONTIER_TOP_K` | Candidates pushed per discovery pass (default `24`). |
| `RESEARCH_PER_HOST_CAP` | Max candidates per host in any pass (default `4`). |
| `RESEARCH_POP_PER_ITERATION` | URLs fetched per iteration (default `8`). |
| `RESEARCH_RANK_W_SIM` / `..._W_TRUST` / `..._W_NOVELTY` | Ranking weights (default `0.55 / 0.25 / 0.20`). |
| `RESEARCH_LLM_RERANK_TOP_K` | LLM reranks top-K candidates (default `12`). |
| `RESEARCH_DEFAULT_MINUTES` | Default budget when neither `--time` nor `--minutes` given (default `20`). |

### Final report detail

| Variable | Effect |
| --- | --- |
| `RESEARCH_SUMMARY_MAX_CLAIMS` | Claims fed into the summarizer prompt (default `120`). Higher = richer, more detailed reports. |
| `RESEARCH_SUMMARY_MAX_PAGE_EXCERPTS` | Top page excerpts included in the prompt (default `14`). |
| `RESEARCH_SUMMARY_EXCERPT_CHARS` | Characters kept per page excerpt (default `2000`). |
| `RESEARCH_SUMMARY_PROMPT_MAX_CHARS` | Hard cap on the assembled corpus block so it can't overrun the model context (default `90000`). |

### Paths and output layout

| Variable | Effect |
| --- | --- |
| `RESEARCH_MEMORY_ROOT` | Override `memory/` location (intermediate data). |
| `RESEARCH_OUTPUT_ROOT` | Override `output/` location (final HTML reports). |
| `RESEARCH_OUTPUT_LAYOUT` | `flat_by_date` (default — `output/<YYYY-MM-DD>/<topic-slug>--<HHMM-SS>.html`) or `per_run_dir` (legacy — `output/<topic-slug>/<run_id>/index.html`). |

---

## Troubleshooting

- **"No LLM provider configured" / placeholder responses** — set
  `MINIMAX_API_KEY` (or any of the alternative provider keys). The agent
  will run end-to-end without one, but every LLM-driven step (planning,
  ranking, claim extraction, summarization) will return a placeholder and
  the report will be empty.
- **`playwright._impl._errors.Error: Executable doesn't exist...`** — run
  `playwright install chromium` once. Playwright is only used as an
  escalation tier so this only matters when a site needs JS rendering.
- **`sentence-transformers unavailable; using hash fallback`** — first
  import needs network access to download the model. The hash fallback is
  deterministic and keeps the pipeline running, but ranking quality is lower.
- **Sites still blocked after the full ladder** — try lowering parallelism
  (`--workers 2`), increasing pacing (`RESEARCH_PER_HOST_MIN_INTERVAL=4`),
  or shortening the per-URL deadline (`RESEARCH_URL_TOTAL_DEADLINE=8`) so
  one bad URL doesn't burn iteration budget. The Wayback fallback usually
  catches the article eventually.
- **Chromium memory growth on multi-day runs** — lower
  `RESEARCH_BROWSER_RECYCLE_EVERY` (e.g. `25`) or set
  `RESEARCH_BROWSER_DEFAULT=off` to disable Playwright entirely and rely
  solely on the HTTP tier + Jina + Wayback.
- **Search results stop coming in** — DuckDuckGo occasionally rate-limits
  HTML scraping. Adding `RESEARCH_SEARXNG_URL` (self-hosted) or
  `RESEARCH_BRAVE_API_KEY` immediately restores coverage.

---

## Architecture

For the deep dive on how all the pieces fit together, see
[architecture.md](architecture.md).
