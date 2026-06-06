"""Central configuration for the research agent.

All knobs that are reasonable to tune without touching code live here. Values
can be overridden via environment variables so the same code path works in
local dev and in longer headless runs.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_ROOT = Path(os.environ.get("RESEARCH_MEMORY_ROOT", PROJECT_ROOT / "memory"))
OUTPUT_ROOT = Path(os.environ.get("RESEARCH_OUTPUT_ROOT", PROJECT_ROOT / "output"))


LLM_PROVIDER_ENV = "RESEARCH_LLM_PROVIDER"
LLM_DEFAULT_TEMPERATURE = _env_float("RESEARCH_LLM_TEMPERATURE", 0.4)
LLM_TIMEOUT_SECONDS = _env_int("RESEARCH_LLM_TIMEOUT", 180)


EMBEDDING_PROVIDER_ENV = "RESEARCH_EMBEDDING_PROVIDER"
EMBEDDING_LOCAL_MODEL = os.environ.get(
    "RESEARCH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM_FALLBACK = 384


RESPECT_ROBOTS = _env_bool("RESPECT_ROBOTS", True)
PER_HOST_MIN_INTERVAL = _env_float("RESEARCH_PER_HOST_MIN_INTERVAL", 1.5)
PAGE_NAV_TIMEOUT_MS = _env_int("RESEARCH_PAGE_NAV_TIMEOUT_MS", 30000)
PAGE_SETTLE_SECONDS = _env_float("RESEARCH_PAGE_SETTLE_SECONDS", 2.0)
MIN_CONTENT_LEN = _env_int("RESEARCH_MIN_CONTENT_LEN", 500)
SNIPPET_CHARS = _env_int("RESEARCH_SNIPPET_CHARS", 4000)
MAX_LINKS_PER_PAGE = _env_int("RESEARCH_MAX_LINKS_PER_PAGE", 80)
WORKERS_DEFAULT = _env_int("RESEARCH_WORKERS", 4)


URL_TOTAL_DEADLINE = _env_float("RESEARCH_URL_TOTAL_DEADLINE", 12.0)
HTTP_TIMEOUT = _env_float("RESEARCH_HTTP_TIMEOUT", 8.0)
HTTP_PER_HOST_CONCURRENCY = _env_int("RESEARCH_HTTP_PER_HOST_CONCURRENCY", 2)
HTTP_MAX_CONNECTIONS = _env_int("RESEARCH_HTTP_MAX_CONNECTIONS", 32)
BROWSER_DEFAULT = (os.environ.get("RESEARCH_BROWSER_DEFAULT", "escalation_only").strip().lower() or "escalation_only")
BROWSER_POOL_SIZE = _env_int("RESEARCH_BROWSER_POOL_SIZE", 4)
BROWSER_RECYCLE_EVERY = _env_int("RESEARCH_BROWSER_RECYCLE_EVERY", 50)


DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_RESULTS_PER_QUERY = _env_int("RESEARCH_DDG_RESULTS", 12)

SEARXNG_URL = os.environ.get("RESEARCH_SEARXNG_URL", "").strip().rstrip("/")
BRAVE_API_KEY = os.environ.get("RESEARCH_BRAVE_API_KEY", "").strip()


JINA_READER_PREFIX = "https://r.jina.ai/"
WAYBACK_PREFIX = "https://web.archive.org/web/"


OUTPUT_LAYOUT = (os.environ.get("RESEARCH_OUTPUT_LAYOUT", "flat_by_date").strip().lower() or "flat_by_date")


CONVICTION_TARGET = _env_float("RESEARCH_CONVICTION_TARGET", 0.75)
COVERAGE_TARGET = _env_float("RESEARCH_COVERAGE_TARGET", 0.7)
PLATEAU_NEW_CLAIMS_THRESHOLD = _env_int("RESEARCH_PLATEAU_NEW_CLAIMS", 2)
PLATEAU_WINDOW = _env_int("RESEARCH_PLATEAU_WINDOW", 3)
STAGNATION_ITERATIONS = _env_int("RESEARCH_STAGNATION_ITERATIONS", 3)
MIN_DOMAINS_FOR_READY = _env_int("RESEARCH_MIN_DOMAINS_READY", 3)


FRONTIER_PUSH_TOP_K = _env_int("RESEARCH_FRONTIER_TOP_K", 24)
FRONTIER_PER_HOST_CAP = _env_int("RESEARCH_PER_HOST_CAP", 4)
FRONTIER_POP_PER_ITERATION = _env_int("RESEARCH_POP_PER_ITERATION", 8)
RANK_WEIGHT_SIM = _env_float("RESEARCH_RANK_W_SIM", 0.55)
RANK_WEIGHT_TRUST = _env_float("RESEARCH_RANK_W_TRUST", 0.25)
RANK_WEIGHT_NOVELTY = _env_float("RESEARCH_RANK_W_NOVELTY", 0.20)
LLM_RERANK_TOP_K = _env_int("RESEARCH_LLM_RERANK_TOP_K", 12)


CLAIM_DEDUPE_SIM_THRESHOLD = _env_float("RESEARCH_CLAIM_DEDUPE_SIM", 0.86)
CLAIM_CONTRADICTION_SIM_THRESHOLD = _env_float("RESEARCH_CLAIM_CONTRADICT_SIM", 0.78)
MAX_CLAIMS_PER_PAGE = _env_int("RESEARCH_MAX_CLAIMS_PER_PAGE", 8)


# Final summarization detail budget. These control how much of the accumulated
# corpus is fed into the summarizer prompt. Higher values produce richer,
# more detailed reports at the cost of a larger (and slower / costlier) prompt.
# SUMMARY_PROMPT_MAX_CHARS is a hard safety cap applied after assembling the
# claims + excerpt blocks so a huge corpus can't blow past the model context.
SUMMARY_MAX_CLAIMS = _env_int("RESEARCH_SUMMARY_MAX_CLAIMS", 120)
SUMMARY_MAX_PAGE_EXCERPTS = _env_int("RESEARCH_SUMMARY_MAX_PAGE_EXCERPTS", 14)
SUMMARY_EXCERPT_CHARS = _env_int("RESEARCH_SUMMARY_EXCERPT_CHARS", 2000)
SUMMARY_PROMPT_MAX_CHARS = _env_int("RESEARCH_SUMMARY_PROMPT_MAX_CHARS", 90000)


DEFAULT_TIME_BUDGET_MINUTES = _env_int("RESEARCH_DEFAULT_MINUTES", 20)
MAX_ITERATIONS_HARD_CAP = _env_int("RESEARCH_MAX_ITER_HARDCAP", 500)


def ensure_dirs() -> None:
    """Make sure top-level memory and output trees exist."""
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
