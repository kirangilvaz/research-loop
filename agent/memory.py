"""Disk-backed memory for the research agent.

Each topic gets its own folder under `memory/<topic_slug>/`:

    meta.json              - task variants, first-seen, last-run
    frontier.jsonl         - candidate URLs awaiting fetch
    visited.jsonl          - urls we've fetched + final status
    pages/<sha1>.json      - {url, fetched_at, title, text, links[]}
    claims.jsonl           - atomic claims with sources/support/contradicts
    domain_trust.json      - host -> {ok, fail, blocked}
    embeddings.npy         - stacked claim embeddings (row order matches index)
    embeddings_index.jsonl - per-row metadata for embeddings.npy
    runs/<run_id>/state.json - per-run snapshot

The store is intentionally simple (JSONL + numpy + json). Cross-process safety
is best-effort; the agent is single-process so we don't lock files.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import config
from .embeddings import cosine_matrix, embed_one
from .utils import (
    canonicalize_url,
    host_of,
    retry_on_transient_oserror,
    safe_replace,
    safe_write_text,
    url_hash,
)


@dataclass
class Candidate:
    url: str
    score: float = 0.0
    source: str = ""
    title: str = ""
    snippet: str = ""
    host: str = ""
    discovered_at: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "url": self.url,
            "score": self.score,
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "host": self.host,
            "discovered_at": self.discovered_at,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        return cls(
            url=d.get("url", ""),
            score=float(d.get("score", 0.0)),
            source=d.get("source", ""),
            title=d.get("title", ""),
            snippet=d.get("snippet", ""),
            host=d.get("host", "") or host_of(d.get("url", "")),
            discovered_at=float(d.get("discovered_at", time.time())),
        )


class TopicMemory:
    """Persistent per-topic memory."""

    def __init__(self, slug: str, root: Optional[Path] = None) -> None:
        self.slug = slug
        self.root = (root or config.MEMORY_ROOT) / slug
        self.pages_dir = self.root / "pages"
        self.runs_dir = self.root / "runs"

        self.meta_path = self.root / "meta.json"
        self.frontier_path = self.root / "frontier.jsonl"
        self.visited_path = self.root / "visited.jsonl"
        self.claims_path = self.root / "claims.jsonl"
        self.domain_trust_path = self.root / "domain_trust.json"
        self.embeddings_path = self.root / "embeddings.npy"
        self.embeddings_index_path = self.root / "embeddings_index.jsonl"

        self._lock = threading.RLock()
        self._frontier: Dict[str, Candidate] = {}
        self._visited: Dict[str, dict] = {}
        self._claims: List[dict] = []
        self._claim_embeddings: List[np.ndarray] = []
        self._embeddings_meta: List[dict] = []
        self._domain_trust: Dict[str, Dict[str, int]] = {}
        self._meta: Dict[str, Any] = {}

        self._ensure_layout()
        self._load()

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _read_jsonl(self, path: Path) -> List[dict]:
        if not path.exists():
            return []
        out: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _append_jsonl(self, path: Path, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"

        def _do_append() -> None:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

        retry_on_transient_oserror(_do_append)

    def _rewrite_jsonl(self, path: Path, records: Iterable[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        materialized = list(records)

        def _write_tmp() -> None:
            with tmp.open("w", encoding="utf-8") as f:
                for rec in materialized:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        retry_on_transient_oserror(_write_tmp)
        safe_replace(tmp, path)

    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._meta = {}
        if self.domain_trust_path.exists():
            try:
                self._domain_trust = json.loads(
                    self.domain_trust_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._domain_trust = {}

        for rec in self._read_jsonl(self.frontier_path):
            cand = Candidate.from_json(rec)
            if cand.url:
                self._frontier[cand.url] = cand

        for rec in self._read_jsonl(self.visited_path):
            url = rec.get("url")
            if url:
                self._visited[url] = rec

        self._claims = self._read_jsonl(self.claims_path)

        if self.embeddings_path.exists() and self.embeddings_index_path.exists():
            try:
                arr = np.load(self.embeddings_path)
                self._claim_embeddings = [np.asarray(row, dtype=np.float32) for row in arr]
            except Exception:
                self._claim_embeddings = []
            self._embeddings_meta = self._read_jsonl(self.embeddings_index_path)

    def init_meta(self, task: str) -> None:
        with self._lock:
            now = datetime.now().isoformat(timespec="seconds")
            if not self._meta:
                self._meta = {
                    "slug": self.slug,
                    "first_seen": now,
                    "task_variants": [task],
                    "last_run": now,
                }
            else:
                variants = self._meta.get("task_variants") or []
                if task and task not in variants:
                    variants.append(task)
                self._meta["task_variants"] = variants
                self._meta["last_run"] = now
            safe_write_text(
                self.meta_path,
                json.dumps(self._meta, indent=2, ensure_ascii=False),
            )

    def begin_run(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def snapshot_run(self, run_id: str, payload: Dict[str, Any]) -> None:
        run_dir = self.begin_run(run_id)
        safe_write_text(
            run_dir / "state.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    def add_candidate(
        self,
        url: str,
        score: float,
        source: str,
        title: str = "",
        snippet: str = "",
    ) -> Optional[Candidate]:
        url = canonicalize_url(url)
        if not url:
            return None
        with self._lock:
            if url in self._visited:
                return None
            existing = self._frontier.get(url)
            cand = Candidate(
                url=url,
                score=score,
                source=source,
                title=title or (existing.title if existing else ""),
                snippet=snippet or (existing.snippet if existing else ""),
                host=host_of(url),
            )
            if existing and existing.score >= score:
                cand.score = existing.score
            self._frontier[url] = cand
            self._append_jsonl(self.frontier_path, cand.to_json())
            return cand

    def has_candidate(self, url: str) -> bool:
        url = canonicalize_url(url)
        with self._lock:
            return url in self._frontier or url in self._visited

    def is_visited(self, url: str) -> bool:
        url = canonicalize_url(url)
        with self._lock:
            return url in self._visited

    def frontier_size(self) -> int:
        with self._lock:
            return len(self._frontier)

    def visited_count(self) -> int:
        with self._lock:
            return len(self._visited)

    def get_candidates(self) -> List[Candidate]:
        with self._lock:
            return list(self._frontier.values())

    def pop_top_n(self, n: int, per_host_cap: int) -> List[Candidate]:
        """Pop highest-scoring candidates, with a per-host cap."""
        with self._lock:
            ordered = sorted(
                self._frontier.values(), key=lambda c: c.score, reverse=True
            )
            picked: List[Candidate] = []
            host_count: Dict[str, int] = {}
            for cand in ordered:
                if len(picked) >= n:
                    break
                if host_count.get(cand.host, 0) >= per_host_cap:
                    continue
                picked.append(cand)
                host_count[cand.host] = host_count.get(cand.host, 0) + 1
            for cand in picked:
                self._frontier.pop(cand.url, None)
            self._rewrite_jsonl(
                self.frontier_path, (c.to_json() for c in self._frontier.values())
            )
            return picked

    def mark_visited(self, url: str, status: str, **extra: Any) -> None:
        url = canonicalize_url(url)
        if not url:
            return
        with self._lock:
            rec = {
                "url": url,
                "status": status,
                "host": host_of(url),
                "visited_at": datetime.now().isoformat(timespec="seconds"),
            }
            rec.update(extra)
            self._visited[url] = rec
            self._append_jsonl(self.visited_path, rec)
            self._frontier.pop(url, None)

    def record_page(
        self,
        url: str,
        title: str,
        text: str,
        links: List[str],
        snippet_chars: int,
    ) -> Path:
        url = canonicalize_url(url)
        digest = url_hash(url)
        path = self.pages_dir / f"{digest}.json"
        snippet = (text or "")[:snippet_chars]
        record = {
            "url": url,
            "host": host_of(url),
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "title": title or "",
            "snippet": snippet,
            "text_len": len(text or ""),
            "links": links[: config.MAX_LINKS_PER_PAGE],
        }
        with self._lock:
            safe_write_text(
                path,
                json.dumps(record, ensure_ascii=False, indent=2),
            )
        return path

    def read_page(self, url: str) -> Optional[dict]:
        url = canonicalize_url(url)
        if not url:
            return None
        path = self.pages_dir / f"{url_hash(url)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def all_pages(self) -> List[dict]:
        out: List[dict] = []
        if not self.pages_dir.exists():
            return out
        for p in self.pages_dir.glob("*.json"):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def successful_urls(self) -> List[str]:
        with self._lock:
            return [
                rec["url"]
                for rec in self._visited.values()
                if rec.get("status") == "ok"
            ]

    def failed_urls(self) -> List[str]:
        with self._lock:
            return [
                rec["url"]
                for rec in self._visited.values()
                if rec.get("status") not in ("ok",)
            ]

    _TIER_NAMES = ("api", "http", "browser", "jina", "wayback")

    def _ensure_domain_record(self, host: str) -> Dict[str, Any]:
        """Get or create a domain record with the full schema."""
        d = self._domain_trust.setdefault(host, {})
        d.setdefault("ok", 0)
        d.setdefault("fail", 0)
        d.setdefault("blocked", 0)
        tier_success = d.setdefault("tier_success", {})
        tier_attempts = d.setdefault("tier_attempts", {})
        for tier in self._TIER_NAMES:
            tier_success.setdefault(tier, 0)
            tier_attempts.setdefault(tier, 0)
        d.setdefault("preferred_tier", None)
        return d

    def _persist_domain_trust(self) -> None:
        safe_write_text(
            self.domain_trust_path,
            json.dumps(self._domain_trust, indent=2),
        )

    def bump_domain(self, host: str, kind: str) -> None:
        if not host:
            return
        with self._lock:
            d = self._ensure_domain_record(host)
            if kind in ("ok", "fail", "blocked"):
                d[kind] = int(d.get(kind, 0)) + 1
            self._persist_domain_trust()

    def record_tier(self, host: str, tier: str, success: bool) -> None:
        """Record that `tier` was attempted on `host`, optionally succeeded.

        On success, recompute `preferred_tier` as the tier with the highest
        success ratio (min 2 attempts), preferring tiers earlier in the ladder
        on ties so we don't lock into the heaviest tier prematurely.
        """
        if not host or tier not in self._TIER_NAMES:
            return
        with self._lock:
            d = self._ensure_domain_record(host)
            d["tier_attempts"][tier] = int(d["tier_attempts"].get(tier, 0)) + 1
            if success:
                d["tier_success"][tier] = int(d["tier_success"].get(tier, 0)) + 1

            best_tier: Optional[str] = None
            best_ratio = -1.0
            for t in self._TIER_NAMES:
                attempts = int(d["tier_attempts"].get(t, 0))
                successes = int(d["tier_success"].get(t, 0))
                if attempts < 2:
                    continue
                ratio = successes / attempts
                if ratio > best_ratio + 1e-9:
                    best_ratio = ratio
                    best_tier = t
            if best_tier is not None and best_ratio >= 0.5:
                d["preferred_tier"] = best_tier
            self._persist_domain_trust()

    def preferred_tier(self, host: str) -> Optional[str]:
        """Return the tier that has historically worked best for this host."""
        if not host:
            return None
        with self._lock:
            d = self._domain_trust.get(host)
            if not d:
                return None
            tier = d.get("preferred_tier")
            return tier if tier in self._TIER_NAMES else None

    def domain_trust_score(self, host: str) -> float:
        with self._lock:
            d = self._domain_trust.get(host)
            if not d:
                return 0.0
            ok = int(d.get("ok", 0))
            fail = int(d.get("fail", 0)) + int(d.get("blocked", 0))
            total = ok + fail
            if total == 0:
                return 0.0
            return ok / total

    def replay_good_urls(self, min_score: float = 0.5) -> List[str]:
        """Return previously-visited OK URLs from trusted domains for warm-start."""
        with self._lock:
            urls = []
            for rec in self._visited.values():
                if rec.get("status") != "ok":
                    continue
                host = rec.get("host") or host_of(rec.get("url", ""))
                if self.domain_trust_score(host) >= min_score:
                    urls.append(rec["url"])
            return urls

    def add_claim(
        self,
        text: str,
        source_url: str,
        embedding: Optional[np.ndarray] = None,
        quote: str = "",
    ) -> Tuple[dict, str]:
        """Add a claim, deduping near-matches. Returns (claim_record, action).

        action is one of: 'added', 'merged', 'contradicts'.
        """
        text = (text or "").strip()
        if not text:
            return {}, "skipped"
        if embedding is None:
            embedding = embed_one(text)

        with self._lock:
            if self._claim_embeddings:
                mat = np.vstack(self._claim_embeddings)
                sims = cosine_matrix(embedding, mat)
                if sims.size > 0:
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= config.CLAIM_DEDUPE_SIM_THRESHOLD:
                        existing = self._claims[best_idx]
                        sources = list(existing.get("sources") or [])
                        if source_url and source_url not in sources:
                            sources.append(source_url)
                        existing["sources"] = sources
                        existing["support"] = len(sources)
                        existing["last_seen"] = datetime.now().isoformat(timespec="seconds")
                        self._rewrite_jsonl(self.claims_path, self._claims)
                        return existing, "merged"

            claim_id = f"c{len(self._claims):06d}"
            record = {
                "id": claim_id,
                "text": text,
                "quote": quote[:300],
                "sources": [source_url] if source_url else [],
                "support": 1 if source_url else 0,
                "contradicts": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "last_seen": datetime.now().isoformat(timespec="seconds"),
            }
            self._claims.append(record)
            self._claim_embeddings.append(np.asarray(embedding, dtype=np.float32))
            self._embeddings_meta.append({"id": claim_id, "kind": "claim"})
            self._append_jsonl(self.claims_path, record)
            self._save_embeddings()
            return record, "added"

    def _save_embeddings(self) -> None:
        if not self._claim_embeddings:
            return
        try:
            arr = np.vstack(self._claim_embeddings)
            retry_on_transient_oserror(
                lambda: np.save(self.embeddings_path, arr)
            )
            self._rewrite_jsonl(self.embeddings_index_path, self._embeddings_meta)
        except Exception as e:  # noqa: BLE001
            print(f"  [memory] could not persist embeddings: {e}")

    def all_claims(self) -> List[dict]:
        with self._lock:
            return list(self._claims)

    def claim_count(self) -> int:
        with self._lock:
            return len(self._claims)

    def unique_supporting_domains(self) -> List[str]:
        with self._lock:
            domains = set()
            for c in self._claims:
                for u in c.get("sources") or []:
                    h = host_of(u)
                    if h:
                        domains.add(h)
            return sorted(domains)

    def domain_trust_table(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {h: dict(v) for h, v in self._domain_trust.items()}
