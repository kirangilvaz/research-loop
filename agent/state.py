"""Shared mutable state that flows through the agent nodes."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config
from .memory import TopicMemory


@dataclass
class State:
    task: str
    topic_slug: str = ""
    run_id: str = ""

    plan: str = ""
    interpretation: str = ""
    facets: List[str] = field(default_factory=list)
    initial_queries: List[str] = field(default_factory=list)
    presentation_hint: Dict[str, Any] = field(default_factory=dict)

    iteration: int = 0
    deadline_ts: Optional[float] = None
    started_ts: float = field(default_factory=time.time)

    sufficient: bool = False
    confidence: float = 0.0
    missing: List[str] = field(default_factory=list)
    summary_raw: str = ""

    headline: str = ""
    presentation: Dict[str, Any] = field(default_factory=dict)
    report_type: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)

    new_claims_history: List[int] = field(default_factory=list)
    creativity_boosts: int = 0
    stagnant_streak: int = 0
    last_signals: Dict[str, Any] = field(default_factory=dict)

    # Health-monitoring sliding windows. These power `agent/health.py`:
    # `print_health` reads the tails for alert detection. We keep them short
    # because alerts only ever look at the last few iterations.
    conviction_history: List[float] = field(default_factory=list)
    source_mix_history: List[Dict[str, int]] = field(default_factory=list)
    productivity_history: List[Dict[str, int]] = field(default_factory=list)

    queue_pop_size: int = config.FRONTIER_POP_PER_ITERATION

    output_dir: str = ""
    results_html: str = ""

    memory: Optional[TopicMemory] = None

    research_file: str = ""
    sources: List[str] = field(default_factory=list)
    attempted_sources: List[str] = field(default_factory=list)
    successful_sources: List[str] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)

    def time_left(self) -> Optional[float]:
        if self.deadline_ts is None:
            return None
        return max(0.0, self.deadline_ts - time.time())

    def time_elapsed(self) -> float:
        return max(0.0, time.time() - self.started_ts)

    def ensure_run_id(self) -> str:
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        return self.run_id

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "topic_slug": self.topic_slug,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "facets": self.facets,
            "initial_queries": self.initial_queries,
            "missing": self.missing,
            "confidence": self.confidence,
            "sufficient": self.sufficient,
            "report_type": self.report_type,
            "items_count": len(self.items),
            "groups_count": len(self.groups),
            "creativity_boosts": self.creativity_boosts,
            "stagnant_streak": self.stagnant_streak,
            "last_signals": self.last_signals,
            "successful_sources_count": len(self.successful_sources),
            "failed_sources_count": len(self.failed_sources),
            "elapsed_seconds": self.time_elapsed(),
        }
