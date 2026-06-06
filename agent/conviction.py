"""Multi-signal conviction model.

Five signals combined into a single 0-1 score plus a `should_stop` flag:
    - coverage:        fraction of facets covered by >=2 distinct domains
    - diversity:       distinct supporting domains, normalized
    - plateau:         new-unique-claim rate over the last K iterations
    - contradiction:   weighted negative
    - llm_confidence:  optional LLM judgement on the running brief

The orchestrator passes `state` and recent history; we update conviction in
place and decide whether to keep going.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import config
from .claims import facet_coverage
from .llm import llm_chat_json
from .memory import TopicMemory
from .utils import shorten


@dataclass
class ConvictionSignals:
    coverage: float = 0.0
    diversity: float = 0.0
    plateau: float = 0.0
    contradiction: float = 0.0
    llm_confidence: float = 0.0
    overall: float = 0.0
    new_claims_recent: int = 0
    distinct_domains: int = 0
    facet_coverage: Dict[str, Dict[str, object]] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "coverage": self.coverage,
            "diversity": self.diversity,
            "plateau": self.plateau,
            "contradiction": self.contradiction,
            "llm_confidence": self.llm_confidence,
            "overall": self.overall,
            "new_claims_recent": self.new_claims_recent,
            "distinct_domains": self.distinct_domains,
            "notes": self.notes,
        }


def _diversity(domains: int) -> float:
    target = max(1, config.MIN_DOMAINS_FOR_READY)
    return max(0.0, min(1.0, domains / float(target * 2)))


def _plateau(history: Sequence[int], window: int = config.PLATEAU_WINDOW) -> float:
    """Returns 1.0 when claim growth has plateaued, 0 when growing healthily."""
    recent = list(history)[-window:]
    if len(recent) < 2:
        return 0.0
    mean_new = statistics.fmean(recent)
    threshold = max(1, config.PLATEAU_NEW_CLAIMS_THRESHOLD)
    if mean_new <= threshold:
        return 1.0
    return max(0.0, 1.0 - (mean_new - threshold) / (threshold * 4.0))


def _contradiction(memory: TopicMemory) -> float:
    claims = memory.all_claims()
    if not claims:
        return 0.0
    contradicts = sum(1 for c in claims if c.get("contradicts"))
    if contradicts == 0:
        return 0.0
    return min(1.0, contradicts / max(8.0, len(claims) / 4.0))


def _llm_judge(
    task: str,
    facets: Sequence[str],
    coverage_summary: Dict[str, Dict[str, object]],
    sample_claims: List[dict],
) -> Optional[Dict[str, object]]:
    if not sample_claims:
        return None
    coverage_lines = []
    for f, info in coverage_summary.items():
        domains = info.get("domains") or []
        coverage_lines.append(
            f"- {f}: {len(info.get('supporting_claims') or [])} claims across {len(domains)} domains"
        )
    sample_lines = [
        f"- {shorten(c.get('text', ''), 200)} [support={c.get('support', 0)}]"
        for c in sample_claims[:24]
    ]
    prompt = (
        "Judge whether the accumulated research is sufficient to answer the task. "
        "Be honest and conservative. Output STRICT JSON only.\n\n"
        f"Task: {task}\n"
        f"Facets: {', '.join(facets) or '(none)'}\n\n"
        "Facet coverage summary:\n"
        + "\n".join(coverage_lines or ["(none)"])
        + "\n\nSample claims:\n"
        + "\n".join(sample_lines or ["(none)"])
        + "\n\nSchema:\n"
        '{ "confidence": 0.0-1.0, "reasoning": "...", '
        '"missing": ["..."] }'
    )
    parsed = llm_chat_json(prompt, temperature=0.2)
    if not isinstance(parsed, dict):
        return None
    return parsed


def assess_conviction(
    task: str,
    facets: Sequence[str],
    memory: TopicMemory,
    new_claims_history: Sequence[int],
    use_llm_judge: bool = True,
) -> ConvictionSignals:
    facets = [f for f in (facets or []) if f]
    coverage_map = facet_coverage(memory, facets) if facets else {}

    if facets:
        covered = 0
        for info in coverage_map.values():
            domains = info.get("domains") or []
            if isinstance(domains, list) and len(domains) >= 2:
                covered += 1
        coverage = covered / len(facets) if facets else 0.0
    else:
        coverage = min(1.0, memory.claim_count() / 12.0)

    domains = memory.unique_supporting_domains()
    diversity = _diversity(len(domains))
    plateau = _plateau(new_claims_history)
    contradiction = _contradiction(memory)

    llm_conf = 0.0
    notes = ""
    judge_payload: Dict[str, object] = {}
    if use_llm_judge:
        sample = sorted(memory.all_claims(), key=lambda c: -int(c.get("support", 0)))[:24]
        result = _llm_judge(task, facets, coverage_map, sample)
        if isinstance(result, dict):
            try:
                llm_conf = float(result.get("confidence", 0.0))
            except Exception:
                llm_conf = 0.0
            llm_conf = max(0.0, min(1.0, llm_conf))
            notes = str(result.get("reasoning", ""))[:400]
            judge_payload = result

    overall = (
        0.35 * coverage
        + 0.20 * diversity
        + 0.15 * plateau
        + 0.30 * llm_conf
        - 0.20 * contradiction
    )
    overall = max(0.0, min(1.0, overall))

    sig = ConvictionSignals(
        coverage=coverage,
        diversity=diversity,
        plateau=plateau,
        contradiction=contradiction,
        llm_confidence=llm_conf,
        overall=overall,
        new_claims_recent=int(sum(list(new_claims_history)[-config.PLATEAU_WINDOW:])),
        distinct_domains=len(domains),
        facet_coverage=coverage_map,
        notes=notes,
    )
    sig._judge = judge_payload  # type: ignore[attr-defined]
    return sig


def should_stop(sig: ConvictionSignals) -> bool:
    if sig.overall >= config.CONVICTION_TARGET and sig.coverage >= config.COVERAGE_TARGET:
        return True
    if (
        sig.plateau >= 0.95
        and sig.coverage >= config.COVERAGE_TARGET
        and sig.distinct_domains >= config.MIN_DOMAINS_FOR_READY
    ):
        return True
    return False


def is_stagnating(history: Sequence[int]) -> bool:
    """True if the last STAGNATION_ITERATIONS produced near-zero new claims."""
    recent = list(history)[-config.STAGNATION_ITERATIONS:]
    if len(recent) < config.STAGNATION_ITERATIONS:
        return False
    return all(n <= 1 for n in recent)
