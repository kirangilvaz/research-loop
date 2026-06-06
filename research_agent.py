"""Generic, time-bounded AI research agent.

Pipeline:
    plan
      -> loop (until deadline OR conviction target):
           discover_sources   (DDG search + link harvest + LLM + memory replay)
           research           (parallel stealth fetch with r.jina.ai/Wayback fallback)
           synthesize         (claim extraction, generates next-iteration queries)
           assess_conviction  (coverage / diversity / plateau / contradictions / LLM judge)
           if stagnating:
               creativity_boost  (radically different angles)
      -> summarize             (classifies report_type, optional grouped items)
      -> generate_html         (report-type template or layout; tabs when grouped)

Setup:
    pip install -r requirements.txt
    playwright install chromium

Env (any one is enough):
    MINIMAX_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_HOST

Run:
    python research_agent.py "Compare lithium iron phosphate vs NMC batteries"
    python research_agent.py "High conviction tech stocks" --time 2h
    python research_agent.py "Effect of GLP-1 drugs on cardiovascular outcomes" --minutes 30
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from agent import (
    State,
    TopicMemory,
    assess_conviction,
    creativity_boost,
    discover_sources,
    ensure_dirs,
    generate_html,
    init_research_file,
    is_stagnating,
    plan,
    research,
    should_stop,
    summarize,
    synthesize_knowledge,
)
from agent import config as agent_config
from agent.browser.stealth import HostRateLimiter
from agent.health import print_health
from agent.search import print_backend_self_tests, print_backend_status
from agent.utils import parse_duration, slugify


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generic, time-bounded AI research agent."
    )
    p.add_argument("task", nargs="*", help="Research topic / question.")
    p.add_argument(
        "--time",
        dest="time_spec",
        default=None,
        help="Wall-clock research budget (e.g. 90s, 15m, 2h, 1h30m).",
    )
    p.add_argument(
        "--minutes",
        type=int,
        default=None,
        help="Wall-clock research budget in minutes (alternative to --time).",
    )
    p.add_argument(
        "--topic-slug",
        default=None,
        help="Override the auto-generated memory/output folder name.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=agent_config.WORKERS_DEFAULT,
        help="Parallel browser workers per iteration.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=agent_config.MAX_ITERATIONS_HARD_CAP,
        help="Hard cap on iterations (safety net).",
    )
    p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip the LLM-based conviction judge (faster, cheaper).",
    )
    return p


def _parse_budget(args: argparse.Namespace) -> Optional[int]:
    """Resolve time budget in seconds. Returns None if no budget set."""
    if args.time_spec:
        secs = parse_duration(args.time_spec)
        if secs and secs > 0:
            return secs
    if args.minutes is not None and args.minutes > 0:
        return int(args.minutes * 60)
    if agent_config.DEFAULT_TIME_BUDGET_MINUTES > 0:
        return int(agent_config.DEFAULT_TIME_BUDGET_MINUTES * 60)
    return None


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def run_agent(
    task: str,
    time_budget_seconds: Optional[int] = None,
    topic_slug_override: Optional[str] = None,
    workers: int = agent_config.WORKERS_DEFAULT,
    max_iterations: int = agent_config.MAX_ITERATIONS_HARD_CAP,
    use_llm_judge: bool = True,
) -> State:
    ensure_dirs()
    topic_slug = topic_slug_override or slugify(task)
    print(f"=== Research agent ===")
    print(f"Task     : {task!r}")
    print(f"Slug     : {topic_slug}")
    print(f"Budget   : {('unlimited' if not time_budget_seconds else _format_duration(time_budget_seconds))}")
    print(f"Workers  : {workers}")
    print()
    print_backend_status()
    print()
    print_backend_self_tests()
    print()

    state = State(task=task, topic_slug=topic_slug)
    state.memory = TopicMemory(topic_slug)
    state.memory.init_meta(task)
    state.ensure_run_id()
    if time_budget_seconds and time_budget_seconds > 0:
        state.deadline_ts = state.started_ts + time_budget_seconds

    init_research_file(state)
    plan(state)

    rate_limiter = HostRateLimiter()

    for i in range(1, max_iterations + 1):
        state.iteration = i
        time_left = state.time_left()
        if time_left is not None and time_left <= 0:
            print(f"Deadline reached before iteration {i}; stopping.\n")
            break

        print(f"--- Iteration {i}/{max_iterations} "
              f"(elapsed {_format_duration(state.time_elapsed())}, "
              f"left {('-' if time_left is None else _format_duration(time_left))}) ---")

        discovered = discover_sources(state, None, rate_limiter)
        time_left = state.time_left()
        if time_left is not None and time_left <= 0:
            break

        fetched = research(state, rate_limiter, workers=workers)
        time_left = state.time_left()
        if time_left is not None and time_left <= 0:
            break

        new_claims = synthesize_knowledge(state, fetched)

        signals = assess_conviction(
            task=state.task,
            facets=state.facets,
            memory=state.memory,
            new_claims_history=state.new_claims_history,
            use_llm_judge=use_llm_judge,
        )
        state.last_signals = signals.to_json()
        state.conviction_history.append(float(signals.overall))
        if len(state.conviction_history) > 32:
            state.conviction_history = state.conviction_history[-32:]
        print(
            f"  Conviction: overall={signals.overall:.2f} "
            f"coverage={signals.coverage:.2f} diversity={signals.diversity:.2f} "
            f"plateau={signals.plateau:.2f} contradictions={signals.contradiction:.2f} "
            f"llm_conf={signals.llm_confidence:.2f}"
        )
        if signals.notes:
            print(f"  Judge note: {signals.notes[:200]}")
        print()

        print_health(state)

        if should_stop(signals):
            print(f"Conviction target met "
                  f"(overall={signals.overall:.2f} >= {agent_config.CONVICTION_TARGET}). "
                  f"Stopping research loop.\n")
            state.confidence = signals.overall
            state.sufficient = True
            break

        if is_stagnating(state.new_claims_history):
            state.stagnant_streak += 1
            creativity_boost(state)
        else:
            state.stagnant_streak = 0

        time_left = state.time_left()
        if time_left is not None and time_left <= 0:
            print("Deadline reached after iteration; stopping.\n")
            break

        if state.memory.frontier_size() == 0 and discovered == 0 and not fetched:
            if state.stagnant_streak >= 2:
                print("Frontier exhausted and stagnating; final summary.\n")
                break

    else:
        print("Max iterations reached without confident answer; final summary.\n")

    final_signals = assess_conviction(
        task=state.task,
        facets=state.facets,
        memory=state.memory,
        new_claims_history=state.new_claims_history,
        use_llm_judge=use_llm_judge,
    )
    state.last_signals = final_signals.to_json()
    summarize(state, final_signals, force_ready=True)

    out_path = generate_html(state)

    print(
        f"Done. Output: {out_path}\n"
        f"      Memory: {state.memory.root}\n"
        f"      Run log: {state.research_file}"
    )
    return state


def main(argv: Optional[list] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    task = " ".join(args.task).strip() or "Find high conviction stocks"

    budget = _parse_budget(args)

    try:
        run_agent(
            task=task,
            time_budget_seconds=budget,
            topic_slug_override=args.topic_slug,
            workers=max(1, int(args.workers or agent_config.WORKERS_DEFAULT)),
            max_iterations=max(1, int(args.max_iterations or agent_config.MAX_ITERATIONS_HARD_CAP)),
            use_llm_judge=not args.no_llm_judge,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
