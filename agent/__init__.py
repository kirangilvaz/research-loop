"""Agent package: nodes, state, and supporting layers wired by research_agent.py."""

from .config import ensure_dirs
from .conviction import ConvictionSignals, assess_conviction, is_stagnating, should_stop
from .discover_node import creativity_boost, discover_sources
from .health import print_health
from .html_node import generate_html
from .memory import TopicMemory
from .plan_node import plan
from .research_node import init_research_file, research
from .state import State
from .summarize_node import summarize
from .synthesize_node import synthesize_knowledge

__all__ = [
    "State",
    "TopicMemory",
    "ConvictionSignals",
    "ensure_dirs",
    "plan",
    "discover_sources",
    "creativity_boost",
    "research",
    "init_research_file",
    "synthesize_knowledge",
    "summarize",
    "generate_html",
    "assess_conviction",
    "should_stop",
    "is_stagnating",
    "print_health",
]
