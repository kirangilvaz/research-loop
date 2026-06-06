"""Pluggable LLM layer.

The agent uses one provider for all chat-style calls. Provider is auto-detected
from environment variables in priority order, but can be forced with
`RESEARCH_LLM_PROVIDER`.

Supported providers:
    minimax   - MINIMAX_API_KEY (https://api.minimax.io)
    openai    - OPENAI_API_KEY  (https://api.openai.com)
    anthropic - ANTHROPIC_API_KEY
    ollama    - OLLAMA_HOST (default http://localhost:11434)

If none are configured, returns a clearly-labelled placeholder string so the
rest of the pipeline can still execute end-to-end during dry runs.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import requests

from . import config


_PLACEHOLDER_PREFIX = "[placeholder response"
_ERROR_PREFIX = "[error calling LLM"


def _detect_provider() -> Optional[str]:
    forced = os.environ.get(config.LLM_PROVIDER_ENV, "").strip().lower()
    if forced:
        return forced
    if os.environ.get("MINIMAX_API_KEY", "").strip():
        return "minimax"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.environ.get("OLLAMA_HOST", "").strip():
        return "ollama"
    return None


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "..."


def is_placeholder(text: str) -> bool:
    if not text:
        return True
    head = text.lstrip()[:40].lower()
    return head.startswith(_PLACEHOLDER_PREFIX) or head.startswith(_ERROR_PREFIX)


def _call_minimax(prompt: str, system: str, temperature: float) -> str:
    key = os.environ["MINIMAX_API_KEY"].strip()
    payload = {
        "model": os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post(
        "https://api.minimax.io/v1/text/chatcompletion_v2",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return "[empty response]"
    return ((choices[0].get("message") or {}).get("content") or "").strip()


def _call_openai(prompt: str, system: str, temperature: float) -> str:
    key = os.environ["OPENAI_API_KEY"].strip()
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return "[empty response]"
    return ((choices[0].get("message") or {}).get("content") or "").strip()


def _call_anthropic(prompt: str, system: str, temperature: float) -> str:
    key = os.environ["ANTHROPIC_API_KEY"].strip()
    payload = {
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        "max_tokens": 2048,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    parts: List[str] = []
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts).strip() or "[empty response]"


def _call_ollama(prompt: str, system: str, temperature: float) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(
        f"{host}/api/chat",
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    msg = (data.get("message") or {}).get("content")
    if msg:
        return msg.strip()
    return (data.get("response") or "").strip() or "[empty response]"


_DISPATCH = {
    "minimax": _call_minimax,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "ollama": _call_ollama,
}


def llm_chat(
    prompt: str,
    system: str = "You are a meticulous research assistant.",
    temperature: Optional[float] = None,
) -> str:
    """Single entry point for chat-style LLM calls."""
    provider = _detect_provider()
    if not provider:
        return f"{_PLACEHOLDER_PREFIX} (no LLM provider configured): {_truncate(prompt)}]"

    fn = _DISPATCH.get(provider)
    if fn is None:
        return f"{_PLACEHOLDER_PREFIX} (unknown provider {provider!r})]"

    temp = config.LLM_DEFAULT_TEMPERATURE if temperature is None else temperature
    try:
        return fn(prompt, system, temp)
    except Exception as e:  # noqa: BLE001 - surface error to caller as text
        return f"{_ERROR_PREFIX} via {provider}: {e}]"


def minimax_chat(prompt: str, system: str = "You are a concise research assistant.", temperature: float = 0.4) -> str:
    """Backwards-compatible alias for code that hasn't migrated yet."""
    return llm_chat(prompt, system=system, temperature=temperature)


def llm_chat_json(
    prompt: str,
    system: str = "You output strict JSON only with no commentary.",
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """Convenience for callers expecting JSON; returns None on failure."""
    raw = llm_chat(prompt, system=system, temperature=temperature)
    if is_placeholder(raw):
        return None
    from .utils import extract_json

    parsed = extract_json(raw)
    return parsed if isinstance(parsed, (dict, list)) else None
