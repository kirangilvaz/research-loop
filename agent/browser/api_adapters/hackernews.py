"""Hacker News adapter.

Recognizes `news.ycombinator.com/item?id=<n>` URLs and pulls the story +
its comment tree from the public HN Firebase API.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import parse_qs, urlparse

from ..http_fetch import get_with_backoff
from ...utils import strip_html

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


_API_HOST = "hacker-news.firebaseio.com"
_API_BASE = f"https://{_API_HOST}/v0/item"
_MAX_COMMENTS = 30


async def _fetch_item(client: "HttpClient", item_id: int, deadline: Optional[float]) -> Optional[dict]:
    url = f"{_API_BASE}/{item_id}.json"
    resp = await get_with_backoff(client, url, host=_API_HOST, deadline=deadline)
    if resp is None or resp.status_code >= 400:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _flatten_text(item: dict) -> str:
    raw = item.get("text") or ""
    if not raw:
        return ""
    return strip_html(raw)


async def try_handle(
    url: str,
    client: "HttpClient",
    deadline: Optional[float] = None,
) -> Optional["AdapterResult"]:
    from . import AdapterResult

    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host not in ("news.ycombinator.com", "hckrnews.com"):
        return None
    if not parsed.path.startswith("/item"):
        return None
    qs = parse_qs(parsed.query)
    raw_id = (qs.get("id") or [""])[0]
    try:
        item_id = int(raw_id)
    except ValueError:
        return None

    root = await _fetch_item(client, item_id, deadline)
    if not root:
        return None

    title = (root.get("title") or "").strip()
    by = (root.get("by") or "").strip()
    story_url = (root.get("url") or "").strip()
    body = _flatten_text(root)

    kids = root.get("kids") or []
    kids = kids[:_MAX_COMMENTS]
    comments: List[str] = []
    if kids:
        tasks = [_fetch_item(client, kid, deadline) for kid in kids]
        for fetched in await asyncio.gather(*tasks, return_exceptions=False):
            if not fetched:
                continue
            if fetched.get("deleted") or fetched.get("dead"):
                continue
            text = _flatten_text(fetched)
            if text:
                comments.append(text)

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if by:
        parts.append(f"Submitter: {by}")
    if story_url:
        parts.append(f"Story URL: {story_url}")
    if body:
        parts.extend(["", "Submitter text:", body])
    if comments:
        parts.extend(["", "Top comments:"])
        for i, c in enumerate(comments, start=1):
            parts.append(f"[{i}] {c}")
    text = "\n".join(parts).strip()
    if len(text) < 80:
        return None
    return AdapterResult(
        text=text,
        title=title or f"HN item {item_id}",
        links=[story_url] if story_url else [],
        source="hackernews",
    )
