"""OpenAlex adapter.

Recognizes `openalex.org/Wxxxxxxxxxx` and `openalex.org/works/Wxxxxxxxxxx`
URLs and fetches metadata + reconstructed abstract via the OpenAlex API.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Optional
from urllib.parse import urlparse

from ..http_fetch import get_with_backoff

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


_WORK_RE = re.compile(r"/(?:works/)?(W\d{6,})(?:[/?#].*)?$")


def _reconstruct_abstract(idx: Dict[str, list]) -> str:
    """Turn OpenAlex's inverted abstract index back into a readable string."""
    if not idx or not isinstance(idx, dict):
        return ""
    positions: Dict[int, str] = {}
    for word, locs in idx.items():
        if not isinstance(locs, list):
            continue
        for pos in locs:
            try:
                positions[int(pos)] = str(word)
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    n = max(positions) + 1
    parts = [positions.get(i, "") for i in range(n)]
    return " ".join(p for p in parts if p).strip()


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
    if "openalex.org" not in host:
        return None
    m = _WORK_RE.search(parsed.path)
    if not m:
        return None
    work_id = m.group(1)
    api_url = f"https://api.openalex.org/works/{work_id}?mailto=langresearch@example.invalid"
    resp = await get_with_backoff(client, api_url, host="api.openalex.org", deadline=deadline)
    if resp is None or resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    title = (data.get("title") or "").strip()
    if not title:
        return None
    abstract = _reconstruct_abstract(data.get("abstract_inverted_index") or {})
    authors = []
    for au in data.get("authorships") or []:
        name = ((au.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
    venue = ((data.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    year = data.get("publication_year")
    lines = [f"Title: {title}"]
    if authors:
        lines.append("Authors: " + ", ".join(authors[:12]))
    if venue:
        lines.append(f"Venue: {venue}")
    if year:
        lines.append(f"Year: {year}")
    if abstract:
        lines.extend(["", "Abstract:", abstract])
    text = "\n".join(lines).strip()
    if len(text) < 80:
        return None
    return AdapterResult(text=text, title=title, links=[], source="openalex")
