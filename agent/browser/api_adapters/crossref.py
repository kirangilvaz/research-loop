"""Crossref adapter.

Handles DOI-style URLs: `doi.org/<doi>`, `dx.doi.org/<doi>`,
`api.crossref.org/works/<doi>`. Fetches a clean metadata blob via the
Crossref REST API. Many publishers paywall the article PDF but the Crossref
metadata (title, abstract when available, authors, publication info) is
free and parseable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote, urlparse

from ..http_fetch import get_with_backoff
from ...utils import strip_html

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


_DOI_RE = re.compile(r"^/?(10\.[^/\s]+/[^?#\s]+)")


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
    if host not in ("doi.org", "dx.doi.org", "api.crossref.org"):
        return None
    path = parsed.path
    if host == "api.crossref.org" and path.startswith("/works/"):
        doi = path[len("/works/"):]
    else:
        m = _DOI_RE.match(path)
        if not m:
            return None
        doi = m.group(1)
    if not doi:
        return None
    doi = doi.split("?", 1)[0].split("#", 1)[0]
    api_url = f"https://api.crossref.org/works/{quote(doi, safe='/')}?mailto=langresearch@example.invalid"
    resp = await get_with_backoff(client, api_url, host="api.crossref.org", deadline=deadline)
    if resp is None or resp.status_code >= 400:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    message = payload.get("message") or {}
    title_list = message.get("title") or []
    title = (title_list[0] if title_list else "").strip()
    if not title:
        return None
    abstract = message.get("abstract") or ""
    if abstract:
        abstract = strip_html(abstract)
    authors = []
    for au in message.get("author") or []:
        name_parts = [au.get("given") or "", au.get("family") or ""]
        full = " ".join(p for p in name_parts if p).strip()
        if full:
            authors.append(full)
    issued = (message.get("issued") or {}).get("date-parts") or []
    year = ""
    if issued and issued[0]:
        year = str(issued[0][0])
    container = (message.get("container-title") or [""])[0]
    publisher = message.get("publisher") or ""
    lines = [f"Title: {title}"]
    if authors:
        lines.append("Authors: " + ", ".join(authors[:12]))
    if container:
        lines.append(f"Venue: {container}")
    if publisher:
        lines.append(f"Publisher: {publisher}")
    if year:
        lines.append(f"Year: {year}")
    lines.append(f"DOI: {doi}")
    if abstract:
        lines.extend(["", "Abstract:", abstract])
    text = "\n".join(lines).strip()
    if len(text) < 80:
        return None
    return AdapterResult(text=text, title=title, links=[], source="crossref")
