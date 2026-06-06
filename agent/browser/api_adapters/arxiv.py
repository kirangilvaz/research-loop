"""arXiv adapter.

Recognizes `arxiv.org/abs/<id>` and `arxiv.org/pdf/<id>` URLs and fetches the
paper metadata + abstract from the arXiv export API (Atom). The PDF body
itself is not downloaded - the abstract is plenty for claim extraction and
keeps the agent off heavy binary endpoints.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

from ..http_fetch import get_with_backoff

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


_ID_RE = re.compile(r"^/(abs|pdf)/([^/]+?)(?:v\d+)?(?:\.pdf)?/?$")
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


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
    if not (host == "arxiv.org" or host.endswith(".arxiv.org")):
        return None
    m = _ID_RE.match(parsed.path)
    if not m:
        return None
    arxiv_id = m.group(2)
    if not arxiv_id:
        return None
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = await get_with_backoff(client, api_url, host="export.arxiv.org", deadline=deadline)
    if resp is None or resp.status_code >= 400 or not resp.text:
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
    except Exception:
        return None
    entry = root.find("atom:entry", _NS)
    if entry is None:
        return None
    title_el = entry.find("atom:title", _NS)
    summary_el = entry.find("atom:summary", _NS)
    authors = [
        (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
        for a in entry.findall("atom:author", _NS)
    ]
    authors = [a for a in authors if a]
    title = (title_el.text or "").strip() if title_el is not None else f"arXiv:{arxiv_id}"
    summary = (summary_el.text or "").strip() if summary_el is not None else ""
    if not summary:
        return None
    published_el = entry.find("atom:published", _NS)
    published = (published_el.text or "").strip() if published_el is not None else ""
    body_lines = [f"Title: {title}"]
    if authors:
        body_lines.append("Authors: " + ", ".join(authors))
    if published:
        body_lines.append(f"Published: {published}")
    body_lines.extend(["", "Abstract:", summary])
    return AdapterResult(
        text="\n".join(body_lines).strip(),
        title=title,
        links=[],
        source="arxiv",
    )
