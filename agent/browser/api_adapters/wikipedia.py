"""Wikipedia adapter.

For any `*.wikipedia.org/wiki/<Title>` URL, fetch a clean plain-text extract
via the MediaWiki REST API instead of scraping the article HTML.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional
from urllib.parse import unquote, urlparse

from ..http_fetch import get_with_backoff

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


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
    if not host.endswith("wikipedia.org"):
        return None
    if not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path[len("/wiki/"):]).split("#", 1)[0].strip()
    if not title or title.startswith(("Special:", "File:", "Talk:", "Help:")):
        return None
    lang = host.split(".", 1)[0]
    if not lang or lang == "wikipedia":
        lang = "en"

    api_url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=extracts|info"
        "&explaintext=1&exsectionformat=plain&inprop=url&redirects=1"
        f"&titles={title}"
    )
    resp = await get_with_backoff(client, api_url, host=f"{lang}.wikipedia.org", deadline=deadline)
    if resp is None or resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    pages = (data.get("query") or {}).get("pages") or {}
    if not pages:
        return None
    page = next(iter(pages.values()))
    extract = (page.get("extract") or "").strip()
    if not extract:
        return None
    return AdapterResult(
        text=extract,
        title=page.get("title") or title.replace("_", " "),
        links=[],
        source="wikipedia",
    )
