"""RSS/Atom feed + sitemap.xml adapter.

If a URL ends in `.rss`, `.atom`, `.xml`, contains `/rss/` or `/feed/` in the
path, or is a `sitemap*.xml` resource, we parse it with `feedparser` (for
feeds) or as a sitemap to collect the entry/loc URLs. The "text" returned is
a compact list of entry titles + summaries; the `links` list contains
discovered URLs which the orchestrator can feed into its frontier on the
next discover pass.

Adapter is deliberately conservative: it requires a strong content-type or
suffix signal to fire, so generic HTML URLs that happen to contain `/feed/`
in their path still fall through to Tier 1.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import urlparse

from ..http_fetch import get_with_backoff
from ...utils import canonicalize_url, strip_html

if TYPE_CHECKING:
    from . import AdapterResult
    from ..http_fetch import HttpClient


_FEED_SUFFIX_RE = re.compile(r"\.(rss|atom)(\?|$)", re.I)
_FEED_HINT_RE = re.compile(r"/(rss|feed|feeds|atom)(/|$)", re.I)
_SITEMAP_RE = re.compile(r"sitemap[^/]*\.xml(\?|$)", re.I)


def _looks_like_feed_url(url: str) -> bool:
    if _FEED_SUFFIX_RE.search(url):
        return True
    if _FEED_HINT_RE.search(url):
        return True
    return False


def _looks_like_sitemap_url(url: str) -> bool:
    return bool(_SITEMAP_RE.search(url))


def _content_type_is_feed(ctype: str) -> bool:
    ctype = (ctype or "").lower()
    return (
        "application/rss" in ctype
        or "application/atom" in ctype
        or "application/xml" in ctype
        or "text/xml" in ctype
    )


def _parse_sitemap(xml_text: str) -> List[str]:
    locs = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml_text or "", flags=re.I)
    out: List[str] = []
    seen = set()
    for raw in locs:
        cu = canonicalize_url(raw.strip())
        if not cu or cu in seen:
            continue
        seen.add(cu)
        out.append(cu)
        if len(out) >= 200:
            break
    return out


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
    if not host:
        return None

    is_sitemap = _looks_like_sitemap_url(url)
    is_feed_hint = _looks_like_feed_url(url)
    if not (is_sitemap or is_feed_hint):
        return None

    resp = await get_with_backoff(client, url, host=host, deadline=deadline)
    if resp is None or resp.status_code >= 400 or not resp.text:
        return None
    ctype = resp.headers.get("content-type", "")

    if is_sitemap:
        locs = _parse_sitemap(resp.text)
        if not locs:
            return None
        body = "Sitemap entries:\n" + "\n".join(f"- {u}" for u in locs[:60])
        return AdapterResult(text=body, title=f"sitemap {host}", links=locs, source="sitemap")

    if not (_content_type_is_feed(ctype) or "<rss" in resp.text[:512].lower() or "<feed" in resp.text[:512].lower()):
        return None

    try:
        import feedparser  # type: ignore

        feed = feedparser.parse(resp.text)
    except Exception:
        return None
    entries = list(getattr(feed, "entries", []) or [])
    if not entries:
        return None
    title = (getattr(feed.feed, "title", None) or host) if getattr(feed, "feed", None) else host
    parts: List[str] = [f"Feed: {title}", ""]
    links: List[str] = []
    seen = set()
    for entry in entries[:30]:
        e_title = (entry.get("title") or "").strip()
        e_link = (entry.get("link") or "").strip()
        e_summary = entry.get("summary") or entry.get("description") or ""
        e_summary = strip_html(e_summary)[:500]
        published = entry.get("published") or entry.get("updated") or ""
        if e_title:
            parts.append(f"- {e_title}" + (f"  ({published})" if published else ""))
            if e_summary:
                parts.append(f"  {e_summary}")
        if e_link:
            cu = canonicalize_url(e_link)
            if cu and cu not in seen:
                seen.add(cu)
                links.append(cu)
    body = "\n".join(parts).strip()
    if len(body) < 80:
        return None
    return AdapterResult(text=body, title=str(title)[:200], links=links, source="rss")
