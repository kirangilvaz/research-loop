"""Content extraction + quality gate.

The new pipeline prefers Trafilatura for article-class content and falls back
to readability-lxml, then to a plain HTML strip. A separate `quality_ok` check
decides whether the extracted text is good enough to keep or whether the
fetcher should escalate to the next tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set
from urllib.parse import urljoin

from .. import config
from ..utils import canonicalize_url, host_of, strip_html


_BAD_LINK_PREFIXES = ("javascript:", "mailto:", "tel:", "#")

_BLOCK_MARKERS = (
    "captcha",
    "verify you are human",
    "are you a robot",
    "access denied",
    "unusual traffic",
    "enable javascript and cookies",
    "request blocked",
    "checking your browser",
    "please enable javascript",
)


@dataclass
class ContentExtraction:
    title: str = ""
    text: str = ""
    links: List[str] = field(default_factory=list)
    extractor: str = ""


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+")


def _extract_title(html: str) -> str:
    m = _TITLE_RE.search(html or "")
    if not m:
        return ""
    return strip_html(m.group(1))[:300]


def _trafilatura_extract(html: str, base_url: str) -> Optional[str]:
    if not html:
        return None
    try:
        import trafilatura  # type: ignore

        text = trafilatura.extract(
            html,
            url=base_url,
            favor_recall=True,
            include_links=False,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
        if text and text.strip():
            return text.strip()
        return None
    except Exception:
        return None


def _readability_extract(html: str) -> str:
    try:
        from readability import Document  # type: ignore

        doc = Document(html)
        cleaned_html = doc.summary(html_partial=True)
        title = doc.short_title() or ""
        text = strip_html(cleaned_html)
        return text if not title else f"{title}\n\n{text}"
    except Exception:
        return ""


def _extract_links_lxml(html: str, base_url: str) -> List[str]:
    try:
        from lxml import html as lhtml  # type: ignore

        tree = lhtml.fromstring(html)
        urls: List[str] = []
        seen: Set[str] = set()
        base_host = host_of(base_url)
        for a in tree.xpath("//a[@href]"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(_BAD_LINK_PREFIXES):
                continue
            absolute = urljoin(base_url, href)
            canonical = canonicalize_url(absolute)
            if not canonical or canonical in seen:
                continue
            host = host_of(canonical)
            if not host:
                continue
            if host == base_host and len(canonical) <= len(base_url) + 1:
                continue
            seen.add(canonical)
            urls.append(canonical)
            if len(urls) >= config.MAX_LINKS_PER_PAGE * 3:
                break
        return urls
    except Exception:
        return _extract_links_regex(html, base_url)


def _extract_links_regex(html: str, base_url: str) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for href in _HREF_RE.findall(html or ""):
        href = href.strip()
        if not href or href.startswith(_BAD_LINK_PREFIXES):
            continue
        absolute = urljoin(base_url, href)
        canonical = canonicalize_url(absolute)
        if not canonical or canonical in seen:
            continue
        if not host_of(canonical):
            continue
        seen.add(canonical)
        out.append(canonical)
        if len(out) >= config.MAX_LINKS_PER_PAGE * 3:
            break
    return out


def extract_content(html: str, base_url: str) -> ContentExtraction:
    """Run the extraction ladder: Trafilatura -> readability-lxml -> strip_html."""
    if not html:
        return ContentExtraction()

    title = _extract_title(html)
    text = ""
    extractor = ""

    t = _trafilatura_extract(html, base_url)
    if t:
        text = t
        extractor = "trafilatura"
    else:
        r = _readability_extract(html)
        if r and len(r) >= 200:
            text = r
            extractor = "readability"

    if not text or len(text) < config.MIN_CONTENT_LEN:
        fallback = strip_html(html)
        if len(fallback) > len(text):
            text = fallback
            extractor = extractor or "strip_html"

    links = _extract_links_lxml(html, base_url)
    return ContentExtraction(title=title, text=text, links=links, extractor=extractor)


def quality_ok(text: str, *, min_len: Optional[int] = None) -> bool:
    """Decide whether extracted text looks like real prose worth keeping.

    Combines three checks:
      - non-empty and >= min_len characters
      - no anti-bot block-marker substring in the first 400 chars
      - prose-ish heuristic: enough sentences AND high alpha-density
    """
    if not text:
        return False
    threshold = config.MIN_CONTENT_LEN if min_len is None else int(min_len)
    if len(text) < threshold:
        return False
    head = text[:400].lower()
    if any(marker in head for marker in _BLOCK_MARKERS):
        return False
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < 3:
        return False
    alpha = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    if alpha / max(1, len(text)) < 0.65:
        return False
    return True
