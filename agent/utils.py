"""Small pure helpers shared between nodes."""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, List, Optional
from urllib.parse import urldefrag, urlparse, urlunparse


# Windows-only race: os.replace and write_text fail with WinError 5 / 32
# (PermissionError / OSError) when Defender, the Search Indexer, OneDrive, or
# even a transient Python GC handle still has the target file briefly open.
# Linux/macOS do not exhibit this. The fix is a short retry loop with
# exponential backoff that outlives any reasonable AV scan window (~3s total).
_RETRY_SLEEPS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def _is_retryable_oserror(err: BaseException) -> bool:
    """True when the OS error is likely a transient Windows handle-conflict."""
    if not isinstance(err, OSError):
        return False
    # PermissionError covers WinError 5 (ERROR_ACCESS_DENIED).
    # OSError winerror 32 (ERROR_SHARING_VIOLATION) is the other common one.
    if isinstance(err, PermissionError):
        return True
    winerr = getattr(err, "winerror", None)
    if winerr in (5, 32):
        return True
    return False


def retry_on_transient_oserror(func: Callable[[], Any]) -> Any:
    """Run `func()` with retries on transient Windows file-lock errors."""
    last_exc: Optional[BaseException] = None
    for sleep_for in (*_RETRY_SLEEPS, None):
        try:
            return func()
        except OSError as e:  # noqa: BLE001 - intentionally broad
            if not _is_retryable_oserror(e):
                raise
            last_exc = e
            if sleep_for is None:
                break
            time.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    # Unreachable; keep type-checkers happy.
    raise RuntimeError("retry_on_transient_oserror exited without resolution")


def safe_replace(src: "os.PathLike[str] | str", dst: "os.PathLike[str] | str") -> None:
    """`os.replace` with a Windows-safe retry on transient handle conflicts."""
    retry_on_transient_oserror(lambda: os.replace(src, dst))


def safe_write_text(path: "Path | str", content: str, encoding: str = "utf-8") -> None:
    """`Path.write_text` with a Windows-safe retry on transient handle conflicts."""
    p = Path(path)
    retry_on_transient_oserror(lambda: p.write_text(content, encoding=encoding))


def safe_write_bytes(path: "Path | str", content: bytes) -> None:
    """`Path.write_bytes` with a Windows-safe retry on transient handle conflicts."""
    p = Path(path)
    retry_on_transient_oserror(lambda: p.write_bytes(content))


def strip_html(html: str) -> str:
    """Very basic HTML -> text: drop scripts/styles, tags, collapse whitespace."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_urls(text: str) -> List[str]:
    """Extract unique http(s) URLs from a blob of text."""
    urls = re.findall(r"https?://[^\s)>\]\"']+", text)
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,);:]")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response, tolerating fences."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def esc(s: Any) -> str:
    return _html.escape(str(s or ""))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 64) -> str:
    """Lowercase, ascii, hyphen-separated slug with a stable hash suffix."""
    base = (text or "").strip().lower()
    base = _SLUG_RE.sub("-", base).strip("-")
    if not base:
        base = "topic"
    if len(base) > max_len:
        base = base[:max_len].rstrip("-")
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:6]
    return f"{base}-{digest}"


def host_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "ref_url",
    "yclid",
    "_ga",
    "_gl",
}


def canonicalize_url(url: str) -> str:
    """Strip fragments, default ports, and common tracking params."""
    if not url:
        return ""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        url, _ = urldefrag(url)
        p = urlparse(url)
    except Exception:
        return ""
    scheme = (p.scheme or "https").lower()
    netloc = (p.hostname or "").lower()
    if not netloc:
        return ""
    if p.port and not (
        (scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)
    ):
        netloc = f"{netloc}:{p.port}"
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query_pairs = []
    if p.query:
        for kv in p.query.split("&"):
            if not kv:
                continue
            k = kv.split("=", 1)[0]
            if k.lower() in _TRACKING_PARAMS:
                continue
            query_pairs.append(kv)
    query = "&".join(query_pairs)
    return urlunparse((scheme, netloc, path, "", query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha1((url or "").encode("utf-8")).hexdigest()


def shorten(text: str, n: int = 200) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def parse_duration(spec: Optional[str]) -> Optional[int]:
    """Parse a duration spec like '90s', '15m', '2h', '1h30m' -> seconds.

    Returns None if `spec` is falsy. Bare integers are interpreted as seconds.
    """
    if not spec:
        return None
    s = spec.strip().lower()
    if s.isdigit():
        return int(s)
    total = 0
    matched = False
    for value, unit in re.findall(r"(\d+)\s*([smhd])", s):
        matched = True
        v = int(value)
        if unit == "s":
            total += v
        elif unit == "m":
            total += v * 60
        elif unit == "h":
            total += v * 3600
        elif unit == "d":
            total += v * 86400
    if not matched:
        try:
            return int(float(s))
        except ValueError:
            return None
    return total
