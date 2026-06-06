"""Async httpx client wrapper used by Tier 1 (and tier 3/4 helpers).

Provides:
    - `HttpClient`: a thin context-managed wrapper around `httpx.AsyncClient`
      with realistic browser headers and global + per-host semaphores.
    - `polite_headers(extra=...)`: realistic Chrome-on-macOS-ish headers
      that work for most servers including ones that 403 a default `python-httpx` UA.
    - `get_with_backoff(client, url, ...)`: retries on 429/5xx with jittered
      exponential backoff, respecting `Retry-After`.

The client is reused across an entire research iteration (one per
`asyncio.run`), so HTTP/2 connection pooling and TLS sessions are amortized
across hundreds of URLs in the same loop.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from typing import Any, Dict, Optional

import httpx

from .. import config


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def polite_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {
        "User-Agent": _BROWSER_UA,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
    if extra:
        h.update(extra)
    return h


class HttpClient:
    """Async httpx client with global + per-host concurrency caps.

    Usage::

        async with HttpClient() as client:
            r = await client.get(url)
    """

    def __init__(
        self,
        *,
        max_connections: int = config.HTTP_MAX_CONNECTIONS,
        per_host: int = config.HTTP_PER_HOST_CONCURRENCY,
        timeout: float = config.HTTP_TIMEOUT,
    ) -> None:
        self._max_connections = max_connections
        self._per_host = max(1, per_host)
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._global_sem = asyncio.Semaphore(max_connections)
        self._host_sems: Dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self._per_host)
        )

    async def __aenter__(self) -> "HttpClient":
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=max(4, self._max_connections // 2),
        )
        try:
            self._client = httpx.AsyncClient(
                http2=True,
                follow_redirects=True,
                limits=limits,
                timeout=httpx.Timeout(self._timeout, connect=min(5.0, self._timeout)),
                headers=polite_headers(),
            )
        except Exception:
            self._client = httpx.AsyncClient(
                http2=False,
                follow_redirects=True,
                limits=limits,
                timeout=httpx.Timeout(self._timeout, connect=min(5.0, self._timeout)),
                headers=polite_headers(),
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _host_sem(self, host: str) -> asyncio.Semaphore:
        return self._host_sems[host or ""]

    @property
    def raw(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpClient used outside its async context")
        return self._client

    async def get(
        self,
        url: str,
        *,
        host: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
    ) -> httpx.Response:
        """Issue a GET with global + per-host semaphore protection.

        `deadline` is an absolute seconds value (asyncio time) after which the
        request must abort. If set and smaller than the configured timeout,
        the request's effective timeout is reduced to match.
        """
        if self._client is None:
            raise RuntimeError("HttpClient used outside its async context")
        eff_timeout = timeout or self._timeout
        if deadline is not None:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"deadline exceeded before GET {url}")
            eff_timeout = max(0.5, min(eff_timeout, remaining))
        async with self._global_sem, self._host_sem(host):
            return await self._client.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(eff_timeout, connect=min(5.0, eff_timeout)),
            )


async def get_with_backoff(
    client: HttpClient,
    url: str,
    *,
    host: str = "",
    max_attempts: int = 2,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> Optional[httpx.Response]:
    """GET with limited jittered backoff on 429/5xx. Returns None if all attempts fail.

    `deadline` (absolute event-loop time) is checked before each attempt and
    used to clip the per-request timeout. Once the deadline is in the past we
    bail without further retries.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.get(
                url,
                host=host,
                headers=headers,
                timeout=timeout,
                deadline=deadline,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - we surface up after retries
            last_exc = e
            if attempt >= max_attempts:
                break
            await _backoff_sleep(attempt, deadline=deadline)
            continue

        if resp.status_code < 400:
            return resp
        if resp.status_code in (429,) or 500 <= resp.status_code < 600:
            if attempt >= max_attempts:
                return resp
            retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            await _backoff_sleep(attempt, retry_after=retry_after, deadline=deadline)
            continue
        # 4xx other than 429 -- not worth retrying
        return resp
    if last_exc is not None:
        # bubble the last exception up via None + caller-side logging
        return None
    return None


def _parse_retry_after(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        secs = float(raw)
        return max(0.0, min(secs, 30.0))
    except ValueError:
        return None


async def _backoff_sleep(
    attempt: int,
    *,
    retry_after: Optional[float] = None,
    deadline: Optional[float] = None,
) -> None:
    base = retry_after if retry_after is not None else min(0.5 * (2 ** (attempt - 1)), 4.0)
    jitter = random.uniform(0.0, 0.5)
    nap = base + jitter
    if deadline is not None:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            return
        nap = min(nap, max(0.0, remaining - 0.1))
    if nap > 0:
        await asyncio.sleep(nap)
