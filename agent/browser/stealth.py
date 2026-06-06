"""Async stealth Playwright browser with a context pool.

The browser is the escalation tier of the fetch ladder, not the default. We
lazily start it the first time a caller requests a context, and we maintain a
small pool of stealth contexts (default 4) that are reused across pages and
periodically recycled (default every 50 pages) to prevent Chromium memory
bloat on multi-day runs.

Each context still uses randomized UA / viewport / locale / timezone +
`playwright-stealth` patches (when available) + an init script that hides
common automation markers. The pool gives us those benefits without paying
context-creation cost per URL.

Compatibility shim: a tiny `HostRateLimiter` + `RobotsCache` stay here so
the rest of the codebase (and prior memory snapshots) keep importing the
same symbols.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
import urllib.robotparser
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional

from .. import config
from ..utils import host_of


_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

_VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1680, "height": 1050},
    {"width": 1920, "height": 1080},
]

_LOCALES = ["en-US", "en-GB", "en-CA", "en-AU"]
_TIMEZONES = [
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "Europe/London",
    "Europe/Berlin",
]

_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = window.chrome || { runtime: {} };
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
  window.navigator.permissions.query = (parameters) =>
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters);
}
"""


class HostRateLimiter:
    """Token-bucket-ish per-host pacing: minimum interval between hits."""

    def __init__(self, min_interval: float = config.PER_HOST_MIN_INTERVAL) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        if not host:
            return
        with self._lock:
            last = self._last.get(host, 0.0)
            now = time.monotonic()
            wait_for = self.min_interval - (now - last)
            if wait_for > 0:
                time.sleep(wait_for + random.uniform(0, 0.4))
            self._last[host] = time.monotonic()

    async def wait_async(self, host: str) -> None:
        if not host:
            return
        last = self._last.get(host, 0.0)
        now = time.monotonic()
        wait_for = self.min_interval - (now - last)
        if wait_for > 0:
            await asyncio.sleep(wait_for + random.uniform(0, 0.4))
        self._last[host] = time.monotonic()


class RobotsCache:
    """Lightweight robots.txt cache with fail-open semantics."""

    def __init__(self, user_agent: str = "*", respect: bool = config.RESPECT_ROBOTS) -> None:
        self.user_agent = user_agent
        self.respect = respect
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._lock = threading.Lock()

    def _get_or_load(self, host: str) -> Optional[urllib.robotparser.RobotFileParser]:
        with self._lock:
            rp = self._cache.get(host)
            if rp is not None:
                return rp
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
            except Exception:
                return None
            self._cache[host] = rp
            return rp

    def allowed(self, url: str) -> bool:
        if not self.respect:
            return True
        host = host_of(url)
        if not host:
            return True
        rp = self._get_or_load(host)
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def allowed_async(self, url: str) -> bool:
        if not self.respect:
            return True
        return await asyncio.to_thread(self.allowed, url)


class _PooledContext:
    """A reusable stealth context with a recycling counter."""

    __slots__ = ("ctx", "pages_used")

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.pages_used = 0


class AsyncStealthBrowser:
    """Async Playwright owner with a small pool of pre-warmed stealth contexts."""

    def __init__(
        self,
        headless: bool = True,
        pool_size: int = config.BROWSER_POOL_SIZE,
        recycle_every: int = config.BROWSER_RECYCLE_EVERY,
    ) -> None:
        self.headless = headless
        self.pool_size = max(1, int(pool_size))
        self.recycle_every = max(5, int(recycle_every))
        self._pw = None
        self._browser = None
        self._pool: List[_PooledContext] = []
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--no-sandbox",
                ],
            )
            self._started = True

    async def close(self) -> None:
        async with self._lock:
            for pooled in self._pool:
                try:
                    await pooled.ctx.close()
                except Exception:
                    pass
            self._pool.clear()
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._pw is not None:
                try:
                    await self._pw.stop()
                except Exception:
                    pass
                self._pw = None
            self._started = False

    async def _new_context(self) -> _PooledContext:
        ctx = await self._browser.new_context(
            user_agent=random.choice(_USER_AGENTS),
            viewport=random.choice(_VIEWPORTS),
            locale=random.choice(_LOCALES),
            timezone_id=random.choice(_TIMEZONES),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        await ctx.add_init_script(_INIT_SCRIPT)
        try:
            from playwright_stealth import stealth_async  # type: ignore

            async def _stealth_apply(page) -> None:
                try:
                    await stealth_async(page)
                except Exception:
                    pass

            ctx.on("page", lambda page: asyncio.create_task(_stealth_apply(page)))
        except Exception:
            pass
        ctx.set_default_navigation_timeout(config.PAGE_NAV_TIMEOUT_MS)
        ctx.set_default_timeout(config.PAGE_NAV_TIMEOUT_MS)
        return _PooledContext(ctx)

    @asynccontextmanager
    async def acquire_context(self) -> AsyncIterator["object"]:
        """Lease a stealth context from the pool.

        The context is automatically returned to the pool, and recycled if it
        has served too many pages. Callers should `await ctx.new_page()` and
        close the page when done, but they should NOT close the context.
        """
        await self.start()
        pooled: Optional[_PooledContext] = None
        async with self._lock:
            if self._pool:
                pooled = self._pool.pop()
        if pooled is None:
            pooled = await self._new_context()
        try:
            yield pooled.ctx
        finally:
            pooled.pages_used += 1
            should_recycle = pooled.pages_used >= self.recycle_every
            if should_recycle:
                try:
                    await pooled.ctx.close()
                except Exception:
                    pass
            else:
                async with self._lock:
                    if len(self._pool) < self.pool_size:
                        self._pool.append(pooled)
                    else:
                        try:
                            await pooled.ctx.close()
                        except Exception:
                            pass

    @property
    def started(self) -> bool:
        return self._started


# Backwards-compatible alias so older imports keep working.
StealthBrowser = AsyncStealthBrowser


async def humanize(page) -> None:
    """Tiny mouse jitter + scroll to look more human."""
    try:
        for _ in range(random.randint(1, 3)):
            x = random.randint(20, 800)
            y = random.randint(20, 600)
            await page.mouse.move(x, y, steps=random.randint(2, 8))
        await page.evaluate(
            "() => window.scrollTo({ top: Math.random() * 600, behavior: 'smooth' })"
        )
    except Exception:
        pass


async def settle_async(min_seconds: Optional[float] = None) -> None:
    base = config.PAGE_SETTLE_SECONDS if min_seconds is None else float(min_seconds)
    await asyncio.sleep(max(0.0, base + random.uniform(0, 0.6)))


def settle(min_seconds: Optional[float] = None) -> None:
    base = config.PAGE_SETTLE_SECONDS if min_seconds is None else float(min_seconds)
    time.sleep(max(0.0, base + random.uniform(0, 0.6)))
