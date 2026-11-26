"""Free-tier plumbing for LLM calls: a request+token rate limiter and an on-disk response cache.

Why a custom limiter? LangChain's InMemoryRateLimiter only counts requests. On Groq's free tier
the binding limit is usually TOKENS per minute (8K), not requests (30/min). So we keep a
sliding 60-second window of both, and wait before a call that would break either limit.
The provider SDK's retry-on-429 (honoring Retry-After) stays as a second safety net.

How it plugs in: BaseChatModel calls `rate_limiter.acquire()` before every (non-cached) request,
and `TokenLedger` (a callback) records the tokens each response actually used.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.rate_limiters import BaseRateLimiter

WINDOW_SECONDS = 60.0

# Only these classes may be rebuilt from the cache file (it could be tampered with on disk).
_CACHE_ALLOWED = [Generation, ChatGeneration, AIMessage]


class FreeTierRateLimiter(BaseRateLimiter):
    """Sliding-window limiter on requests/minute AND tokens/minute (shared per model)."""

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int | None = None,
        est_tokens_per_call: int = 2000,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.est_tokens = est_tokens_per_call
        self._clock, self._sleep = clock, sleep
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self.total_wait_seconds = 0.0

    # --- bookkeeping -------------------------------------------------------------------

    def record_tokens(self, n: int) -> None:
        with self._lock:
            self._tokens.append((self._clock(), int(n)))

    def _prune(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= WINDOW_SECONDS:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0][0] >= WINDOW_SECONDS:
            self._tokens.popleft()

    def tokens_in_window(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return sum(n for _, n in self._tokens)

    def _wait_needed(self, now: float) -> float:
        """Seconds to wait before a new request fits; 0 if it fits now."""
        self._prune(now)
        waits = []
        if len(self._requests) >= self.rpm:
            waits.append(WINDOW_SECONDS - (now - self._requests[0]))
        if self.tpm is not None and self._tokens:
            used = sum(n for _, n in self._tokens)
            if used + self.est_tokens > self.tpm:
                # Wait until enough old entries expire to make room for the estimate.
                need, acc = used + self.est_tokens - self.tpm, 0
                for ts, n in self._tokens:
                    acc += n
                    if acc >= need:
                        waits.append(WINDOW_SECONDS - (now - ts))
                        break
        return max([0.0, *waits])

    # --- BaseRateLimiter API -------------------------------------------------------------

    def _try_acquire(self) -> float:
        """Take a slot and return 0, or return how many seconds to wait (no slot taken)."""
        with self._lock:
            now = self._clock()
            wait = self._wait_needed(now)
            if wait <= 0:
                self._requests.append(now)
                return 0.0
        return min(wait + 0.05, WINDOW_SECONDS + 1)  # small margin past the window edge

    def acquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._try_acquire()) > 0:
            if not blocking:
                return False
            self.total_wait_seconds += wait
            self._sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._try_acquire()) > 0:
            if not blocking:
                return False
            self.total_wait_seconds += wait
            await asyncio.sleep(wait)
        return True


def _total_tokens(response: LLMResult) -> int:
    """Read total tokens from a response (provider metadata, else message usage_metadata)."""
    usage = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage") or {}
    if usage.get("total_tokens"):
        return int(usage["total_tokens"])
    total = 0
    for gens in response.generations:
        for g in gens:
            meta = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
            total += int(meta.get("total_tokens", 0))
    return total


class TokenLedger(BaseCallbackHandler):
    """Callback that feeds actual token usage back into the limiter after each LLM call."""

    def __init__(self, limiter: FreeTierRateLimiter):
        self.limiter = limiter

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        n = _total_tokens(response)
        if n:
            self.limiter.record_tokens(n)


class SQLiteLLMCache(BaseCache):
    """Exact-match LLM response cache in a local SQLite file (development only).

    Key = (prompt, llm_string). llm_string encodes model name + parameters, so changing the
    model or temperature naturally misses the cache. Re-running an identical question costs
    zero quota. Values are serialized with LangChain's own dumps/loads.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (prompt TEXT, llm TEXT, value TEXT, PRIMARY KEY (prompt, llm))"
            )
            self._conn.commit()

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM llm_cache WHERE prompt = ? AND llm = ?", (prompt, llm_string)
            ).fetchone()
        return loads(row[0], allowed_objects=_CACHE_ALLOWED) if row else None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (prompt, llm_string, dumps(list(return_val)))
            )
            self._conn.commit()

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
