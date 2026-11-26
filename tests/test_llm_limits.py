"""Rate limiter (fake clock: no real sleeping) and the SQLite LLM cache."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tests.fakes import FakeLLM
from finsight.llm_limits import FreeTierRateLimiter, SQLiteLLMCache, TokenLedger


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def limiter(rpm=2, tpm=None, est=0):
    c = Clock()
    return FreeTierRateLimiter(rpm, tpm, est, clock=c, sleep=c.sleep), c


def test_requests_per_minute():
    lim, c = limiter(rpm=2)
    assert lim.acquire() and lim.acquire()
    assert c.slept == []
    assert lim.acquire()  # 3rd request must wait for the 1st to leave the 60s window
    assert c.slept and c.t == pytest.approx(60.05)


def test_non_blocking_returns_false_without_waiting():
    lim, c = limiter(rpm=1)
    assert lim.acquire(blocking=False)
    assert lim.acquire(blocking=False) is False
    assert c.slept == [] and lim.total_wait_seconds == 0


def test_tokens_per_minute():
    # TPM 8000, reserve 2500 per call. After a 6000-token call, 6000+2500 > 8000 -> wait.
    lim, c = limiter(rpm=100, tpm=8000, est=2500)
    lim.acquire()
    c.t = 10
    lim.record_tokens(6000)
    c.t = 20
    lim.acquire()  # must wait until the 6000-token entry (t=10) expires at t=70
    assert c.t == pytest.approx(70.05)
    assert lim.tokens_in_window() == 0


def test_tokens_fit_without_waiting():
    lim, c = limiter(rpm=100, tpm=8000, est=2500)
    lim.record_tokens(3000)
    assert lim.acquire() and c.slept == []  # 3000 + 2500 <= 8000


def test_token_ledger_reads_usage_from_response():
    lim, _ = limiter(rpm=10, tpm=8000)
    ledger = TokenLedger(lim)
    ledger.on_llm_end(LLMResult(generations=[[]], llm_output={"token_usage": {"total_tokens": 1234}}))
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    ledger.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert lim.tokens_in_window() == 1384


def test_sqlite_cache_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = SQLiteLLMCache(path)
    gen = [ChatGeneration(message=AIMessage(content="cached answer"))]
    assert cache.lookup("p", "llm") is None
    cache.update("p", "llm", gen)
    assert SQLiteLLMCache(path).lookup("p", "llm")[0].message.content == "cached answer"  # survives reopen
    assert cache.lookup("p", "other-llm") is None  # different model/params -> miss
    cache.clear()
    assert len(cache) == 0


def test_cache_hit_skips_the_model(tmp_path):
    cache = SQLiteLLMCache(tmp_path / "c.sqlite")
    llm = FakeLLM(script=[AIMessage(content="first")], cache=cache)
    assert llm.invoke("same question").content == "first"
    assert llm.invoke("same question").content == "first"  # 2nd call served from cache
    assert len(llm._calls) == 1  # model only ran once (a 2nd real call would raise ScriptExhausted)
