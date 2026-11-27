"""Settings + get_llm + cost guard. Dummy keys; constructing a chat model makes no network call."""

import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from finsight.config import (
    MAX_RETRIES,
    OPENROUTER_BASE_URL,
    OPENROUTER_RPM,
    TEMPERATURE,
    MissingAPIKeyError,
    PaidModelRefusedError,
    Settings,
    check_cost_guard,
    get_llm,
)
from finsight.llm_limits import FreeTierRateLimiter, SQLiteLLMCache, TokenLedger


def keys(**over):
    return Settings(**({"groq_api_key": "k", "openrouter_api_key": "k"} | over))


# ---- zero-cost defaults ------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["router", "agent", "report"])
def test_all_roles_default_to_groq_free_tier(test_settings, role):
    llm = get_llm(role, test_settings)
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == test_settings.model_for(role)
    assert llm.max_retries == MAX_RETRIES
    assert llm.max_tokens == test_settings.max_output_tokens
    # ChatGroq swaps temperature 0 for 1e-8 (Groq's API treats exactly 0 specially).
    assert llm.temperature == pytest.approx(TEMPERATURE, abs=1e-6)


def test_default_models():
    s = Settings(groq_api_key=None, openrouter_api_key=None)
    assert (s.router_model, s.agent_model, s.report_model) == (
        "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b")
    assert {s.router_provider, s.agent_provider, s.report_provider} == {"groq"}
    assert s.allow_paid is False


def test_reasoning_effort_only_for_gpt_oss(test_settings):
    assert get_llm("router", test_settings).reasoning_effort == "low"  # gpt-oss
    assert get_llm("agent", test_settings).reasoning_effort is None  # qwen


def test_rate_limiter_attached_and_shared_per_model():
    s = keys(router_model="same/model", report_model="same/model")
    r, rep, agent = get_llm("router", s), get_llm("report", s), get_llm("agent", s)
    assert isinstance(r.rate_limiter, FreeTierRateLimiter)
    assert r.rate_limiter is rep.rate_limiter  # same model -> same quota
    assert agent.rate_limiter is not r.rate_limiter
    assert (r.rate_limiter.rpm, r.rate_limiter.tpm) == (s.groq_rpm, s.groq_tpm)
    assert any(isinstance(cb, TokenLedger) for cb in r.callbacks)


def test_cache_only_when_enabled(tmp_path):
    assert get_llm("router", keys()).cache is None
    s = keys(llm_cache=True, llm_cache_path=str(tmp_path / "c.sqlite"))
    assert isinstance(get_llm("router", s).cache, SQLiteLLMCache)


# ---- cost guard ---------------------------------------------------------------------------------


def test_paid_openrouter_model_refused_without_opt_in():
    s = keys(agent_provider="openrouter", agent_model="anthropic/claude-haiku-5.5")
    with pytest.raises(PaidModelRefusedError, match="FINSIGHT_ALLOW_PAID"):
        get_llm("agent", s)


def test_free_openrouter_model_allowed():
    s = keys(agent_provider="openrouter", agent_model="google/gemma-4-31b-it:free")
    llm = get_llm("agent", s)
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.rate_limiter.rpm == OPENROUTER_RPM


def test_paid_allowed_only_with_explicit_flag(monkeypatch):
    monkeypatch.setenv("FINSIGHT_ALLOW_PAID", "true")
    s = Settings(groq_api_key="k", openrouter_api_key="k", agent_provider="openrouter", agent_model="paid/model")
    assert s.allow_paid is True
    assert get_llm("agent", s).model_name == "paid/model"


def test_cost_guard_ignores_groq():
    check_cost_guard("groq", "openai/gpt-oss-120b", allow_paid=False)  # Groq free plan: no-op


# ---- keys, roles, settings --------------------------------------------------------------------


@pytest.mark.parametrize(
    "role, provider_field, missing, env_name",
    [("router", "router_provider", "groq_api_key", "GROQ_API_KEY"),
     ("agent", "agent_provider", "openrouter_api_key", "OPENROUTER_API_KEY")],
)
def test_missing_key_gives_clear_error(role, provider_field, missing, env_name):
    provider = "groq" if missing == "groq_api_key" else "openrouter"
    s = keys(**{missing: None, provider_field: provider, "agent_model": "x/y:free"})
    with pytest.raises(MissingAPIKeyError, match=f"{env_name} is not set in .env"):
        get_llm(role, s)


def test_blank_key_counts_as_missing():
    with pytest.raises(MissingAPIKeyError):
        get_llm("router", keys(groq_api_key="   "))


def test_unknown_role(test_settings):
    with pytest.raises(ValueError, match="unknown role"):
        get_llm("boss", test_settings)


def test_secrets_never_appear_in_repr(test_settings):
    text = repr(test_settings) + str(test_settings) + test_settings.model_dump_json()
    assert "test-groq" not in text and "test-or" not in text


def test_analysis_defaults():
    s = Settings(groq_api_key=None, openrouter_api_key=None)
    assert (s.risk_free_rate, s.history_period, s.benchmark_ticker, s.var_confidence) == (0.04, "1y", "SPY", 0.95)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("FINSIGHT_BENCHMARK_TICKER", "QQQ")
    monkeypatch.setenv("FINSIGHT_RISK_FREE_RATE", "0.05")
    s = Settings()
    assert s.benchmark_ticker == "QQQ" and s.risk_free_rate == 0.05
