"""Settings + get_llm. Uses dummy keys; constructing a chat model makes no network call."""

import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from finsight.config import OPENROUTER_BASE_URL, MissingAPIKeyError, Settings, get_llm


def test_router_role_uses_groq(test_settings):
    llm = get_llm("router", test_settings)
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == test_settings.router_model
    assert llm.max_retries == test_settings.max_retries
    assert llm.max_tokens == test_settings.max_output_tokens
    # ChatGroq swaps temperature 0 for 1e-8 (Groq's API treats exactly 0 specially).
    assert llm.temperature == pytest.approx(test_settings.temperature, abs=1e-6)


@pytest.mark.parametrize("role, attr", [("agent", "agent_model"), ("report", "report_model")])
def test_agent_and_report_roles_use_openrouter(test_settings, role, attr):
    llm = get_llm(role, test_settings)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == getattr(test_settings, attr)
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.max_retries == test_settings.max_retries
    assert llm.max_tokens == test_settings.max_output_tokens


def test_model_names_are_configurable():
    s = Settings(groq_api_key="k", openrouter_api_key="k", agent_model="some/other-model")
    assert get_llm("agent", s).model_name == "some/other-model"


@pytest.mark.parametrize(
    "role, missing, env_name",
    [("router", "groq_api_key", "GROQ_API_KEY"), ("agent", "openrouter_api_key", "OPENROUTER_API_KEY")],
)
def test_missing_key_gives_clear_error(role, missing, env_name):
    keys = {"groq_api_key": "k", "openrouter_api_key": "k"} | {missing: None}
    s = Settings(**keys)
    with pytest.raises(MissingAPIKeyError, match=f"{env_name} is not set in .env"):
        get_llm(role, s)


def test_blank_key_counts_as_missing():
    s = Settings(groq_api_key="   ", openrouter_api_key="k")
    with pytest.raises(MissingAPIKeyError):
        get_llm("router", s)


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
