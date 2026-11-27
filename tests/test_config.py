"""get_llm and the cost guard. No real API calls: building a model object makes no request."""

import pytest

from finsight.config import Settings, check_cost_guard, get_llm


def test_cost_guard_allows_free_and_groq_models():
    check_cost_guard("openrouter", "some/model:free", allow_paid=False)
    check_cost_guard("groq", "openai/gpt-oss-20b", allow_paid=False)


def test_cost_guard_refuses_paid_openrouter_models():
    with pytest.raises(ValueError, match="Refusing paid OpenRouter model"):
        check_cost_guard("openrouter", "some/paid-model", allow_paid=False)
    # Only an explicit opt-in lets it through.
    check_cost_guard("openrouter", "some/paid-model", allow_paid=True)


def test_get_llm_refuses_paid_model_from_settings(test_settings):
    test_settings.agent_provider = "openrouter"
    test_settings.agent_model = "some/paid-model"
    with pytest.raises(ValueError, match="Refusing paid"):
        get_llm("agent", test_settings)


def test_get_llm_builds_each_role(test_settings):
    for role in ["router", "agent", "report"]:
        llm = get_llm(role, test_settings)
        assert llm.max_retries == 5
        assert llm.rate_limiter is not None


def test_roles_on_the_same_model_share_one_rate_limiter(test_settings):
    test_settings.report_model = test_settings.router_model
    assert get_llm("router", test_settings).rate_limiter is get_llm("report", test_settings).rate_limiter


def test_missing_key_and_unknown_role():
    settings = Settings(groq_api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        get_llm("router", settings)
    with pytest.raises(ValueError, match="Unknown role"):
        get_llm("boss", settings)


def test_keys_are_hidden_when_printed(test_settings):
    assert "test-groq" not in repr(test_settings)
