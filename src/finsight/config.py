"""Central configuration: loads .env via python-dotenv, exposes Settings and get_llm(role).

Roles -> providers:
  router         -> Groq (ChatGroq): fast/cheap; supervisor routing, guardrail checks, free testing
  agent, report  -> OpenRouter via ChatOpenAI(base_url=...): stronger models for analysis/writing

Every value can be overridden in .env without code changes, e.g. FINSIGHT_AGENT_MODEL=...
API keys are SecretStr so they never appear in reprs or logs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["router", "agent", "report"]
ROLES: tuple[str, ...] = get_args(Role)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class MissingAPIKeyError(RuntimeError):
    """A provider key needed for the requested role is not configured."""


class Settings(BaseSettings):
    """All tunable values. Env vars use the FINSIGHT_ prefix, except the provider API keys."""

    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", extra="ignore", populate_by_name=True)

    # Provider keys (standard names, no prefix). Never print these.
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")

    # Models per role. Groq: check tool-calling support at console.groq.com/docs/tool-use.
    # OpenRouter: filter openrouter.ai/models?supported_parameters=tools.
    router_model: str = "openai/gpt-oss-20b"
    agent_model: str = "anthropic/claude-haiku-5.5"
    report_model: str = "anthropic/claude-sonnet-5.5"

    # LLM call behavior
    temperature: float = 0.0  # deterministic-ish: analysis, not creativity
    max_retries: int = 5  # SDK-level exponential backoff; honors 429 Retry-After headers
    request_timeout: float = 60.0  # seconds per HTTP request
    # Cap on output tokens per LLM call. Also matters for cost: OpenRouter checks you can afford
    # max_tokens BEFORE running a request; unset, it assumes the model's maximum (e.g. 65k).
    max_output_tokens: int = 2000

    # Analysis constants
    risk_free_rate: float = 0.04  # annual, as a fraction
    history_period: str = "1y"  # yfinance period for risk history
    benchmark_ticker: str = "SPY"
    var_confidence: float = 0.95


@lru_cache
def get_settings() -> Settings:
    """Load .env into os.environ (so LangSmith picks up LANGSMITH_*), then build Settings once."""
    load_dotenv(override=False)  # real env vars win over .env
    return Settings()


def _require(key: SecretStr | None, env_name: str, role: str) -> SecretStr:
    if key is None or not key.get_secret_value().strip():
        raise MissingAPIKeyError(f"{env_name} is not set in .env (needed for role '{role}')")
    return key


def get_llm(role: Role, settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model configured for `role`. Raises MissingAPIKeyError if its key is unset."""
    s = settings or get_settings()
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {', '.join(ROLES)}")

    if role == "router":
        from langchain_groq import ChatGroq  # imported lazily: only needed for this role

        return ChatGroq(
            model=s.router_model,
            api_key=_require(s.groq_api_key, "GROQ_API_KEY", role),
            temperature=s.temperature,
            max_retries=s.max_retries,
            timeout=s.request_timeout,
            max_tokens=s.max_output_tokens,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.agent_model if role == "agent" else s.report_model,
        api_key=_require(s.openrouter_api_key, "OPENROUTER_API_KEY", role),
        base_url=OPENROUTER_BASE_URL,
        temperature=s.temperature,
        max_retries=s.max_retries,
        timeout=s.request_timeout,
        max_tokens=s.max_output_tokens,
    )
