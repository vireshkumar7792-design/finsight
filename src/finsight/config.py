"""Central configuration: loads .env via python-dotenv, exposes Settings and get_llm(role).

ZERO-COST POLICY: every default model is free.
  router  -> Groq openai/gpt-oss-20b    supervisor routing (structured output, no tools)
  agent   -> Groq qwen/qwen3.8-27b      specialists (tool calling; parallel calls = fewer requests)
  report  -> Groq openai/gpt-oss-120b   final report (structured output)
Any role can be switched to OpenRouter with FINSIGHT_<ROLE>_PROVIDER=openrouter, but only
":free" model ids are accepted there unless FINSIGHT_ALLOW_PAID=true (the cost guard).

Every model gets a FreeTierRateLimiter sized to its provider's free limits, shared by all roles
that use the same model id (provider limits are per model). Optional dev cache:
FINSIGHT_LLM_CACHE=true stores responses in SQLite so identical re-runs cost nothing.
API keys are SecretStr so they never appear in reprs or logs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from finsight.llm_limits import FreeTierRateLimiter, SQLiteLLMCache, TokenLedger

Role = Literal["router", "agent", "report"]
ROLES: tuple[str, ...] = get_args(Role)
Provider = Literal["groq", "openrouter"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FREE_SUFFIX = ":free"


class MissingAPIKeyError(RuntimeError):
    """A provider key needed for the requested role is not configured."""


class PaidModelRefusedError(RuntimeError):
    """The cost guard refused a model that is not free."""


class Settings(BaseSettings):
    """All tunable values. Env vars use the FINSIGHT_ prefix, except the provider API keys."""

    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", extra="ignore", populate_by_name=True)

    # Provider keys (standard names, no prefix). Never print these.
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")

    # Provider + model per role. Verified free (Nov 2026): Groq free plan; OpenRouter ":free" ids.
    router_provider: Provider = "groq"
    router_model: str = "openai/gpt-oss-20b"
    agent_provider: Provider = "groq"
    agent_model: str = "qwen/qwen3.8-27b"  # preview on Groq; fallback: openai/gpt-oss-120b
    report_provider: Provider = "groq"
    report_model: str = "openai/gpt-oss-120b"

    # Cost guard: refuse non-":free" OpenRouter models unless explicitly allowed.
    allow_paid: bool = False

    # Free-tier limits (measured from Groq response headers; OpenRouter docs for :free models).
    groq_rpm: int = 30
    groq_tpm: int = 8000
    openrouter_rpm: int = 20
    est_tokens_per_call: int = 2500  # reserve used by the token limiter before a call

    # LLM call behavior
    temperature: float = 0.0  # deterministic-ish: analysis, not creativity
    max_retries: int = 5  # SDK-level exponential backoff; honors 429 Retry-After headers
    request_timeout: float = 90.0  # seconds; generous because the limiter may queue us
    # Cap on output tokens per call. OpenRouter checks you can afford max_tokens BEFORE a request.
    max_output_tokens: int = 2000
    reasoning_effort: str = "low"  # gpt-oss models: fewer hidden reasoning tokens = less TPM used

    # Dev cache + checkpoints (both under gitignored data/ paths)
    llm_cache: bool = False
    llm_cache_path: str = "data/cache/llm_cache.sqlite"
    checkpoint_path: str = "data/checkpoints/finsight.sqlite"

    # Analysis constants
    risk_free_rate: float = 0.04  # annual, as a fraction
    history_period: str = "1y"  # yfinance period for risk history
    benchmark_ticker: str = "SPY"
    var_confidence: float = 0.95

    def provider_for(self, role: Role) -> Provider:
        return getattr(self, f"{role}_provider")

    def model_for(self, role: Role) -> str:
        return getattr(self, f"{role}_model")


@lru_cache
def get_settings() -> Settings:
    """Load .env into os.environ (so LangSmith picks up LANGSMITH_*), then build Settings once."""
    load_dotenv(override=False)  # real env vars win over .env
    return Settings()


def _require(key: SecretStr | None, env_name: str, role: str) -> SecretStr:
    if key is None or not key.get_secret_value().strip():
        raise MissingAPIKeyError(f"{env_name} is not set in .env (needed for role '{role}')")
    return key


def check_cost_guard(provider: Provider, model: str, allow_paid: bool) -> None:
    """Refuse OpenRouter models that aren't ':free' unless paid use is explicitly enabled."""
    if provider == "openrouter" and not model.endswith(FREE_SUFFIX) and not allow_paid:
        raise PaidModelRefusedError(
            f"refusing paid OpenRouter model {model!r}: zero-cost policy. Use a '{FREE_SUFFIX}' "
            "model id, or set FINSIGHT_ALLOW_PAID=true to opt in to paid usage."
        )


# One limiter per (provider, model): roles sharing a model share its quota.
_LIMITERS: dict[tuple[str, str], FreeTierRateLimiter] = {}
_CACHES: dict[str, SQLiteLLMCache] = {}


def get_rate_limiter(provider: Provider, model: str, s: Settings) -> FreeTierRateLimiter:
    key = (provider, model)
    if key not in _LIMITERS:
        if provider == "groq":
            _LIMITERS[key] = FreeTierRateLimiter(s.groq_rpm, s.groq_tpm, s.est_tokens_per_call)
        else:
            _LIMITERS[key] = FreeTierRateLimiter(s.openrouter_rpm, None)  # OpenRouter limits requests only
    return _LIMITERS[key]


def get_cache(s: Settings) -> SQLiteLLMCache | None:
    if not s.llm_cache:
        return None
    if s.llm_cache_path not in _CACHES:
        _CACHES[s.llm_cache_path] = SQLiteLLMCache(s.llm_cache_path)
    return _CACHES[s.llm_cache_path]


def get_llm(role: Role, settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model configured for `role`.

    Raises MissingAPIKeyError if its key is unset, PaidModelRefusedError if the cost guard trips.
    """
    s = settings or get_settings()
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {', '.join(ROLES)}")
    provider, model = s.provider_for(role), s.model_for(role)
    check_cost_guard(provider, model, s.allow_paid)

    limiter = get_rate_limiter(provider, model, s)
    common = dict(
        model=model,
        temperature=s.temperature,
        max_retries=s.max_retries,
        timeout=s.request_timeout,
        max_tokens=s.max_output_tokens,
        rate_limiter=limiter,
        callbacks=[TokenLedger(limiter)],
        cache=get_cache(s),  # None = no caching
    )

    if provider == "groq":
        from langchain_groq import ChatGroq  # imported lazily: only needed for this provider

        extra = {"reasoning_effort": s.reasoning_effort} if model.startswith("openai/gpt-oss") else {}
        return ChatGroq(api_key=_require(s.groq_api_key, "GROQ_API_KEY", role), **common, **extra)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=_require(s.openrouter_api_key, "OPENROUTER_API_KEY", role), base_url=OPENROUTER_BASE_URL, **common
    )
