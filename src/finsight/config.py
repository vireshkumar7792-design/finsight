"""Settings loaded from .env, and get_llm(role) which only hands out FREE models."""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# =========================
# Fixed LLM behaviour
# =========================

ROLES = ("router", "agent", "report")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2000
# The SDK retries with backoff and waits for Groq's 429 "Retry-After" header.
# That is what handles the tokens-per-minute limit.
MAX_RETRIES = 5
TIMEOUT_SECONDS = 90

# Free-tier request limits (requests per minute).
GROQ_REQUESTS_PER_MINUTE = 30
OPENROUTER_REQUESTS_PER_MINUTE = 20


# =========================
# Settings
# =========================


class Settings(BaseSettings):
    """Values from .env. FinSight options use the FINSIGHT_ prefix; API keys use their usual names."""

    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", extra="ignore", populate_by_name=True)

    # Secrets: SecretStr hides the value when printed.
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    sec_user_agent: SecretStr | None = Field(default=None, validation_alias="SEC_USER_AGENT")

    # One provider + model per role. All defaults are Groq free tier.
    router_provider: str = "groq"
    router_model: str = "openai/gpt-oss-20b"
    agent_provider: str = "groq"
    agent_model: str = "qwen/qwen3.8-27b"
    report_provider: str = "groq"
    report_model: str = "openai/gpt-oss-120b"

    # Cost guard: paid OpenRouter models are refused unless this is true.
    allow_paid: bool = False

    # Where things are stored on disk (all git-ignored).
    market_cache_dir: str = "data/cache/market"
    tavily_cache_dir: str = "data/cache/tavily"
    checkpoint_path: str = "data/checkpoints/finsight.sqlite"
    chroma_path: str = "data/chroma"
    filings_dir: str = "data/filings"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_settings() -> Settings:
    # load_dotenv also exports LANGSMITH_* so LangSmith tracing turns on.
    load_dotenv()
    return Settings()


# =========================
# Cost guard
# =========================


def check_cost_guard(provider: str, model: str, allow_paid: bool) -> None:
    """Refuse an OpenRouter model that is not ':free', unless paid use is allowed."""
    if provider == "openrouter" and not model.endswith(":free") and not allow_paid:
        raise ValueError(
            f"Refusing paid OpenRouter model '{model}'. Use a ':free' model id "
            "(zero-cost policy), or set FINSIGHT_ALLOW_PAID=true."
        )


# =========================
# Rate limiters
# =========================

# One limiter per model: free limits are per model, so roles on the same model share it.
RATE_LIMITERS: dict[str, InMemoryRateLimiter] = {}


def get_rate_limiter(provider: str, model: str) -> InMemoryRateLimiter:
    if model not in RATE_LIMITERS:
        if provider == "groq":
            requests_per_minute = GROQ_REQUESTS_PER_MINUTE
        else:
            requests_per_minute = OPENROUTER_REQUESTS_PER_MINUTE
        RATE_LIMITERS[model] = InMemoryRateLimiter(
            requests_per_second=requests_per_minute / 60,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return RATE_LIMITERS[model]


# =========================
# get_llm
# =========================


def get_provider_and_model(role: str, settings: Settings) -> tuple[str, str]:
    if role == "router":
        return settings.router_provider, settings.router_model
    if role == "agent":
        return settings.agent_provider, settings.agent_model
    if role == "report":
        return settings.report_provider, settings.report_model
    raise ValueError(f"Unknown role '{role}'. Expected one of: {', '.join(ROLES)}")


def require_key(key: SecretStr | None, env_name: str) -> SecretStr:
    if key is None or not key.get_secret_value().strip():
        raise ValueError(f"{env_name} is not set in .env")
    return key


def get_llm(role: str, settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model for a role ('router', 'agent' or 'report')."""
    settings = settings or get_settings()
    provider, model = get_provider_and_model(role, settings)
    check_cost_guard(provider, model, settings.allow_paid)
    rate_limiter = get_rate_limiter(provider, model)

    if provider == "groq":
        from langchain_groq import ChatGroq

        extra_options = {}
        if model.startswith("openai/gpt-oss"):
            # Less hidden "thinking" = fewer tokens used from the 8K tokens/minute budget.
            extra_options["reasoning_effort"] = "low"
        return ChatGroq(
            model=model,
            api_key=require_key(settings.groq_api_key, "GROQ_API_KEY"),
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
            **extra_options,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=require_key(settings.openrouter_api_key, "OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
        )

    raise ValueError(f"Unknown provider '{provider}'. Expected 'groq' or 'openrouter'.")
