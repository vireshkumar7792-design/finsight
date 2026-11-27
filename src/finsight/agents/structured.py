"""Helpers for reading LLM output, shared by all agents.

structured_invoke: free models are less reliable at structured output, so every failure mode
(provider error, invalid JSON, schema mismatch, empty result) becomes one exception type the
caller can catch and fall back from.
message_text: the plain text of an AI message.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def message_text(msg: AIMessage) -> str:
    """Final text of an AI message (content may be a string or a list of content blocks)."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


class StructuredOutputError(RuntimeError):
    """The model did not return a valid instance of the requested schema."""


def structured_invoke(llm: BaseChatModel, schema: type[T], messages: list[AnyMessage]) -> T:
    """Ask `llm` for a `schema` instance using strict JSON-schema mode (Groq/OpenAI-style).

    include_raw=True makes LangChain return parse errors instead of raising mid-chain, so we can
    report them clearly.
    """
    runnable = llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
    try:
        out = runnable.invoke(messages)
    except Exception as e:  # provider/HTTP errors, including a 400 for schema-violating output
        raise StructuredOutputError(f"{type(e).__name__}: {e}") from e
    if out.get("parsing_error") is not None:
        raise StructuredOutputError(f"parse error: {out['parsing_error']}")
    parsed = out.get("parsed")
    if parsed is None:
        raise StructuredOutputError("model returned no structured output")
    try:
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    except ValidationError as e:
        raise StructuredOutputError(f"schema validation failed: {e}") from e
