"""Test fakes (no API calls, no cost): scripted chat model and a deterministic embedder."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, PrivateAttr, ValidationError


class ScriptExhausted(AssertionError):
    """The graph made more LLM calls than the test scripted."""


class FakeLLM(BaseChatModel):
    """Plays back scripted responses and records every prompt it receives.

    script:     items returned by normal calls (tool-calling loops, plain-text fallback):
                AIMessage, or an Exception to raise.
    structured: items returned by with_structured_output(...) calls: a pydantic instance,
                a dict (validated against the schema; invalid -> parsing_error), or an Exception.
    """

    script: list[Any] = []
    structured: list[Any] = []
    _calls: list = PrivateAttr(default_factory=list)
    _structured_calls: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls.append(list(messages))
        i = len(self._calls) - 1
        if i >= len(self.script):
            raise ScriptExhausted(f"unscripted LLM call #{i + 1}")
        item = self.script[i]
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])

    def with_structured_output(self, schema, **kwargs):
        def run(messages):
            self._structured_calls.append(list(messages))
            i = len(self._structured_calls) - 1
            if i >= len(self.structured):
                raise ScriptExhausted(f"unscripted structured call #{i + 1}")
            item = self.structured[i]
            if isinstance(item, Exception):
                raise item
            raw = AIMessage(content=str(item))
            if isinstance(item, BaseModel):
                return {"raw": raw, "parsed": item, "parsing_error": None}
            try:
                return {"raw": raw, "parsed": schema.model_validate(item), "parsing_error": None}
            except ValidationError as e:
                return {"raw": raw, "parsed": None, "parsing_error": e}

        return RunnableLambda(run)

    @property
    def n_calls(self) -> int:
        return len(self._calls) + len(self._structured_calls)


class HashEmbedder:
    """Deterministic bag-of-words embedding for offline tests: texts sharing words are close.

    Not semantic (no synonyms), but it behaves like a real embedder for ranking tests.
    Same two methods as finsight.rag.embeddings.LocalEmbedder.
    """

    name = "hash-test-embedder"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def tool_call(name: str, args: dict | None = None, id: str | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": id or f"call_{name}"}


def ai_tools(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))
