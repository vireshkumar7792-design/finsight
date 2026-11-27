"""Fakes for offline tests: a scripted chat model and a word-hash embedder. Nothing costs money."""

import hashlib
import math
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class FakeLLM(BaseChatModel):
    """script: answers for normal calls (AIMessage or an Exception to raise).
    structured: answers for with_structured_output calls (dict, Pydantic object, or Exception).
    An unscripted call fails the test, so tests also prove how many LLM calls were made.
    """

    script: list = []
    structured: list = []
    calls: int = 0
    structured_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls >= len(self.script):
            raise AssertionError(f"unscripted LLM call #{self.calls + 1}")
        answer = self.script[self.calls]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return ChatResult(generations=[ChatGeneration(message=answer)])

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(self.next_structured_answer)

    def next_structured_answer(self, messages):
        if self.structured_calls >= len(self.structured):
            raise AssertionError(f"unscripted structured call #{self.structured_calls + 1}")
        answer = self.structured[self.structured_calls]
        self.structured_calls += 1
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def total_calls(self) -> int:
        return self.calls + self.structured_calls


def tool_calls_message(*calls) -> AIMessage:
    """An AI message that asks for tools, e.g. tool_calls_message(("get_risk_metrics", {}))."""
    tool_calls = []
    for name, args in calls:
        tool_calls.append({"name": name, "args": args, "id": f"call_{name}"})
    return AIMessage(content="", tool_calls=tool_calls)


class HashEmbedder:
    """A fake embedder: texts that share words get similar vectors. No model download."""

    name = "hash-test-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]
