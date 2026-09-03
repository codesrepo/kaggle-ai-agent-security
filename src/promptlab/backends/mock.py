"""A deterministic, offline backend.

This exists so the whole harness can be exercised — and its statistics validated —
without a model, a GPU, or a network. It is also the recommended way to write a test
for your own objective before you spend GPU hours on it.

The mock is *rule-driven*: you give it rules that map a prompt to a scripted response
and a decode-token count. That makes it possible to reproduce, deterministically, the
two failure modes that a token-only metric reports as wins:

    - the "cheap trap": a variant that decodes FEWER tokens but never calls the tool,
    - the "runaway":    a variant that fires but decodes an order of magnitude more.

`build_demo_backend()` ships both, and `tests/` asserts the screener catches them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import Completion, Message, ToolCall

__all__ = ["Rule", "MockBackend", "build_demo_backend"]

Matcher = str | re.Pattern[str] | Callable[[str], bool]


def _as_predicate(m: Matcher) -> Callable[[str], bool]:
    if callable(m) and not isinstance(m, (str, re.Pattern)):
        return m
    if isinstance(m, re.Pattern):
        return lambda s: bool(m.search(s))
    return lambda s: m in s


@dataclass
class Rule:
    """Fires when `match` matches the rendered prompt.

    `fire_rate` in [0, 1] is applied *deterministically* — the decision is a hash of
    the prompt plus the repetition index, so a variant that fails, fails reproducibly
    at the same rate across runs. That is what makes mock-backed tests stable.
    """

    match: Matcher
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    completion_tokens: int = 20
    fire_rate: float = 1.0
    # tokens decoded on the runs where the rule does NOT fire (usually cheaper!)
    miss_completion_tokens: int | None = None
    miss_text: str = "ok"

    def __post_init__(self) -> None:
        self._pred = _as_predicate(self.match)

    def matches(self, prompt: str) -> bool:
        return self._pred(prompt)

    def fires(self, prompt: str, rep: int) -> bool:
        if self.fire_rate >= 1.0:
            return True
        if self.fire_rate <= 0.0:
            return False
        h = hashlib.blake2b(f"{prompt}\x00{rep}".encode(), digest_size=8).digest()
        return (int.from_bytes(h, "big") % 10_000) < int(self.fire_rate * 10_000)


class MockBackend:
    """Deterministic scriptable backend. No network, no model, no randomness."""

    def __init__(
        self,
        rules: Sequence[Rule] | None = None,
        default: Rule | None = None,
        name: str = "mock",
        chars_per_token: float = 4.0,
    ) -> None:
        self.name = name
        self.rules = list(rules or [])
        self.default = default or Rule(match=lambda _s: True, text="", completion_tokens=8)
        self.chars_per_token = chars_per_token
        self.calls = 0
        self._rep_counter: dict[str, int] = {}

    # -- Backend protocol ------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        self.calls += 1
        prompt = "\n".join(m.content for m in messages)
        rep = kwargs.get("rep")
        if rep is None:
            rep = self._rep_counter.get(prompt, 0)
            self._rep_counter[prompt] = rep + 1

        rule = next((r for r in self.rules if r.matches(prompt)), self.default)
        if rule.fires(prompt, int(rep)):
            return Completion(
                text=rule.text,
                tool_calls=list(rule.tool_calls),
                prompt_tokens=self.count_tokens(prompt),
                completion_tokens=rule.completion_tokens,
                hops=1,
            )
        miss = rule.miss_completion_tokens
        return Completion(
            text=rule.miss_text,
            tool_calls=[],
            prompt_tokens=self.count_tokens(prompt),
            completion_tokens=miss if miss is not None else max(1, rule.completion_tokens // 8),
            hops=1,
        )

    # -- tokenizer -------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.token_ids(text))

    def token_ids(self, text: str) -> list[int]:
        """A crude but *stable* word/punctuation tokenizer.

        It is not any real BPE. It exists so `promptlab.tokens` can be unit-tested:
        the vocabulary below deliberately contains multi-character entries so that
        'some strings are one token and some are not' is reproducible offline.
        """
        vocab = self._vocab_set()
        out: list[int] = []
        i = 0
        while i < len(text):
            for length in (6, 5, 4, 3, 2):
                chunk = text[i : i + length]
                if chunk in vocab:
                    out.append(hash(chunk) & 0xFFFFFF)
                    i += length
                    break
            else:
                out.append(hash(text[i]) & 0xFFFFFF)
                i += 1
        return out

    _VOCAB: tuple[str, ...] = (
        # multi-char entries: make some fragments single-token and others not
        "cale", "cam", "bab", "zoq", "kot", "ping", "note", "memo",
        ".com", ".net", ".org", "@get", "@app", "@bab",
        "Subject", "Body", "Reply", "only", "tool", "args",
    )

    def _vocab_set(self) -> frozenset[str]:
        if not hasattr(self, "_vocab_cache"):
            self._vocab_cache = frozenset(self._VOCAB)
        return self._vocab_cache

    def vocab(self) -> Iterable[str]:
        return tuple(self._VOCAB)


def build_demo_backend(tool_name: str = "notify") -> MockBackend:
    """A backend that reproduces the real traps, for tests and for the tutorial.

    The scenario is benign: we want the assistant to call a `notify` tool with a
    well-formed `to` argument. Some phrasings do that reliably and cheaply; some are
    cheaper but stop calling the tool; one runs away.
    """

    def call(to: str = "user") -> list[ToolCall]:
        return [ToolCall(name=tool_name, arguments={"to": to, "subject": "ok", "body": "ok"})]

    return MockBackend(
        name="demo",
        rules=[
            # The runaway: the verb "repeat" makes the model actually repeat.
            Rule(match="Repeat one word:", tool_calls=call(), completion_tokens=653, fire_rate=0.375),
            # The cheap trap: reads as "just answer me", so the tool call is suppressed.
            # Note it decodes FEWER tokens than the control -- it looks like a win.
            Rule(match="Just answer:", tool_calls=call(), completion_tokens=31, fire_rate=0.65,
                 miss_completion_tokens=24),
            Rule(match="Only acknowledge:", tool_calls=call(), completion_tokens=31, fire_rate=0.66,
                 miss_completion_tokens=24),
            # The catastrophic trap: 'Print' is read as "emit text", so the model
            # answers instead of calling -- and the failures are CHEAP, so the
            # aggregate token count looks healthy.
            Rule(match="Print the arguments", tool_calls=call(), completion_tokens=25,
                 fire_rate=0.035, miss_completion_tokens=2),
            # Good, cost-uniform variants.
            Rule(match="Reply only:", tool_calls=call(), completion_tokens=35),
            Rule(match="Respond:", tool_calls=call(), completion_tokens=35),
            Rule(match="Echo with:", tool_calls=call(), completion_tokens=35),
            Rule(match="Answer with:", tool_calls=call(), completion_tokens=36),
            # Fires, but expensive: a "back:" form implies a reply thread.
            Rule(match="Reply back:", tool_calls=call(), completion_tokens=53),
            Rule(match="Log with:", tool_calls=call(), completion_tokens=64),
        ],
        default=Rule(match=lambda _s: True, tool_calls=call(), completion_tokens=42),
    )
