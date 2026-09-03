"""Backend protocol.

Everything else in promptlab talks to a model through this interface and nothing else,
which is what makes the harness model-agnostic. A backend needs to do three things:

1. produce a completion for a list of chat messages,
2. report how many tokens it decoded (this is the cost metric that matters),
3. tokenize a string (needed for pool auditing in `promptlab.tokens`).

Only (1) and (2) are required. A backend that cannot tokenize simply disables the
token-floor tooling; everything else still works.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Message", "ToolCall", "Completion", "Backend", "TokenizerUnavailable"]


class TokenizerUnavailable(RuntimeError):
    """Raised when a backend is asked to tokenize but has no tokenizer."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation parsed out of a completion."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, name: str, raw: str) -> ToolCall:
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            args = {}
        return cls(name=name, arguments=args if isinstance(args, dict) else {})


@dataclass
class Completion:
    """What a backend returns for one generation.

    `completion_tokens` is the number the whole harness optimizes against. If a
    backend cannot report it, it must estimate it — a wrong-but-consistent estimate
    still ranks variants correctly; a missing one silently disables the cost model.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hops: int = 1
    raw: Any = None

    @property
    def fired(self) -> bool:
        return bool(self.tool_calls)

    def calls_to(self, tool_name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == tool_name]


@runtime_checkable
class Backend(Protocol):
    """The only thing promptlab needs from a model."""

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """Generate one completion. Must populate `completion_tokens`."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens `text` encodes to. May raise TokenizerUnavailable."""

    def token_ids(self, text: str) -> list[int]:
        """Token ids for `text`. May raise TokenizerUnavailable."""

    def vocab(self) -> Iterable[str]:
        """All vocabulary strings, for single-token scans. May raise TokenizerUnavailable."""
