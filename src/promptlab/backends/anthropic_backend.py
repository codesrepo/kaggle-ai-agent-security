"""Anthropic Messages API backend.

Requires `pip install anthropic`. Credentials resolve from the environment
(`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile), so the
zero-argument constructor is usually right.

A note on what `completion_tokens` means here, because it decides what the whole harness
is optimizing. `usage.output_tokens` **includes thinking tokens**. That is deliberate:
thinking tokens are decode tokens, they are billed as decode tokens, and they cost
wall-clock like decode tokens. If your objective is "how expensive is this phrasing",
you want them counted. Use `effort` to move that number, not a token-accounting trick.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .base import Completion, Message, TokenizerUnavailable, ToolCall

__all__ = ["AnthropicBackend"]

DEFAULT_MODEL = "claude-opus-5"


class AnthropicBackend:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 4096,
        effort: str | None = "low",
        system: str | None = None,
        client: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        # max_tokens is deliberately small: this harness measures short tool calls across
        # thousands of repetitions, so a low ceiling is a cost cap, not an accident. Raise
        # it if your objective produces long outputs.
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "AnthropicBackend requires the official SDK: pip install anthropic"
            ) from exc
        import anthropic

        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.system = system
        self.extra = dict(extra or {})
        self._client = client or anthropic.Anthropic()

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "messages": [m.as_dict() for m in messages if m.role != "system"],
        }
        system = self.system or next((m.content for m in messages if m.role == "system"), None)
        if system:
            req["system"] = system
        if tools:
            req["tools"] = list(tools)
        if self.effort:
            req["output_config"] = {"effort": self.effort}
        req.update(self.extra)
        req.update(kwargs)

        resp = self._client.messages.create(**req)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Tool inputs arrive already parsed; never string-match the serialized form.
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(name=block.name, arguments=args))

        return Completion(
            text="".join(text_parts),
            tool_calls=calls,
            prompt_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            hops=1,
            raw=resp,
        )

    def count_tokens(self, text: str) -> int:
        """Exact server-side count. Costs an API call — cache it if you scan a big pool."""
        resp = self._client.messages.count_tokens(
            model=self.model, messages=[{"role": "user", "content": text}]
        )
        return int(resp.input_tokens)

    def token_ids(self, text: str) -> list[int]:
        raise TokenizerUnavailable(
            "The Messages API exposes token counts, not token ids. "
            "Use count_tokens() for pool auditing, or a local backend for id-level work."
        )

    def vocab(self) -> Iterable[str]:
        raise TokenizerUnavailable("No local vocabulary; single-token scans need a local tokenizer.")
