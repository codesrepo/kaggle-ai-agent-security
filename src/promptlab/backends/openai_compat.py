"""OpenAI-compatible HTTP backend.

Covers most self-hosted servers, which is where you want to be for this kind of work —
they are the only setups that let you run tens of thousands of generations cheaply and
give you a local tokenizer:

    vLLM             python -m vllm.entrypoints.openai.api_server --model <hf-id>
    llama.cpp        llama-server -m model.gguf --port 8080
    Ollama           ollama serve         (base_url http://localhost:11434/v1)
    LM Studio, TGI, OpenRouter, ...

Only `requests` is needed. If `transformers` is installed and you pass `tokenizer=`,
the token-floor tooling in `promptlab.tokens` lights up as well.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from .base import Completion, Message, TokenizerUnavailable, ToolCall

__all__ = ["OpenAICompatBackend"]


class OpenAICompatBackend:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        tokenizer: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("OpenAICompatBackend requires: pip install requests") from exc

        self.name = f"openai-compat:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # temperature=0 by default: variant screening needs the decode count to be a
        # property of the prompt, not of the sampler.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra = dict(extra or {})
        self._tok = tokenizer

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Completion:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        payload.update(self.extra)
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(ToolCall.from_json(fn.get("name", ""), fn.get("arguments", "{}")))

        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if not completion_tokens:
            # Some servers omit usage. Estimate rather than silently reporting 0 — a 0
            # would make every variant look free and quietly break the whole ranking.
            body = (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or "")
            completion_tokens = self._estimate(body)

        return Completion(
            text=msg.get("content") or "",
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=completion_tokens,
            hops=1,
            raw=data,
        )

    # -- tokenizer -------------------------------------------------------------
    def _estimate(self, text: str) -> int:
        try:
            return self.count_tokens(text)
        except TokenizerUnavailable:
            return max(1, len(text) // 4)

    def count_tokens(self, text: str) -> int:
        return len(self.token_ids(text))

    def token_ids(self, text: str) -> list[int]:
        if self._tok is None:
            raise TokenizerUnavailable(
                "Pass tokenizer=AutoTokenizer.from_pretrained(<model>) to enable token tooling."
            )
        return list(self._tok.encode(text, add_special_tokens=False))

    def vocab(self) -> Iterable[str]:
        if self._tok is None:
            raise TokenizerUnavailable("No tokenizer configured.")
        get = getattr(self._tok, "get_vocab", None)
        if get is None:
            raise TokenizerUnavailable("Tokenizer exposes no get_vocab().")
        return tuple(get().keys())
