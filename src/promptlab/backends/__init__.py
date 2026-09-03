"""Backend registry.

    from promptlab.backends import load_backend
    be = load_backend({"kind": "mock"})
    be = load_backend({"kind": "openai_compat", "model": "...", "base_url": "..."})
    be = load_backend({"kind": "anthropic",     "model": "claude-opus-5"})

`load_backend` takes a plain dict so a backend can come straight out of a YAML/JSON
config file — see `config.example.yaml`. Real backends are imported lazily so that
`pip install promptlab` with no extras still runs the full test suite on the mock.
"""

from __future__ import annotations

from typing import Any

from .base import Backend, Completion, Message, TokenizerUnavailable, ToolCall
from .mock import MockBackend, Rule, build_demo_backend

__all__ = [
    "Backend",
    "Completion",
    "Message",
    "ToolCall",
    "TokenizerUnavailable",
    "MockBackend",
    "Rule",
    "build_demo_backend",
    "load_backend",
    "REGISTRY",
]

REGISTRY = ("mock", "demo", "openai_compat", "anthropic")


def load_backend(config: dict[str, Any]) -> Backend:
    cfg = dict(config)
    kind = cfg.pop("kind", "mock")

    if kind == "mock":
        return MockBackend(**cfg)
    if kind == "demo":
        return build_demo_backend(**cfg)
    if kind == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(**cfg)
    if kind == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(**cfg)

    raise ValueError(f"unknown backend kind {kind!r}; expected one of {REGISTRY}")
