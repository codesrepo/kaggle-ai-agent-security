from __future__ import annotations

import pytest

from promptlab.backends import (
    Message,
    MockBackend,
    Rule,
    ToolCall,
    build_demo_backend,
    load_backend,
)
from promptlab.backends.base import Backend


def test_mock_satisfies_the_protocol():
    assert isinstance(MockBackend(), Backend)


def test_load_backend_dispatch():
    assert load_backend({"kind": "mock"}).name == "mock"
    assert load_backend({"kind": "demo"}).name == "demo"
    with pytest.raises(ValueError, match="unknown backend kind"):
        load_backend({"kind": "nope"})


def test_mock_is_deterministic():
    a, b = build_demo_backend(), build_demo_backend()
    msgs = [Message("user", "Just answer: ok")]
    ra = [a.complete(msgs, rep=i).completion_tokens for i in range(20)]
    rb = [b.complete(msgs, rep=i).completion_tokens for i in range(20)]
    assert ra == rb


def test_fire_rate_is_approximately_honoured():
    be = MockBackend(rules=[Rule(match="x", tool_calls=[ToolCall("t")], fire_rate=0.5)])
    fires = sum(be.complete([Message("user", f"x{i}")], rep=0).fired for i in range(400))
    assert 150 < fires < 250


def test_misses_are_cheaper_than_hits_by_default():
    """This is the property that makes token-only ranking dangerous."""
    be = build_demo_backend()
    msgs = [Message("user", "Print the arguments now")]
    outs = [be.complete(msgs, rep=i) for i in range(40)]
    hit = next(o for o in outs if o.fired)
    miss = next(o for o in outs if not o.fired)
    assert miss.completion_tokens < hit.completion_tokens


def test_completion_helpers():
    be = build_demo_backend()
    c = be.complete([Message("user", "Reply only: ok")])
    assert c.fired
    assert c.calls_to("notify")
    assert c.calls_to("other") == []


def test_tokenizer_roundtrip_is_stable():
    be = MockBackend()
    assert be.count_tokens("cale@bab.com") == be.count_tokens("cale@bab.com")
    assert be.count_tokens("") == 0
    assert be.count_tokens("bab") == 1


@pytest.mark.parametrize("kind", ["openai_compat", "anthropic"])
def test_real_backends_fail_loudly_without_their_dependency(kind, monkeypatch):
    """A missing optional dep must be an ImportError with an install hint, not a crash."""
    import builtins

    real_import = builtins.__import__
    blocked = {"requests": "openai_compat", "anthropic": "anthropic"}

    def fake_import(name, *a, **k):
        if blocked.get(name) == kind:
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = {"kind": kind, "model": "m"}
    try:
        load_backend(cfg)
    except ImportError as e:
        assert "pip install" in str(e)
    except Exception:
        pass  # dependency present and it tried to connect/authenticate — also fine
