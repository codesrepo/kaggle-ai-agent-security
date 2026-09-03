"""The screener must reject the traps that a token metric would select FOR."""

from __future__ import annotations

import pytest

from promptlab import Segment, Template, ToolCallObjective, screen_pool
from promptlab.backends import build_demo_backend


@pytest.fixture
def setup():
    backend = build_demo_backend()
    template = Template(
        [
            "A note for ",
            Segment("to", [f"u{i}@bab.com" for i in range(50)]),
            ". ",
            Segment("frame", ["Reply only:"]),
            " ok",
        ]
    )
    objective = ToolCallObjective(tool_name="notify", required_args=["to"], ceiling=1.0)
    return backend, template, objective


VARIANTS = [
    "Reply only:",
    "Respond:",
    "Echo with:",
    "Answer with:",
    "Just answer:",
    "Only acknowledge:",
    "Print the arguments",
    "Repeat one word:",
    "Reply back:",
    "Log with:",
]


def test_keeps_only_ceiling_and_cost_uniform(setup):
    backend, template, objective = setup
    r = screen_pool(backend, template, "frame", VARIANTS, objective,
                    reps=8, control="Reply only:", token_tolerance=0.02)
    assert set(r.kept) == {"Reply only:", "Respond:", "Echo with:"}


def test_rejects_the_cheap_trap(setup):
    """'Print the arguments' is by far the CHEAPEST variant and must still be rejected."""
    backend, template, objective = setup
    r = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                    control="Reply only:")
    by_variant = {x.variant: x for x in r.results}

    trap = by_variant["Print the arguments"]
    control = by_variant["Reply only:"]
    assert trap.mean_tokens < control.mean_tokens, "precondition: the trap looks cheaper"
    assert trap.variant not in r.kept
    assert "below ceiling" in dict(r.rejected)[trap.variant]


def test_rejects_the_runaway(setup):
    backend, template, objective = setup
    r = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                    control="Reply only:")
    by_variant = {x.variant: x for x in r.results}
    assert by_variant["Repeat one word:"].max_tokens_seen > 500
    assert "Repeat one word:" not in r.kept


def test_a_token_only_ranking_would_pick_a_broken_variant(setup):
    """This is the whole reason the module exists — assert the failure mode is real."""
    backend, template, objective = setup
    r = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                    control="Reply only:")
    cheapest = min(r.results, key=lambda x: x.mean_tokens)
    assert cheapest.mean_score < objective.ceiling
    assert cheapest.variant not in r.kept


def test_require_full_fire_can_be_relaxed(setup):
    backend, template, objective = setup
    strict = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                         control="Reply only:", require_full_fire=True)
    loose = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                        control="Reply only:", require_full_fire=False)
    assert set(strict.kept) <= set(loose.kept)


def test_forbid_guard_runs_on_the_assembled_message(setup):
    """Hard constraints apply to the WHOLE prompt, not just the segment under test."""
    backend, template, objective = setup
    seen: list[str] = []

    def forbid(msg: str) -> str | None:
        seen.append(msg)
        return "contains banned token" if "bab.com" in msg else None

    r = screen_pool(backend, template, "frame", ["Reply only:", "Respond:"], objective,
                    reps=2, forbid=forbid)
    assert seen and all("A note for" in m for m in seen)
    assert r.kept == []  # every message tripped the guard


def test_reliability_is_reported(setup):
    backend, template, objective = setup
    r = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                    control="Reply only:")
    assert r.token_reliability is not None
    assert 0.9 <= r.token_reliability <= 1.0  # mock is deterministic


def test_empty_pool_raises(setup):
    backend, template, objective = setup
    with pytest.raises(ValueError):
        screen_pool(backend, template, "frame", [], objective)


def test_low_reps_can_let_a_flaky_variant_through_but_high_reps_catches_it(setup):
    """Why the CLI nags about reps: 8 repetitions does not bound a fire rate.

    'Just answer:' fires ~70% of the time. At reps=8 it can look perfect; at reps=40 it
    is correctly rejected. Screen wide and cheap, then CONFIRM the survivors.
    """
    backend, template, objective = setup
    lo = screen_pool(backend, template, "frame", VARIANTS, objective, reps=8,
                     control="Reply only:")
    hi = screen_pool(backend, template, "frame", VARIANTS, objective, reps=40,
                     control="Reply only:")
    by_hi = {x.variant: x for x in hi.results}
    assert by_hi["Just answer:"].fire_rate < 1.0
    assert "Just answer:" not in hi.kept
    assert set(hi.kept) <= set(lo.kept) or "Just answer:" in lo.kept
