from __future__ import annotations

import math

import pytest

from promptlab import Segment, Template, next_prime, suggest_periods
from promptlab.template import is_pairwise_coprime


def test_render_and_derive():
    t = Template(
        [
            Segment("open", ["A", "B"]),
            " ",
            Segment("to", ["x@y.com", "p@q.com", "r@s.com"]),
            " ",
            Segment("body", derive=lambda c: c["to"].split("@")[0]),
        ]
    )
    assert t.render(0) == "A x@y.com x"
    assert t.render(1) == "B p@q.com p"
    assert t.render(2) == "A r@s.com r"


def test_derive_makes_a_segment_unique_per_candidate():
    n = 50
    t = Template(
        [
            Segment("to", [f"u{i}@bab.com" for i in range(n)]),
            " body=",
            Segment("body", derive=lambda c: c["to"].split("@")[0]),
        ]
    )
    bodies = [t.render_parts(i)["body"] for i in range(n)]
    assert len(set(bodies)) == n


def test_window_survival_equals_pool_size_minus_one():
    """Round-robin is provably optimal: repeat distance == number of values."""
    s = Segment("x", [f"v{i}" for i in range(7)])
    assert s.survives_window() == 6
    assert not s.survives_permanent(100)
    assert s.survives_permanent(7)


def test_stride_must_be_coprime_with_period():
    with pytest.raises(ValueError, match="shares a factor"):
        Segment("x", [f"v{i}" for i in range(10)], stride=2)
    Segment("x", [f"v{i}" for i in range(10)], stride=3)  # fine


def test_audit_flags_phase_lock():
    t = Template([Segment("a", ["1", "2", "3", "4"]), Segment("b", ["x", "y"])])
    audit = t.audit(20)
    assert any("pairwise coprime" in w for w in audit["warnings"])
    assert audit["combined_period"] == 4  # lcm(4,2) — b never varies independently


def test_audit_clean_when_coprime():
    t = Template([Segment("a", [str(i) for i in range(3)]),
                  Segment("b", [str(i) for i in range(5)]),
                  Segment("c", [str(i) for i in range(7)])])
    audit = t.audit(20)
    assert audit["warnings"] == []
    assert audit["combined_period"] == 105


def test_audit_flags_unused_values():
    t = Template([Segment("a", [str(i) for i in range(10)], period=3)])
    assert any("never used" in w for w in t.audit(10)["warnings"])


def test_binding_axis_is_the_narrowest():
    t = Template([Segment("wide", [str(i) for i in range(101)]),
                  Segment("narrow", [str(i) for i in range(3)])])
    audit = t.audit(50)
    assert audit["binding_axis"] == "narrow"
    assert audit["max_window_survived"] == 2


def test_with_override_isolates_one_segment():
    t = Template([Segment("a", ["1", "2"]), "-", Segment("b", ["x", "y"])])
    o = t.with_override("a", ["Z"])
    assert o.render(0) == "Z-x"
    assert o.render(1) == "Z-y"
    assert t.render(1) == "2-y"  # original untouched


def test_duplicate_segment_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Template([Segment("a", ["1"]), Segment("a", ["2"])])


def test_next_prime_and_suggest_periods():
    assert next_prime(100) == 101
    assert next_prime(2) == 2
    periods = suggest_periods([10, 10, 10])
    assert len(set(periods)) == 3
    assert is_pairwise_coprime(periods)
    assert all(p >= 10 for p in periods)


def test_suggested_periods_give_a_huge_combined_period():
    periods = suggest_periods([11, 101, 32])
    assert math.lcm(*periods) > 30_000
