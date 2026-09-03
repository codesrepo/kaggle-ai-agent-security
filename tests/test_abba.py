"""ABBA must cancel host drift. These tests inject drift and check that it does."""

from __future__ import annotations

import pytest

from promptlab import abba_compare, blocks_needed


def make_drifting_arms(true_ratio: float, drift_per_call: float):
    """Two arms with a known efficiency ratio, on a host that gets steadily slower.

    Tokens grow with call order regardless of arm — that is the drift. A naive
    "measure A then measure B" design attributes all of it to B.
    """
    clock = {"n": 0}

    def arm(tokens: float):
        def run(i: int):
            clock["n"] += 1
            t = tokens * (1.0 + drift_per_call * clock["n"])
            return (1.0, int(round(t)), 0.0)
        return run

    return arm(100.0), arm(100.0 / true_ratio)


def test_abba_recovers_the_true_ratio_under_drift():
    a, b = make_drifting_arms(true_ratio=1.10, drift_per_call=0.002)
    res = abba_compare(a, b, blocks=8, per_block=20, warmup=0)
    # B decodes ~1/1.10 as many tokens, so score/token should be ~+10%.
    assert res.delta_efficiency == pytest.approx(0.10, abs=0.02)


def test_naive_sequential_design_is_fooled_by_the_same_drift():
    """Control: A-then-B on the same drifting host gives the wrong answer."""
    a, b = make_drifting_arms(true_ratio=1.0, drift_per_call=0.004)
    tot_a = sum(a(i)[1] for i in range(160))
    tot_b = sum(b(i)[1] for i in range(160))
    naive = tot_a / tot_b - 1.0
    assert abs(naive) > 0.30, "precondition: the drift is big enough to matter"

    a2, b2 = make_drifting_arms(true_ratio=1.0, drift_per_call=0.004)
    res = abba_compare(a2, b2, blocks=8, per_block=20, warmup=0)
    assert abs(res.delta_efficiency) < 0.02, "ABBA should see ~no difference"


def test_totals_not_means_of_ratios():
    """Efficiency is total score / total tokens, so a big block cannot be out-voted."""
    def a(i):
        return (1.0, 10 if i < 5 else 100, 0.0)

    def b(i):
        return (1.0, 20, 0.0)

    res = abba_compare(a, b, blocks=2, per_block=10, warmup=0)
    # A: 5x10 + 15x100 over 20 calls -> mean 77.5; B: 20. So B is far more efficient.
    assert res.b.efficiency > res.a.efficiency


def test_warmup_discards_the_first_candidate_of_each_block():
    seen: list[int] = []

    def a(i):
        seen.append(i)
        return (1.0, 10, 0.0)

    res = abba_compare(a, a, blocks=2, per_block=5, warmup=1)
    assert res.a.n == 2 * (5 - 1)


def test_noise_floor_is_reported_and_flags_unresolvable_deltas():
    import random

    rng = random.Random(0)

    def noisy(base: float):
        def run(i: int):
            return (1.0, int(base * rng.uniform(0.7, 1.3)), 0.0)
        return run

    res = abba_compare(noisy(100), noisy(101), blocks=4, per_block=10, warmup=0)
    assert res.block_sd_pct is not None and res.block_sd_pct > 0
    r = res.resolvable()
    assert r is not None
    assert "noise floor" in res.summary() or abs(res.delta_efficiency) >= r


def test_blocks_needed_matches_the_stated_rule_of_thumb():
    # ~1.9% block sd, wanting to resolve 0.33% -> ~127 blocks.
    assert 100 <= blocks_needed(0.0033, 1.9) <= 140
    # resolving 2% needs only a handful
    assert blocks_needed(0.02, 1.9) <= 5


def test_requires_at_least_two_blocks():
    with pytest.raises(ValueError):
        abba_compare(lambda i: (1, 1, 0), lambda i: (1, 1, 0), blocks=1)
