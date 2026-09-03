from __future__ import annotations

import pytest

from promptlab import fit_cost_model
from promptlab.cost import CostModel


def test_fit_recovers_known_coefficients():
    """The two banked points from the write-up solve exactly."""
    m = fit_cost_model([(8, 170, 36.80), (2, 25, 5.84)])
    assert m.seconds_per_hop == pytest.approx(0.52, abs=0.005)
    assert m.seconds_per_token == pytest.approx(0.192, abs=0.001)


def test_fit_is_overdetermined_safe():
    m = fit_cost_model([(8, 170, 36.80), (2, 25, 5.84), (4, 88, 19.0), (1, 20, 4.4)])
    assert 0.4 < m.seconds_per_hop < 0.7
    assert 0.15 < m.seconds_per_token < 0.25


def test_fit_refuses_a_design_that_cannot_separate_hops_from_tokens():
    """Arms that differ only in tokens cannot tell you the per-hop cost.

    Fitting them anyway is how a per-HOP cost gets misread as a per-CANDIDATE constant.
    """
    with pytest.raises(ValueError, match="hop count"):
        fit_cost_model([(8, 170, 36.8), (8, 88, 20.0)])


def test_prediction_and_throughput():
    m = CostModel(0.52, 0.192)
    assert m.predict(8, 170) == pytest.approx(36.80, abs=0.01)
    assert m.throughput(130, 8, 170) == pytest.approx(130 / 36.80, rel=1e-6)


def test_exchange_rate_is_the_headline_number():
    """One decode token costs what ~46 prompt tokens cost at 8 hops, ~185 at 2."""
    m = CostModel(0.52, 0.192)
    assert m.token_exchange_rate(8, 1000) == pytest.approx(46, abs=2)
    assert m.token_exchange_rate(2, 1000) == pytest.approx(185, abs=5)
    # and per-token, decode is ~370x prefill
    assert m.seconds_per_token * m.prefill_tokens_per_second(1000) == pytest.approx(369, abs=5)


def test_wrapup_tax():
    """A 4-token trailing hop is >20% of a 2-hop candidate, for zero score."""
    m = CostModel(0.52, 0.192)
    assert m.wrapup_cost(4, 2, 25) == pytest.approx(0.221, abs=0.01)
    # ...and at the hop cap the same wrap-up is a much smaller share of a bigger candidate
    assert m.wrapup_cost(4, 8, 170) < 0.05


def test_more_hops_wins_when_the_predicate_scores_per_hop():
    """k=8 at 16 raw/post beats k=1, even though it costs 6x the wall clock."""
    m = CostModel(0.52, 0.192)
    k8 = m.throughput(8 * 16 + 2, 8, 170)
    k1 = m.throughput(1 * 16 + 2, 2, 25)
    assert k8 > k1


def test_more_hops_loses_when_the_predicate_scores_per_candidate():
    """Same model, per-cell predicate: extra hops are pure cost."""
    m = CostModel(0.52, 0.192)
    one_hop = m.throughput(6.0, 2, 32)
    four_hops = m.throughput(6.0, 4, 64)
    assert one_hop > four_hops
