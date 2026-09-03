from __future__ import annotations

import re

import pytest

from promptlab import (
    audit_pool,
    blast_radius,
    build_composed_pool,
    shape_collapse,
    similarity_profile,
    single_token_fragments,
    window_survival,
)
from promptlab.backends import MockBackend
from promptlab.backends.base import TokenizerUnavailable


@pytest.fixture
def backend():
    return MockBackend()


# ---------------------------------------------------------------- tokens
def test_audit_pool_finds_a_non_uniform_pool(backend):
    # 'cale' and 'bab' are single vocabulary entries; 'xqz' is not.
    pool = ["cale@bab.com", "bab@bab.com", "xqzv@bab.com"]
    a = audit_pool(backend, pool)
    assert len(a.histogram) > 1, "pool is not token-uniform"
    assert a.waste_per_candidate > 0
    assert "flattening" in a.summary()


def test_audit_pool_flat_pool_has_no_waste(backend):
    a = audit_pool(backend, ["bab@bab.com", "bab@bab.com"])
    assert a.waste_per_candidate == pytest.approx(0.0)


def test_context_matters(backend):
    """A value's cost in isolation is not its cost in the prompt."""
    pool = ["cale", "bab"]
    alone = audit_pool(backend, pool)
    in_ctx = audit_pool(backend, pool, context=lambda v: f"A note for {v}. Subject: ok")
    assert in_ctx.in_context is True
    assert alone.in_context is False
    # both are well-defined; the point is that they are measured differently
    assert in_ctx.mean >= 0


def test_single_token_fragments(backend):
    frags = single_token_fragments(backend, ["cale", "bab", "kot", "xqzv", "zzzz"])
    assert "bab" in frags and "kot" in frags
    assert "xqzv" not in frags


def test_single_token_fragments_respects_exclude(backend):
    frags = single_token_fragments(backend, ["cale", "bab"], exclude={"bab"})
    assert "bab" not in frags


def test_build_composed_pool_meets_the_size_requirement(backend):
    pool, audit = build_composed_pool(
        backend,
        locals_=[f"{a}{b}{c}" for a in "bcd" for b in "aeiou" for c in "bcdfg"],
        domains=["bab", "cale"],
        tlds=[".com", ".net", ".org"],
        size=120,
    )
    assert len(pool) >= 120
    assert len(set(pool)) == len(pool), "every value must be distinct"


def test_build_composed_pool_refuses_to_silently_undersize(backend):
    """A pool smaller than N is the classic quiet failure — it must raise, not truncate."""
    with pytest.raises(ValueError, match="could only build"):
        build_composed_pool(backend, ["aa"], ["bab"], [".com"], size=500)


def test_word_filter_is_applied(backend):
    pool, _ = build_composed_pool(
        backend,
        locals_=["cat", "dog", "bab", "kot", "vex", "zuq"],
        domains=["bab"],
        tlds=[".com"],
        size=3,
        word_filter=lambda v: v not in {"cat", "dog"},
    )
    assert not any(v.startswith(("cat@", "dog@")) for v in pool)


def test_tokenizer_unavailable_is_explicit():
    class NoTok(MockBackend):
        def token_ids(self, text):
            raise TokenizerUnavailable("nope")

    with pytest.raises(TokenizerUnavailable):
        single_token_fragments(NoTok(), ["ab"])


# ---------------------------------------------------------------- coverage
def test_blast_radius():
    r = blast_radius(["a"] * 70 + ["b"] * 20 + ["c"] * 10)
    assert r.n == 100 and r.unique == 3
    assert r.max_blast_radius == 70
    assert r.coverage_of_worst == pytest.approx(0.70)


def test_window_survival_is_pool_size_minus_one_for_round_robin():
    vals = [f"v{i % 7}" for i in range(100)]
    assert window_survival(vals) == 6


def test_window_survival_zero_when_a_value_repeats_immediately():
    assert window_survival(["a", "a", "b"]) == 0


def test_similarity_profile_detects_cosmetic_variation():
    """Rotating one slot inside a fixed frame does NOT defeat a similarity check."""
    cosmetic = [f"A note for user{i}@bab.com. Subject: ok. Reply only: ok" for i in range(60)]
    prof = similarity_profile(cosmetic, window=50)
    assert prof["p95"] > 0.90
    assert prof["over_threshold"] > 0.5


_SKELETONS = [
    "{u} should review the migration before Friday",
    "Could you ask {u} whether the backup completed?",
    "Reminder: nothing has shipped to {u} since Tuesday",
    "The invoice bounced. {u} needs a corrected copy.",
    "Per our call, I have queued the index rebuild for {u}",
    "Heads up - {u} is on the escalation path for this ticket",
    "Blocked on {u}; the deployment cannot proceed without sign-off",
    "FYI {u}: yesterday's report has been regenerated",
    "Whenever you get a moment, loop {u} into the thread",
    "Closing this out. {u} has the final numbers.",
]


def _stream(n_skeletons: int, n: int) -> list[str]:
    sk = _SKELETONS[:n_skeletons]
    return [sk[i % n_skeletons].format(u=f"user{i}@bab.com") for i in range(n)]


def test_variation_narrower_than_the_window_does_not_defeat_it():
    """3 distinct skeletons inside a window of 20 still repeat - variation must EXCEED W."""
    assert similarity_profile(_stream(3, 60), window=20)["p95"] > 0.90


def test_variation_wider_than_the_window_defeats_it():
    """The same check passes once the skeleton pool is larger than the window."""
    prof = similarity_profile(_stream(10, 60), window=8)
    assert prof["p95"] < 0.90
    assert prof["over_threshold"] < 0.10


def test_slot_swaps_inside_one_frame_are_not_skeleton_variation():
    """The trap: 60 'unique' messages that a fuzzy check sees as one message."""
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    cosmetic = [
        f"A note for user{i}@bab.com. Subject: {words[i % 10]}. Reply only: ok"
        for i in range(60)
    ]
    assert len(set(cosmetic)) == 60
    assert similarity_profile(cosmetic, window=50)["p95"] > 0.90


def test_shape_collapse_catches_identifier_only_rotation():
    """Mask the identifier and 'unique' messages collapse to one skeleton."""
    msgs = [f"A note for user{i}@bab.com. Subject: ok" for i in range(200)]
    assert len(set(msgs)) == 200
    collapsed = shape_collapse(msgs, lambda m: re.sub(r"[\w.]+@[\w.]+", "<ID>", m))
    assert collapsed.unique == 1
    assert collapsed.coverage_of_worst == 1.0
