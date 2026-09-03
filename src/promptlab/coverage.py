"""Coverage: how much of a run does one rule take out?

When you cannot observe a defense, the useful question is not "will this be blocked" but
"if it is, how much do I lose". That is a property of the candidate stream, and you can
compute it exactly, offline, from the rendered messages — no model needed.

Two disciplines worth separating, because they need different pool sizes:

* **permanent** — a value is usable once, ever. Survived iff the pool has >= N values.
* **window(W)** — a value is blocked if it recurs within the last W candidates. Survived
  iff the rotation period exceeds W.

`blast_radius` answers the third question people forget to ask: given that some rule keys
on segment X, how many candidates does a *single* blocked value cost? That number, not
the number of values, is what tells you whether a pool is big enough.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = ["CoverageReport", "blast_radius", "window_survival", "similarity_profile", "shape_collapse"]


@dataclass
class CoverageReport:
    n: int
    unique: int
    max_blast_radius: int
    worst_value: str | None

    @property
    def coverage_of_worst(self) -> float:
        return self.max_blast_radius / self.n if self.n else 0.0

    def summary(self, label: str = "") -> str:
        return (
            f"{label or 'stream'}: {self.n} candidates, {self.unique} distinct values\n"
            f"  worst single value {self.worst_value!r} covers {self.max_blast_radius} "
            f"candidates ({self.coverage_of_worst:.1%} of the run)"
        )


def blast_radius(values: Sequence[str]) -> CoverageReport:
    """How many candidates the most-repeated value accounts for."""
    c = Counter(values)
    if not c:
        return CoverageReport(0, 0, 0, None)
    worst, hits = c.most_common(1)[0]
    return CoverageReport(n=len(values), unique=len(c), max_blast_radius=hits, worst_value=worst)


def window_survival(values: Sequence[str]) -> int:
    """Largest W for which no value recurs inside a window of W candidates.

    Equivalently: (minimum gap between two uses of the same value) - 1. A round-robin over
    n values returns n-1, which is the optimum.
    """
    last: dict[str, int] = {}
    gap = len(values)
    for i, v in enumerate(values):
        if v in last:
            gap = min(gap, i - last[v])
        last[v] = i
    return max(0, gap - 1)


def similarity_profile(
    messages: Sequence[str], *, window: int = 50, threshold: float = 0.90
) -> dict[str, float]:
    """Fuzzy-duplicate profile: how similar is each message to recent ones?

    Rotating slots inside a fixed frame does *not* defeat a similarity check — the frame
    dominates the ratio. If `p95` sits near 1.0 while every message is technically unique,
    your variation is cosmetic and a fuzzy-dedup rule collapses the whole run.
    """
    if len(messages) < 2:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0, "over_threshold": 0.0}
    best: list[float] = []
    for i, m in enumerate(messages):
        lo = max(0, i - window)
        prev = messages[lo:i]
        best.append(max((SequenceMatcher(None, m, p).ratio() for p in prev), default=0.0))
    ranked = sorted(best)
    return {
        "max": ranked[-1],
        "mean": sum(ranked) / len(ranked),
        "p95": ranked[int(0.95 * (len(ranked) - 1))],
        "over_threshold": sum(1 for r in best if r > threshold) / len(best),
    }


def shape_collapse(
    messages: Sequence[str], normalizer: Callable[[str], str]
) -> CoverageReport:
    """Apply a normalizer (e.g. mask every address) and see what is left.

    This is the check that catches the most common self-deception in rotation design:
    varying only the identifier. Mask it out and 2,000 "unique" messages collapse to one
    skeleton — which is exactly what a shape-based rule sees.
    """
    return blast_radius([normalizer(m) for m in messages])
