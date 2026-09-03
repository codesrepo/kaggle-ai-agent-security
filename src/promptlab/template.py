"""Segmented prompt templates with coprime rotation.

A prompt is not one string, it is a sequence of independently varying **segments**.
Splitting it that way buys two things at once:

1. **Screening.** Each segment is an axis you can screen in isolation, in frame, against
   the real objective (see `promptlab.screen`).
2. **Coverage.** If a defense keys on any single observable — a value, a phrase, a shape —
   the damage is bounded by that segment's pool size, not by the whole run.

The rotation arithmetic that matters:

* A segment with ``n`` values, cycled round-robin, repeats a value every ``n`` candidates.
  That is **provably the best you can do**: over any horizon, any schedule using ``n``
  distinct values has some pair at distance <= n. So to survive a "blocked if seen in the
  last W" rule you need exactly ``n = W + 1`` values — and never more.
* To survive a "usable once, ever" rule you need ``n >= N`` (the number of candidates).
* Choose segment periods **pairwise coprime**, or two segments phase-lock and the pair
  repeats far sooner than either segment alone. Periods 1000/5/5 weld two segments
  together every 5 candidates; 1009/101/32 never repeat the tuple inside any realistic run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

__all__ = ["Segment", "Template", "next_prime", "suggest_periods", "is_pairwise_coprime"]


def next_prime(n: int) -> int:
    """Smallest prime >= max(n, 2)."""
    n = max(2, int(n))

    def prime(k: int) -> bool:
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        f = 3
        while f * f <= k:
            if k % f == 0:
                return False
            f += 2
        return True

    while not prime(n):
        n += 1
    return n


def is_pairwise_coprime(periods: Sequence[int]) -> bool:
    return all(math.gcd(a, b) == 1 for a, b in combinations([p for p in periods if p > 1], 2))


def suggest_periods(sizes: Sequence[int]) -> list[int]:
    """Round each pool size up to a prime, keeping every period distinct.

    Distinct primes are automatically pairwise coprime, which is the cheapest way to
    guarantee no two segments phase-lock.
    """
    out: list[int] = []
    for s in sizes:
        p = next_prime(max(2, s))
        while p in out:
            p = next_prime(p + 1)
        out.append(p)
    return out


@dataclass
class Segment:
    """One rotating slot in a template.

    values : the pool. Screen it before you ship it (`promptlab.screen`).
    period : how often a value recurs. Defaults to len(values). Setting it *below*
             len(values) means part of the pool is never used — the class warns about it
             in `audit()` rather than silently dropping values.
    stride : advance by this many indices per candidate. Must be coprime with `period`
             or the segment silently visits only a subset of its own pool.
    derive : instead of a pool, compute the value from the already-rendered segments.
             This is how you make a slot unique-per-candidate for free: reuse a token
             that is already in the prompt (it costs no new decode tokens) rather than
             introducing a new one.
    """

    name: str
    values: Sequence[str] = field(default_factory=tuple)
    period: int | None = None
    stride: int = 1
    derive: Callable[[dict[str, str]], str] | None = None

    def __post_init__(self) -> None:
        self.values = tuple(self.values)
        if self.derive is None:
            if not self.values:
                raise ValueError(f"segment {self.name!r} needs values or a derive()")
            if self.period is None:
                self.period = len(self.values)
            if self.period < 1:
                raise ValueError(f"segment {self.name!r} period must be >= 1")
            if math.gcd(self.stride, self.period) != 1:
                raise ValueError(
                    f"segment {self.name!r}: stride {self.stride} shares a factor with "
                    f"period {self.period}, so it would visit only "
                    f"{self.period // math.gcd(self.stride, self.period)} of its values"
                )
        else:
            self.period = self.period or 1

    @property
    def n_used(self) -> int:
        """How many distinct values this segment actually emits."""
        if self.derive is not None:
            return 0  # unbounded / determined by what it derives from
        return min(len(self.values), self.period or len(self.values))

    def render(self, i: int, ctx: dict[str, str]) -> str:
        if self.derive is not None:
            return self.derive(ctx)
        assert self.period is not None
        return self.values[(i * self.stride) % self.period % len(self.values)]

    def survives_window(self) -> int:
        """Largest W such that no value recurs within the last W candidates."""
        return max(0, self.n_used - 1)

    def survives_permanent(self, n_candidates: int) -> bool:
        """True if every candidate can use a value never used before."""
        return self.n_used >= n_candidates


class Template:
    """An ordered mix of literal strings and Segments.

        t = Template([
            Segment("open", ["A note for", "A ping for"]),
            " ",
            Segment("to", recipients),
            ". Subject: ",
            Segment("subject", subjects),
            ". Body: ",
            Segment("body", derive=lambda c: c["to"].split("@")[0]),
        ])
        t.render(0)
    """

    def __init__(self, parts: Iterable[str | Segment]) -> None:
        self.parts = list(parts)
        self.segments = [p for p in self.parts if isinstance(p, Segment)]
        names = [s.name for s in self.segments]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate segment names: {names}")

    # -- rendering -------------------------------------------------------------
    def render(self, i: int) -> str:
        ctx: dict[str, str] = {}
        out: list[str] = []
        for part in self.parts:
            if isinstance(part, Segment):
                v = part.render(i, ctx)
                ctx[part.name] = v
                out.append(v)
            else:
                out.append(part)
        return "".join(out)

    def render_parts(self, i: int) -> dict[str, str]:
        ctx: dict[str, str] = {}
        for part in self.parts:
            if isinstance(part, Segment):
                ctx[part.name] = part.render(i, ctx)
        return ctx

    def stream(self, n: int) -> list[str]:
        return [self.render(i) for i in range(n)]

    def with_override(self, name: str, values: Sequence[str]) -> Template:
        """A copy with one segment's pool replaced — how the screener isolates an axis."""
        parts: list[str | Segment] = []
        for p in self.parts:
            if isinstance(p, Segment) and p.name == name:
                parts.append(Segment(p.name, values, period=len(values), stride=1))
            else:
                parts.append(p)
        return Template(parts)

    # -- analysis --------------------------------------------------------------
    @property
    def combined_period(self) -> int:
        periods = [s.period or 1 for s in self.segments if s.derive is None]
        return math.lcm(*periods) if periods else 1

    def audit(self, n_candidates: int) -> dict[str, object]:
        """Everything you need to decide whether the rotation is actually adequate."""
        rows = []
        for s in self.segments:
            rows.append(
                {
                    "segment": s.name,
                    "values": len(s.values) if s.derive is None else "derived",
                    "used": s.n_used if s.derive is None else "per-candidate",
                    "period": s.period,
                    "window_survived": (
                        n_candidates - 1 if s.derive is not None else s.survives_window()
                    ),
                    "permanent_safe": (
                        True if s.derive is not None else s.survives_permanent(n_candidates)
                    ),
                }
            )
        periods = [s.period or 1 for s in self.segments if s.derive is None]
        warnings: list[str] = []
        if not is_pairwise_coprime(periods):
            bad = [
                (a.name, b.name)
                for a, b in combinations([s for s in self.segments if s.derive is None], 2)
                if math.gcd(a.period or 1, b.period or 1) > 1
            ]
            warnings.append(
                f"periods are not pairwise coprime {bad}: those segments phase-lock, so the "
                f"PAIR repeats far sooner than either segment alone"
            )
        for s in self.segments:
            if s.derive is None and s.period is not None and s.period < len(s.values):
                warnings.append(
                    f"segment {s.name!r}: period {s.period} < pool {len(s.values)} — "
                    f"{len(s.values) - s.period} values are never used"
                )
        rendered = self.stream(n_candidates)
        return {
            "n_candidates": n_candidates,
            "segments": rows,
            "combined_period": self.combined_period,
            "unique_messages": len(set(rendered)),
            "binding_axis": min(
                (r for r in rows if isinstance(r["window_survived"], int)),
                key=lambda r: r["window_survived"],
                default={"segment": None},
            )["segment"],
            "max_window_survived": min(
                (r["window_survived"] for r in rows if isinstance(r["window_survived"], int)),
                default=0,
            ),
            "warnings": warnings,
        }
