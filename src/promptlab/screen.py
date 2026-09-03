"""Pool screening: find every phrasing that works, at the same cost.

The problem this solves. You want a segment to rotate over many values (so that no single
observable covers the whole run), but every value you add is a value that might be slower,
or might not work at all. The naive resolution is to trade diversity against speed. The
measured answer is that **you do not have to**: screen the pool member-by-member against
the real objective, keep only the ones that hit the ceiling *and* land inside a cost band,
and a 100-value rotation ends up costing the same as the single fixed value it replaced.

The gate order is not negotiable:

    1. score at the ceiling   (a variant that does not fire is not cheap, it is broken)
    2. cost inside the band   (only among variants that already work)

Doing it the other way round selects for the failures, because failures are cheap.

Screening runs are **interleaved by repetition, not blocked by variant** — every variant
is measured once per pass, in a shuffled order, so slow host drift lands on all variants
equally instead of on whichever ones happened to run late.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .backends.base import Backend, Message
from .objective import Objective
from .template import Template

__all__ = ["VariantResult", "ScreenReport", "screen_pool"]


@dataclass
class VariantResult:
    variant: str
    n: int = 0
    scores: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    fires: int = 0
    # split-half accumulators, for the reliability check
    half_tokens: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))

    @property
    def mean_score(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def fire_rate(self) -> float:
        return self.fires / self.n if self.n else 0.0

    @property
    def mean_tokens(self) -> float:
        return statistics.fmean(self.tokens) if self.tokens else 0.0

    @property
    def max_tokens_seen(self) -> int:
        return max(self.tokens) if self.tokens else 0

    @property
    def efficiency(self) -> float:
        """Score per decoded token — the thing a time-limited run actually maximizes."""
        return self.mean_score / self.mean_tokens if self.mean_tokens else 0.0


@dataclass
class ScreenReport:
    results: list[VariantResult]
    kept: list[str]
    rejected: list[tuple[str, str]]  # (variant, reason)
    ceiling: float
    control_tokens: float
    token_tolerance: float
    token_reliability: float | None

    def summary(self) -> str:
        lines = [
            f"screened {len(self.results)} variants",
            f"  at ceiling ({self.ceiling:g}) and 100% fire : "
            f"{sum(1 for r in self.results if r.mean_score >= self.ceiling and r.fire_rate >= 1.0)}",
            f"  ALSO within +{self.token_tolerance:.0%} of control ({self.control_tokens:.2f} tok)"
            f" : {len(self.kept)}",
        ]
        if self.token_reliability is not None:
            lines.append(f"  split-half reliability on token cost : rho = {self.token_reliability:.3f}")
            if self.token_reliability < 0.5:
                lines.append(
                    "  !! low reliability: the token differences you are ranking are mostly "
                    "noise. Raise reps before trusting this ordering."
                )
        return "\n".join(lines)

    def table(self, limit: int | None = 20) -> str:
        rows = sorted(self.results, key=lambda r: (-r.mean_score, r.mean_tokens))
        if limit:
            rows = rows[:limit]
        w = max((len(r.variant) for r in rows), default=10)
        out = [f"{'variant':<{w}}  {'score':>6} {'fire':>6} {'tok':>7} {'max':>6}  verdict"]
        keep = set(self.kept)
        why = dict(self.rejected)
        for r in rows:
            verdict = "KEEP" if r.variant in keep else why.get(r.variant, "-")
            out.append(
                f"{r.variant:<{w}}  {r.mean_score:>6.3f} {r.fire_rate:>6.1%} "
                f"{r.mean_tokens:>7.2f} {r.max_tokens_seen:>6d}  {verdict}"
            )
        return "\n".join(out)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def screen_pool(
    backend: Backend,
    template: Template,
    segment: str,
    variants: Sequence[str],
    objective: Objective,
    *,
    reps: int = 8,
    control: str | None = None,
    token_tolerance: float = 0.02,
    require_full_fire: bool = True,
    system: str | None = None,
    tools: Sequence[dict] | None = None,
    seed: int = 0,
    forbid: Callable[[str], str | None] | None = None,
    index_offset: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> ScreenReport:
    """Measure every value of one segment, in frame, against the real objective.

    control          : the incumbent value. Its measured cost is the band's centre.
                       Defaults to the first variant.
    token_tolerance  : keep variants within (1 + tol) x control cost.
    require_full_fire: reject anything that ever failed to produce a scoring call.
                       Leave this on. A variant at 7/8 is not "97% as good"; it is a
                       variant you do not understand yet.
    forbid           : optional guard run on the fully assembled message, returning a
                       reason string to reject it. Use this for hard constraints that
                       must hold for the *whole* prompt, not just the segment — the
                       assembled string is what the system on the other side sees.
    """
    variants = list(dict.fromkeys(variants))  # de-dupe, keep order
    if not variants:
        raise ValueError("no variants to screen")
    control = control if control is not None else variants[0]
    if control not in variants:
        variants = [control] + variants

    results = {v: VariantResult(variant=v) for v in variants}
    rng = random.Random(seed)
    order = list(range(len(variants)))
    total = reps * len(variants)
    done = 0

    for rep in range(reps):
        rng.shuffle(order)
        half = rep % 2
        for slot, vi in enumerate(order):
            v = variants[vi]
            i = index_offset + rep * len(variants) + slot
            msg = template.with_override(segment, [v]).render(i)

            done += 1
            if progress:
                progress(done, total)

            if forbid is not None:
                reason = forbid(msg)
                if reason:
                    results[v].n += 1
                    results[v].scores.append(0.0)
                    results[v].tokens.append(0)
                    continue

            messages = ([Message("system", system)] if system else []) + [Message("user", msg)]
            completion = backend.complete(messages, tools=tools, rep=rep)

            r = results[v]
            s = objective.score(completion)
            r.n += 1
            r.scores.append(s)
            r.tokens.append(completion.completion_tokens)
            r.half_tokens[half].append(completion.completion_tokens)
            if s > 0:
                r.fires += 1

    ctrl = results[control]
    ctrl_tok = ctrl.mean_tokens or 1.0
    band = ctrl_tok * (1.0 + token_tolerance)

    kept: list[str] = []
    rejected: list[tuple[str, str]] = []
    for v in variants:
        r = results[v]
        if r.mean_score < objective.ceiling:
            rejected.append(
                (v, f"below ceiling ({r.mean_score:.3f} < {objective.ceiling:g})"
                    + (" — CHEAPER than control, classic trap" if r.mean_tokens < ctrl_tok else ""))
            )
        elif require_full_fire and r.fire_rate < 1.0:
            rejected.append((v, f"fire {r.fire_rate:.1%} < 100%"))
        elif r.mean_tokens > band:
            rejected.append((v, f"cost +{(r.mean_tokens / ctrl_tok - 1):.1%} over control"))
        else:
            kept.append(v)

    # Split-half reliability ON THE COST METRIC. If the two halves of the run do not
    # agree on which variants are expensive, the ordering you are about to trust is noise.
    a = [statistics.fmean(results[v].half_tokens[0]) for v in variants if results[v].half_tokens[0]]
    b = [statistics.fmean(results[v].half_tokens[1]) for v in variants if results[v].half_tokens[1]]
    rho = _spearman(a, b) if len(a) == len(b) and len(a) >= 3 else None

    return ScreenReport(
        results=list(results.values()),
        kept=kept,
        rejected=rejected,
        ceiling=objective.ceiling,
        control_tokens=ctrl_tok,
        token_tolerance=token_tolerance,
        token_reliability=rho,
    )
