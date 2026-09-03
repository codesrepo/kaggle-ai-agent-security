"""Token-floor pool building: characters are not tokens.

Any value that the model **echoes back** — into a tool argument, a field, a citation — is
paid at the decode rate, not the prefill rate. On a cell that decodes 32 tokens in total,
one extra token in the echoed value is 3% of throughput. So a pool of "identical-looking"
values is usually nothing of the sort:

    zoq@zuv.ch   -> 7 tokens
    bab@bab.com  -> 5 tokens        same shape, same character count, 40% more decode

We shipped pools with a 5/6/7-token spread for weeks without noticing, because they looked
uniform. Flattening them to the floor was worth +1.7% to +2.9% on its own.

Two levers, in increasing order of specificity:

* `single_token_fragments` — find the fragments that encode to exactly one token, so a
  composed value like `frag@frag.tld` sits at the structural floor.
* `single_token_suffixes` — some tokenizers have `@domain` as one vocabulary entry, which
  removes another token. This is highly model-specific: one of our two target models had
  31 such entries out of a 201k vocabulary, and the other had **zero**. Always check.

Two rules learned the hard way:

* **Measure in context.** A fragment that is one token alone can cost more inside
  `f"{frag}. Subject:"`. Our "cheaper" swap based on isolated counts *lost* 0.65%.
* **Filter real words.** Models rewrite value-looking-like-a-word into something else, so
  a word-shaped pool member silently produces a different argument than the one you built.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .backends.base import Backend, TokenizerUnavailable

__all__ = [
    "PoolAudit",
    "audit_pool",
    "single_token_fragments",
    "single_token_suffixes",
    "build_composed_pool",
]


@dataclass
class PoolAudit:
    histogram: dict[int, int]
    floor: int
    mean: float
    worst: list[tuple[str, int]]
    in_context: bool

    @property
    def waste_per_candidate(self) -> float:
        """Decode tokens per candidate thrown away versus a floor-flat pool."""
        return self.mean - self.floor

    def summary(self) -> str:
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(self.histogram.items()))
        ctx = "in context" if self.in_context else "in ISOLATION (re-check in context!)"
        lines = [
            f"pool token histogram ({ctx}): {{{hist}}}",
            f"  floor {self.floor}   mean {self.mean:.2f}   waste {self.waste_per_candidate:.2f} tok/candidate",
        ]
        if self.worst:
            lines.append("  worst offenders: " + ", ".join(f"{v}={n}" for v, n in self.worst[:5]))
        if self.waste_per_candidate > 0.25:
            lines.append(
                "  -> flattening this pool to the floor is a real, free speed-up."
            )
        return "\n".join(lines)


def audit_pool(
    backend: Backend,
    pool: Sequence[str],
    *,
    context: Callable[[str], str] | None = None,
    top_k: int = 10,
) -> PoolAudit:
    """Token-count histogram for a pool.

    `context` should wrap a value exactly as the prompt does, e.g.
    ``lambda v: f"A note for {v}. Subject:"``. Pass it. Isolated counts mislead —
    the surrounding characters change how a value tokenizes.
    """
    counts: list[tuple[str, int]] = []
    if context is not None:
        # Cost of the value in context = full-string cost minus the frame's own cost.
        base = backend.count_tokens(context(""))
        for v in pool:
            counts.append((v, backend.count_tokens(context(v)) - base))
    else:
        for v in pool:
            counts.append((v, backend.count_tokens(v)))

    hist = Counter(n for _, n in counts)
    floor = min(hist) if hist else 0
    mean = sum(n for _, n in counts) / len(counts) if counts else 0.0
    worst = sorted(counts, key=lambda kv: -kv[1])[:top_k]
    return PoolAudit(
        histogram=dict(hist), floor=floor, mean=mean, worst=worst, in_context=context is not None
    )


def single_token_fragments(
    backend: Backend,
    candidates: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    limit: int | None = None,
) -> list[str]:
    """Keep only the candidates that encode to exactly one token."""
    bad = set(exclude)
    out: list[str] = []
    for c in candidates:
        if c in bad:
            continue
        try:
            if len(backend.token_ids(c)) == 1:
                out.append(c)
        except TokenizerUnavailable:
            raise
        if limit and len(out) >= limit:
            break
    return out


def single_token_suffixes(
    backend: Backend, prefix: str = "@", *, min_len: int = 3, plausible: Callable[[str], bool] | None = None
) -> list[str]:
    """Vocabulary entries that begin with `prefix` — e.g. a whole `@domain` in one token.

    Scans the model's own vocabulary rather than guessing. Highly model-specific: expect
    anywhere from zero to a few dozen usable entries, and expect most of them to be code
    artefacts you would never want in a real value.
    """
    entries = [v for v in backend.vocab() if v.startswith(prefix) and len(v) >= min_len]
    if plausible is not None:
        entries = [v for v in entries if plausible(v)]
    return sorted(set(entries))


def build_composed_pool(
    backend: Backend,
    locals_: Sequence[str],
    domains: Sequence[str],
    tlds: Sequence[str],
    *,
    size: int,
    context: Callable[[str], str] | None = None,
    word_filter: Callable[[str], bool] | None = None,
) -> tuple[list[str], PoolAudit]:
    """Build a `local@domain.tld` pool at the token floor, sized >= `size`.

    The pool must be at least as large as your candidate count if you want to survive a
    "usable once, ever" rule. Getting that wrong is quiet and expensive: one of our pools
    held 1,681 values for a 2,000-candidate run, so ~319 candidates silently collapsed
    onto an earlier value and lost their novelty credit.
    """
    if word_filter is not None:
        locals_ = [v for v in locals_ if word_filter(v)]

    built: list[str] = []
    seen: set[str] = set()
    # Advance both indices so the pairing does not repeat after one cycle of the shorter
    # list -- a stride that resets is the usual cause of an undersized pool.
    nd = len(domains) * len(tlds)
    if not locals_ or not nd:
        raise ValueError("need non-empty locals, domains and tlds")
    i = 0
    while len(built) < size and i < size * 8:
        lp = locals_[i % len(locals_)]
        dom = domains[(i // len(locals_) + i) % len(domains)]
        tld = tlds[(i * 7 + i // len(locals_)) % len(tlds)]
        v = f"{lp}@{dom}{tld}"
        if v not in seen:
            seen.add(v)
            built.append(v)
        i += 1

    if len(built) < size:
        raise ValueError(
            f"could only build {len(built)} unique values, need {size}: widen the "
            f"component pools ({len(locals_)} x {len(domains)} x {len(tlds)})"
        )
    return built, audit_pool(backend, built, context=context)
