"""ABBA blocked comparison.

Why this module exists at all: on a shared or thermally-throttled host, throughput drifts
by tens of percent over the course of a run. If you measure arm A for ten minutes and then
arm B for ten minutes, you have measured the drift. We reversed five single-arm results in
a single day this way.

Three rules are baked in:

1. **Alternate in ABBA order, in contiguous blocks.** ABAB leaves a linear drift term in
   the difference; ABBA cancels it.
2. **Divide totals, never average per-block ratios.** A per-block mean is dominated by
   whichever block ran fastest. Quoting a single mid-run block once overstated our best
   arm by 10% and understated another by 21%.
3. **Report the block-level spread**, and from it how many blocks a given effect size
   actually needs. Most "improvements" below ~2% are not resolvable at the block counts
   people typically run, and no amount of extra candidates *inside* a block fixes it —
   the noise is between blocks, not within them. Add blocks, not candidates.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["ArmTotals", "ABBAResult", "abba_compare", "blocks_needed"]

# One measured candidate: (score, decode_tokens, seconds). Seconds may be 0 if you are
# ranking on the cost model rather than the wall clock.
Sample = tuple[float, int, float]
RunFn = Callable[[int], Sample]


@dataclass
class ArmTotals:
    name: str
    score: float = 0.0
    tokens: int = 0
    seconds: float = 0.0
    n: int = 0
    block_efficiency: list[float] = field(default_factory=list)

    def add(self, s: Sample) -> None:
        self.score += s[0]
        self.tokens += s[1]
        self.seconds += s[2]
        self.n += 1

    @property
    def score_per_candidate(self) -> float:
        return self.score / self.n if self.n else 0.0

    @property
    def tokens_per_candidate(self) -> float:
        return self.tokens / self.n if self.n else 0.0

    @property
    def efficiency(self) -> float:
        """Total score / total decoded tokens. Totals, not means of ratios."""
        return self.score / self.tokens if self.tokens else 0.0

    @property
    def throughput(self) -> float:
        """Total score / total seconds, when wall-clock was recorded."""
        return self.score / self.seconds if self.seconds else 0.0


@dataclass
class ABBAResult:
    a: ArmTotals
    b: ArmTotals
    blocks: int
    block_sd_pct: float | None

    @property
    def delta_efficiency(self) -> float:
        return (self.b.efficiency / self.a.efficiency - 1.0) if self.a.efficiency else 0.0

    @property
    def delta_score_per_candidate(self) -> float:
        base = self.a.score_per_candidate
        return (self.b.score_per_candidate / base - 1.0) if base else 0.0

    @property
    def delta_tokens(self) -> float:
        base = self.a.tokens_per_candidate
        return (self.b.tokens_per_candidate / base - 1.0) if base else 0.0

    def resolvable(self) -> float | None:
        """Smallest effect this run could distinguish from noise (95%), as a fraction."""
        if self.block_sd_pct is None or self.blocks < 2:
            return None
        return 1.96 * (self.block_sd_pct / 100.0) / (self.blocks ** 0.5)

    def summary(self) -> str:
        lines = [
            f"ABBA  {self.blocks} blocks   A={self.a.name}  B={self.b.name}",
            f"  n                 {self.a.n:>10d} {self.b.n:>12d}",
            f"  score/candidate   {self.a.score_per_candidate:>10.4f} "
            f"{self.b.score_per_candidate:>12.4f}   {self.delta_score_per_candidate:+.2%}",
            f"  tokens/candidate  {self.a.tokens_per_candidate:>10.2f} "
            f"{self.b.tokens_per_candidate:>12.2f}   {self.delta_tokens:+.2%}",
            f"  score/token       {self.a.efficiency:>10.4f} {self.b.efficiency:>12.4f}"
            f"   {self.delta_efficiency:+.2%}   <- the decision number",
        ]
        r = self.resolvable()
        if r is not None:
            lines.append(f"  block sd          {self.block_sd_pct:.2f}%   resolvable at 95%: +/-{r:.2%}")
            if abs(self.delta_efficiency) < r:
                lines.append(
                    "  !! the observed delta is INSIDE the noise floor. This run does not "
                    "support a direction. Add blocks (not candidates per block)."
                )
        return "\n".join(lines)


def blocks_needed(effect: float, block_sd_pct: float) -> int:
    """How many ABBA blocks to resolve `effect` (e.g. 0.02) at 95% confidence."""
    if effect <= 0:
        raise ValueError("effect must be > 0")
    return max(2, int(((1.96 * block_sd_pct / 100.0) / effect) ** 2 + 0.999))


def abba_compare(
    run_a: RunFn,
    run_b: RunFn,
    *,
    blocks: int = 4,
    per_block: int = 25,
    name_a: str = "A",
    name_b: str = "B",
    warmup: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> ABBAResult:
    """Run A/B/B/A ... in contiguous blocks over a SHARED candidate index.

    `run_a(i)` / `run_b(i)` must measure candidate index `i` and return
    (score, decode_tokens, seconds). Both arms see the same `i` values, so any
    per-index cost difference (a longer address, a rarer token) cancels instead of
    being confounded with the arm.

    `warmup` discards the first candidate of each block: the first generation after a
    context switch behaves differently from the rest, and at small block sizes that one
    sample can flip the sign.
    """
    if blocks < 2:
        raise ValueError("need at least 2 blocks for ABBA")
    A, B = ArmTotals(name_a), ArmTotals(name_b)
    total = blocks * 2 * per_block
    done = 0

    for blk in range(blocks):
        # ABBA: forward on even blocks, reversed on odd ones.
        pairs = [(A, run_a), (B, run_b)] if blk % 2 == 0 else [(B, run_b), (A, run_a)]
        for arm, fn in pairs:
            before_score, before_tok = arm.score, arm.tokens
            for k in range(per_block):
                i = blk * per_block + k
                sample = fn(i)
                done += 1
                if progress:
                    progress(done, total)
                if k < warmup:
                    continue
                arm.add(sample)
            d_tok = arm.tokens - before_tok
            if d_tok:
                arm.block_efficiency.append((arm.score - before_score) / d_tok)

    # Block-level spread of the A/B ratio: the real noise floor of this rig.
    ratios = [
        b / a
        for a, b in zip(A.block_efficiency, B.block_efficiency, strict=False)
        if a  # guard a zero-scoring block
    ]
    sd = (statistics.stdev(ratios) * 100.0) if len(ratios) >= 2 else None

    return ABBAResult(a=A, b=B, blocks=blocks, block_sd_pct=sd)
