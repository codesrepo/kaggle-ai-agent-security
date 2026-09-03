"""The cost model: what an agent run is actually charged for.

For a tool-using agent under a time budget, wall-clock per candidate is well described by

    t  =  B * hops  +  d * decode_tokens

with **no constant term**. `B` is the per-hop prefill of the system prompt and tool
schemas — the conversation is re-sent in full on every hop — and `d` is the marginal cost
of one decoded token.

Two consequences, and they are not intuitive:

1. **Prompt length is nearly free; output length is the whole budget.** Prefill runs
   ~350x faster per token than decode. Because your message is re-prefilled once per hop,
   one extra decoded token costs about as much as `hops * (prefill_rate * d)` extra prompt
   tokens — around 46 at 8 hops, around 185 at 2. Shrinking the prompt is almost never
   where the win is. Stopping the model from *thinking out loud* almost always is.

2. **Hops carry the fixed cost, not candidates.** A trailing "nothing left to do" turn
   costs a full hop plus its own tokens. On a 2-hop candidate that wrap-up is >20% of the
   budget for zero score. Saturating the hop cap removes it entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["CostModel", "fit_cost_model", "CostPoint"]

# (hops, decode_tokens, seconds)
CostPoint = tuple[int, int, float]


@dataclass(frozen=True)
class CostModel:
    seconds_per_hop: float       # B — prefill of the whole context, once per hop
    seconds_per_token: float     # d — marginal decode

    def predict(self, hops: int, decode_tokens: int) -> float:
        return self.seconds_per_hop * hops + self.seconds_per_token * decode_tokens

    def throughput(self, score: float, hops: int, decode_tokens: int) -> float:
        """Score per second — the objective for any time-limited run."""
        t = self.predict(hops, decode_tokens)
        return score / t if t else 0.0

    def prefill_tokens_per_second(self, context_tokens: int) -> float:
        """Implied prefill rate, given how big the re-sent context is."""
        return context_tokens / self.seconds_per_hop if self.seconds_per_hop else float("inf")

    def token_exchange_rate(self, hops: int, context_tokens: int) -> float:
        """How many PROMPT tokens equal one DECODE token, at this hop count.

        One prompt token is re-prefilled once per hop, so it costs
        `hops / prefill_rate`; one decode token costs `d`.
        """
        rate = self.prefill_tokens_per_second(context_tokens)
        if not rate:
            return float("inf")
        return self.seconds_per_token / (hops / rate)

    def wrapup_cost(self, wrapup_tokens: int, hops: int, decode_tokens: int) -> float:
        """Fraction of a candidate spent on a trailing hop that scores nothing."""
        total = self.predict(hops, decode_tokens)
        return self.predict(1, wrapup_tokens) / total if total else 0.0

    def describe(self, context_tokens: int = 1000) -> str:
        rate = self.prefill_tokens_per_second(context_tokens)
        return (
            f"t = {self.seconds_per_hop:.3f}*hops + {self.seconds_per_token:.4f}*decode_tok\n"
            f"  prefill ~{rate:,.0f} tok/s   decode ~{1/self.seconds_per_token:,.1f} tok/s"
            f"   ({self.seconds_per_token * rate:,.0f}x per token)\n"
            f"  1 decode token == {self.token_exchange_rate(8, context_tokens):.0f} prompt tokens"
            f" at 8 hops, {self.token_exchange_rate(2, context_tokens):.0f} at 2 hops"
        )


def fit_cost_model(points: Sequence[CostPoint]) -> CostModel:
    """Least-squares fit of t = B*hops + d*tokens through the origin.

    Two points with *different hop counts* are enough — and you need that difference.
    Fitting arms that differ only in tokens, as if they differed only in tokens, is how
    a per-hop cost gets misread as a per-candidate constant. We made exactly that mistake
    and it produced a plausible, wrong "fixed cost per candidate" for a week.
    """
    pts = [(float(h), float(c), float(t)) for h, c, t in points]
    if len(pts) < 2:
        raise ValueError("need >= 2 points")
    if len({h for h, _, _ in pts}) < 2:
        raise ValueError(
            "all points share a hop count; B and d are not separable. "
            "Include arms that differ in HOPS, not only in tokens."
        )

    shh = sum(h * h for h, _, _ in pts)
    shc = sum(h * c for h, c, _ in pts)
    scc = sum(c * c for _, c, _ in pts)
    sht = sum(h * t for h, _, t in pts)
    sct = sum(c * t for _, c, t in pts)

    det = shh * scc - shc * shc
    if abs(det) < 1e-12:
        raise ValueError("degenerate design: hops and tokens are collinear across points")

    B = (sht * scc - sct * shc) / det
    d = (shh * sct - shc * sht) / det
    return CostModel(seconds_per_hop=B, seconds_per_token=d)
