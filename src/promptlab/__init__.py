"""promptlab — measure prompt variants honestly, then rotate them safely.

A small harness for a specific problem: you have a prompt with several slots, many
candidate phrasings for each slot, and a metric that is easy to fool. It gives you

  * `template`  — segmented prompts with provably-optimal coprime rotation
  * `screen`    — pool screening that gates on the objective before cost
  * `abba`      — drift-cancelling A/B with an explicit noise floor
  * `cost`      — the hops/decode cost model, fitted from your own timings
  * `tokens`    — token-floor pool building (characters are not tokens)
  * `coverage`  — offline blast-radius, window and fuzzy-similarity analysis
  * `backends`  — one small protocol; mock, OpenAI-compatible, and Anthropic included

Start with `examples/00_tutorial.py`, which runs end to end on the mock backend.

Where it came from, and every number behind it:
https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/11th-place-solution-measure-what-you-can-survive
"""

from __future__ import annotations

from .abba import ABBAResult, abba_compare, blocks_needed
from .backends import Backend, Completion, Message, ToolCall, load_backend
from .cost import CostModel, fit_cost_model
from .coverage import blast_radius, shape_collapse, similarity_profile, window_survival
from .objective import CallableObjective, Objective, ToolCallObjective
from .screen import ScreenReport, screen_pool
from .template import Segment, Template, next_prime, suggest_periods
from .tokens import audit_pool, build_composed_pool, single_token_fragments

__version__ = "0.1.0"

__all__ = [
    "Backend", "Completion", "Message", "ToolCall", "load_backend",
    "Segment", "Template", "next_prime", "suggest_periods",
    "Objective", "ToolCallObjective", "CallableObjective",
    "screen_pool", "ScreenReport",
    "abba_compare", "ABBAResult", "blocks_needed",
    "CostModel", "fit_cost_model",
    "audit_pool", "single_token_fragments", "build_composed_pool",
    "blast_radius", "window_survival", "similarity_profile", "shape_collapse",
]
