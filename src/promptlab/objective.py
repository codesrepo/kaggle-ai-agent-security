"""What "a good candidate" means.

The single most expensive mistake available in this kind of work is to rank prompt
variants by **token cost**. Two failure modes make a bad variant look like a win:

* the *cheap trap* — a phrasing that stops the model calling the tool at all. It decodes
  fewer tokens than the control, so a token metric scores it as an improvement. One such
  clause sat in a shipped pool of ours and was zeroing a quarter of a cell for weeks; it
  hid precisely because its failures were cheap, so the aggregate token count looked fine.
* the *runaway* — a phrasing that fires but occasionally decodes 20x. A mean hides it; a
  ceiling check plus a max does not.

So an Objective returns a **score**, and every screening decision gates on score first
and cost second. `ceiling` is the score a perfect candidate gets; anything below it is a
defect to explain, not a number to average away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .backends.base import Completion

__all__ = ["Objective", "ToolCallObjective", "CallableObjective"]


@runtime_checkable
class Objective(Protocol):
    ceiling: float

    def score(self, completion: Completion) -> float: ...


@dataclass
class ToolCallObjective:
    """Score a completion by whether it produced the tool call you wanted.

    tool_name       : the tool that must be called.
    required_args   : argument names that must be present and non-empty.
    arg_predicates  : optional {arg_name: callable(value) -> bool} for stricter checks.
    per_call        : score per qualifying call. Set ceiling = per_call * expected_calls
                      when a candidate can legitimately produce several.
    max_calls       : cap on counted calls (defaults to 1 — most objectives are per-cell,
                      not per-call; raise it only if extra calls genuinely score).
    """

    tool_name: str
    required_args: Sequence[str] = ()
    arg_predicates: dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    per_call: float = 1.0
    max_calls: int = 1
    ceiling: float = 1.0

    def __post_init__(self) -> None:
        if self.ceiling == 1.0 and self.per_call * self.max_calls != 1.0:
            self.ceiling = self.per_call * self.max_calls

    def _ok(self, call: Any) -> bool:
        if call.name != self.tool_name:
            return False
        for arg in self.required_args:
            val = call.arguments.get(arg)
            if val is None or (isinstance(val, str) and not val.strip()):
                return False
        for arg, pred in self.arg_predicates.items():
            if not pred(call.arguments.get(arg)):
                return False
        return True

    def score(self, completion: Completion) -> float:
        good = sum(1 for c in completion.tool_calls if self._ok(c))
        return self.per_call * min(good, self.max_calls)


@dataclass
class CallableObjective:
    """Wrap any function. Use when the objective is not "did it call a tool"."""

    fn: Callable[[Completion], float]
    ceiling: float = 1.0

    def score(self, completion: Completion) -> float:
        return float(self.fn(completion))
