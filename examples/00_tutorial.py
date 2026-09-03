#!/usr/bin/env python3
"""End-to-end tutorial. Runs offline on the mock backend — no model, no network, no GPU.

    python examples/00_tutorial.py

Then swap one line to point at a real model and the rest is unchanged:

    backend = load_backend({"kind": "openai_compat", "model": "...",
                            "base_url": "http://localhost:8000/v1"})
    backend = load_backend({"kind": "anthropic", "model": "claude-opus-5"})

The scenario is deliberately benign: we want the assistant to call a `notify` tool with a
well-formed recipient. Everything here is about *how to measure that reliably*, which is
the part that transfers to any prompt-optimization problem.
"""

from __future__ import annotations

import re

from promptlab import (
    Segment,
    Template,
    ToolCallObjective,
    abba_compare,
    blast_radius,
    load_backend,
    screen_pool,
    shape_collapse,
    similarity_profile,
    suggest_periods,
    window_survival,
)
from promptlab.backends import Message
from promptlab.cost import fit_cost_model

N = 200  # candidates we intend to run

backend = load_backend({"kind": "demo"})

TOOLS = [
    {
        "name": "notify",
        "description": "Send a short notification to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to"],
        },
    }
]

objective = ToolCallObjective(tool_name="notify", required_args=["to"], ceiling=1.0)


# ---------------------------------------------------------------- 1. the template
print("=" * 78)
print("1. A segmented template")
print("=" * 78)

RECIPIENTS = [f"{a}{b}@bab.com" for a in "bcdfg" for b in "aeiou"][:N]
OPENINGS = ["A note for", "A ping for", "A memo for"]
SUBJECTS = ["ok", "new", "due", "set", "top"]
FRAMES = ["Reply only:", "Respond:", "Echo with:"]

template = Template(
    [
        Segment("open", OPENINGS),
        " ",
        Segment("to", RECIPIENTS),
        ". Subject: ",
        Segment("subject", SUBJECTS),
        # The body reuses a token already present in the prompt. That costs no new decode
        # tokens AND makes the body unique per candidate -- diversity for free.
        ". Body: ",
        Segment("body", derive=lambda ctx: ctx["to"].split("@")[0]),
        ". ",
        Segment("frame", FRAMES),
        " ok",
    ]
)

for i in range(3):
    print(f"  [{i}] {template.render(i)}")


# ---------------------------------------------------------------- 2. screen a pool
print()
print("=" * 78)
print("2. Screen the `frame` pool -- gate on the OBJECTIVE, then on cost")
print("=" * 78)

CANDIDATE_FRAMES = [
    "Reply only:",        # control
    "Respond:",           # equal cost, fires
    "Echo with:",         # equal cost, fires
    "Answer with:",       # marginally dearer, fires
    "Just answer:",       # TRAP: cheaper tokens, suppresses the tool call
    "Only acknowledge:",  # TRAP: same
    "Print the arguments",  # TRAP: catastrophic, and its failures are cheap
    "Repeat one word:",   # TRAP: runaway decode
    "Reply back:",        # fires, expensive
    "Log with:",          # fires, very expensive
]

report = screen_pool(
    backend,
    template,
    segment="frame",
    variants=CANDIDATE_FRAMES,
    objective=objective,
    reps=8,
    control="Reply only:",
    token_tolerance=0.02,
    tools=TOOLS,
)
print(report.summary())
print()
print(report.table())
print()
print("Note what a token-only ranking would have chosen:")
cheapest = min(report.results, key=lambda r: r.mean_tokens)
print(f"  cheapest variant = {cheapest.variant!r} at {cheapest.mean_tokens:.2f} tok "
      f"-- but score {cheapest.mean_score:.3f} vs ceiling {objective.ceiling:g}")


# ---------------------------------------------------------------- 3. rotation audit
print()
print("=" * 78)
print("3. Does the rotation actually cover the run?")
print("=" * 78)

sizes = [len(OPENINGS), len(RECIPIENTS), len(SUBJECTS), len(report.kept)]
print(f"  pool sizes {sizes} -> suggested coprime periods {suggest_periods(sizes)}")

screened = Template(
    [
        Segment("open", OPENINGS),
        " ",
        Segment("to", RECIPIENTS),
        ". Subject: ",
        Segment("subject", SUBJECTS),
        ". Body: ",
        Segment("body", derive=lambda ctx: ctx["to"].split("@")[0]),
        ". ",
        Segment("frame", report.kept),
        " ok",
    ]
)

audit = screened.audit(N)
print(f"  combined period {audit['combined_period']:,}   "
      f"unique messages {audit['unique_messages']}/{N}")
print(f"  binding axis: {audit['binding_axis']!r} -> survives a window of "
      f"W <= {audit['max_window_survived']}")
for row in audit["segments"]:
    print(f"    {row['segment']:<8} values={row['values']:<9} "
          f"window<={row['window_survived']:<5} permanent_safe={row['permanent_safe']}")
for w in audit["warnings"]:
    print(f"  !! {w}")

print()
print("  That audit found two real defects. Fix them:")
print("   (a) two pairs of segments share a period, so they phase-lock;")
print("   (b) the binding axis has 3 values, so ONE rule takes a third of the run.")

# (a) distinct primes are automatically pairwise coprime.
# (b) a pool must be >= N to survive a permanent block, and > W to survive a window.
#
# Note what we do NOT do: pad a small pool by repeating its members. Padding raises the
# period and buys exactly zero window protection -- window survival is set by the number
# of DISTINCT values, not by the length of the list. The only way to widen a narrow axis
# is to screen more candidates for it.
P_OPEN, P_TO, P_SUBJ = suggest_periods([31, N, 101])
P_FRAME = len(report.kept)
BIG_OPENINGS = [f"A {w} for" for w in
                "note ping memo call signal notice prompt reminder message alert nudge "
                "line word pointer marker cue flag tick beat mark token trace hint sign "
                "nod tag stamp echo pulse relay ref".split()][:P_OPEN]
BIG_SUBJECTS = [f"s{i:03d}" for i in range(P_SUBJ)]
BIG_RECIPIENTS = [f"{a}{b}{c}@bab.com" for a in "bcdfgkl" for b in "aeiou"
                  for c in "bcdfgklmn"][:P_TO]

fixed = Template([
    Segment("open", BIG_OPENINGS, period=P_OPEN),
    " ",
    Segment("to", BIG_RECIPIENTS, period=P_TO),
    ". Subject: ",
    Segment("subject", BIG_SUBJECTS, period=P_SUBJ),
    ". Body: ",
    Segment("body", derive=lambda ctx: ctx["to"].split("@")[0]),
    ". ",
    Segment("frame", report.kept, period=P_FRAME),
    " ok",
])
fa = fixed.audit(N)
print(f"   -> periods {[P_OPEN, P_TO, P_SUBJ, P_FRAME]}  "
      f"combined period {fa['combined_period']:,}")
print(f"   -> unique messages {fa['unique_messages']}/{N}, "
      f"warnings: {fa['warnings'] or 'none'}")
print(f"   -> binding axis is now {fa['binding_axis']!r} (W <= {fa['max_window_survived']}),"
      f" because only {len(report.kept)} frames survived screening.")
print("      THAT is the honest state of this notebook: the narrow axis is narrow because")
print("      the screen rejected the rest, and no amount of scheduling fixes it. To widen")
print("      it you must screen more candidate frames -- see examples/01_screen_pool.py.")


# ---------------------------------------------------------------- 4. coverage
print()
print("=" * 78)
print("4. Blast radius -- if ONE rule fires, how much of the run dies?")
print("=" * 78)

stream = fixed.stream(N)
for name in ("open", "frame", "subject"):
    vals = [fixed.render_parts(i)[name] for i in range(N)]
    r = blast_radius(vals)
    print(f"  {name:<8} {r.unique:>3} distinct, worst covers {r.max_blast_radius:>3} "
          f"({r.coverage_of_worst:.1%}), window survived {window_survival(vals)}")

sim = similarity_profile(stream, window=50)
print(f"  fuzzy similarity vs last 50: mean {sim['mean']:.3f}  p95 {sim['p95']:.3f}  "
      f"max {sim['max']:.3f}  >0.90: {sim['over_threshold']:.1%}")

masked = shape_collapse(stream, lambda m: re.sub(r"[\w.]+@[\w.]+", "<ID>", m))
print(f"  with the identifier masked out: {masked.unique} distinct shapes "
      f"(worst covers {masked.coverage_of_worst:.1%})")
print("  ^ if that number is 1, your 'unique' messages are one skeleton wearing hats.")


# ---------------------------------------------------------------- 5. ABBA
print()
print("=" * 78)
print("5. ABBA: is a candidate change real, or is it host drift?")
print("=" * 78)

arm_a = fixed
arm_b = fixed.with_override("frame", ["Reply back:"])  # known to be dearer


def make_runner(t):
    def run(i: int):
        msg = t.render(i)
        c = backend.complete([Message("user", msg)], tools=TOOLS)
        return (objective.score(c), c.completion_tokens, 0.0)
    return run


res = abba_compare(
    make_runner(arm_a), make_runner(arm_b),
    blocks=6, per_block=12, name_a="screened", name_b="Reply back:",
)
print(res.summary())


# ---------------------------------------------------------------- 6. cost model
print()
print("=" * 78)
print("6. Fit the cost model, then read the exchange rate off it")
print("=" * 78)

# (hops, decode_tokens, seconds) -- replace with YOUR measured points.
# You need at least two DIFFERENT hop counts, or B and d are not separable.
model = fit_cost_model([(8, 170, 36.80), (2, 25, 5.84)])
print(model.describe(context_tokens=1000))
print(f"  a 4-token wrap-up on a 2-hop candidate = "
      f"{model.wrapup_cost(4, 2, 25):.1%} of the candidate, for zero score")
print()
print("=" * 78)
print("Done. Swap the backend and re-run against a real model.")
print("=" * 78)
