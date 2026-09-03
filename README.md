# Kaggle · AI Agent Security — Multi-Step Tool Attacks

**#11 of 4,186 teams** on the private leaderboard (30.040), from **#9 public** (136.450).

**Write-up: [Measure What You Can, Survive What You Cannot](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/11th-place-solution-measure-what-you-can-survive)** — the full solution, on Kaggle.

This repository holds the measurement harness that result was built with.

```
src/promptlab/          the harness — model-agnostic, stdlib-only core
examples/               runnable end-to-end, offline, no model required
docs/METHOD.md          the measured findings behind each module
```

> **Scope.** No exploits, payloads, or jailbreaks are published here. What is published is the
> *methodology* — how to measure prompt variants without fooling yourself, and how to reason about
> coverage when you cannot observe the thing you are being scored against. The example objective
> throughout is a benign `notify` tool call.

---

## The problem, in one paragraph

Half the score came from a guardrail we could measure a thousand times a day. The other half came
from a guardrail we would never see, whose only certain property was that it was *stricter*. Both
halves ran under a hard wall-clock budget, so every attack was really a throughput problem wearing a
security costume. The write-up is the story of measuring the first half precisely enough to afford
insurance on the second.

Three findings did most of the work:

- **Decode tokens are the entire budget.** Fitted on our own rig, `t = 0.52·hops + 0.192·decode_tok`
  with no constant term. Prefill runs ~370× faster per token than decode, so one decoded token costs
  what ~46 prompt tokens cost at 8 hops, ~185 at 2. Prompt length is nearly free; *rumination* is not.
- **The private guardrail was invisible, but its runtime was not.** A blocked attack short-circuits
  and the submission finishes early; an allowed one runs the full budget. That turns wall-clock into
  a one-bit readout on a hidden defense, at the cost of one submission slot and no score.
- **Variation is free if you screen for it.** Ranking prompt variants by token cost *selects for
  broken ones*, because a phrasing that stops calling the tool decodes fewer tokens than one that
  works. Gate on the objective first, cost second, and a 100-value rotation costs the same as the
  single fixed value it replaced.

Full detail, with numbers: [**the write-up**](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/writeups/11th-place-solution-measure-what-you-can-survive).

---

## The harness

```bash
git clone https://github.com/codesrepo/kaggle-ai-agent-security.git
cd kaggle-ai-agent-security
pip install -e .

python examples/00_tutorial.py    # end-to-end, offline. No model, no key, no GPU.
pytest                            # 62 tests, all offline
```

| module | what it does |
|---|---|
| `template` | Segmented prompts. Coprime rotation, phase-lock detection, per-segment window/permanent-block analysis. |
| `screen` | Pool screening. Interleaved repetitions, objective-first gating, split-half reliability on the cost metric. |
| `abba` | Drift-cancelling A/B/B/A comparison. Totals not means-of-ratios; reports the noise floor and blocks needed. |
| `cost` | Fits `t = B·hops + d·decode_tokens` from your own timings, and reads the prompt/decode exchange rate off it. |
| `tokens` | Token-floor pool building. Characters are not tokens; audit pools *in context*. |
| `coverage` | Blast radius, window survival, fuzzy-similarity profile, shape collapse. All offline, no model calls. |
| `backends` | One small protocol. Mock (offline, scriptable), OpenAI-compatible, Anthropic. |

### Point it at a model

The backend protocol needs three things: complete a chat, report decoded tokens, tokenize a string.

```python
from promptlab import load_backend

be = load_backend({"kind": "demo"})          # offline, deterministic, ships the known traps

be = load_backend({"kind": "openai_compat",  # vLLM, llama.cpp, Ollama, TGI, LM Studio, OpenRouter
                   "model": "Qwen/Qwen3-8B",
                   "base_url": "http://localhost:8000/v1"})

be = load_backend({"kind": "anthropic",      # credentials from the environment
                   "model": "claude-opus-5", "effort": "low"})
```

A local OpenAI-compatible server is the right target for real screening work — you need tens of
thousands of generations, and you want the tokenizer. Pass
`tokenizer=AutoTokenizer.from_pretrained(...)` to unlock `promptlab.tokens`.

### The trap the whole library exists for

```
variant               score   fire     tok    max  verdict
Reply only:           1.000 100.0%   35.00     35  KEEP
Respond:              1.000 100.0%   35.00     35  KEEP
Just answer:          0.725  72.5%   29.07     31  below ceiling — CHEAPER than control, classic trap
Repeat one word:      0.475  47.5%  352.70    653  below ceiling
Print the arguments   0.050   5.0%    3.15     25  below ceiling — CHEAPER than control, classic trap
```

The cheapest variant here scores **5%**. A clause of exactly this shape sat in a shipped pool of ours
for weeks, zeroing a quarter of a cell — hidden precisely *because* its failures were cheap enough to
keep the aggregate token count looking healthy. `screen_pool` gates on score first and cost second,
and the test suite asserts it rejects every trap above.

Three rules the library enforces, each of which cost us score to learn:

1. **Gate on the objective, then on cost.** A variant that does not fire is not cheap, it is broken.
2. **Screen wide and cheap, then confirm.** `reps=8` separates "works" from "often fails"; it does
   not bound a fire rate. Confirm survivors at `reps>=40`.
3. **Add blocks, not candidates.** A/B noise lives *between* blocks, not within them.
   `blocks_needed(effect, block_sd)` tells you what resolving a given effect would actually take.

---

## Results are model-specific; the method is not

Worth stating plainly, because it is the easiest thing to get wrong. The harness is model-agnostic —
it reads whatever tokenizer the configured backend exposes. **The findings are not.** Tokenization
is a property of one model's vocabulary: the single-token `@domain` lever in the write-up was worth
+2.92% on one model and *did not exist at all* on the other, which had zero such entries. Rebuild
every pool, in context, against the model you will actually deploy against.

## Notes

- `promptlab` is the import name here. The name is taken on PyPI by an unrelated project, so this is
  not published there — install from source.
- The `anthropic` backend counts thinking tokens as decode tokens, because that is what they are.
  Move them with `effort`, not with token accounting.

## License

[**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) — see [LICENSE](LICENSE).
Use it, adapt it, ship it commercially; just credit the source and say what you changed.
