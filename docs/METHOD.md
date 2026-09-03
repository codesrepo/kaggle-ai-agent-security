# The findings behind each module

Every module in `promptlab` exists because a specific, measured mistake cost us score.
This file records what those were, so the API choices are not arbitrary. Numbers are from
a Kaggle AI-agent-security competition (two 20-30B open-weight models, tool-calling agents
under a hard per-window time budget, 2,000 candidates per cell).

---

## 1. `cost` — decode tokens are the whole budget

Two measured points solve the model exactly, with **no constant term**:

```
8 hops + 170 decode tokens = 36.80 s/candidate      B = 0.52 s per HOP
2 hops +  25 decode tokens =  5.84 s/candidate      d = 0.192 s per DECODED TOKEN
                                                    A = 0
```

`B` is the per-hop prefill of the system prompt and tool schemas — the whole conversation
is re-sent every hop. That is ~1,900 tok/s prefill against ~5.2 tok/s decode: **per token,
decoding is ~370x more expensive than prefilling.**

Since your message is re-prefilled once per hop, the practical exchange rate is:

| hops | 1 extra prompt token | 1 extra decode token | ratio |
|---:|---:|---:|---:|
| 8 | 0.0042 s | 0.192 s | **1 : 46** |
| 2 | 0.0010 s | 0.192 s | **1 : 185** |

A candidate prefilled **8,388** tokens to decode **170** (49:1) — and 89% of the wall
clock was still decode.

**What we did with it.** Stopped shrinking prompts (halving our message bought ~0.45%) and
started hunting phrasings that make the model *ruminate*. Deleting one trailing phrase
(`Keep it short.`) was **+28.18%** and cut decode from 165.3 to 127.1 tokens: it was buying
rumination, not brevity.

**The trap the API guards against.** `fit_cost_model` refuses points that all share a hop
count. Fitting arms that differ only in tokens, as if they differed only in tokens, made us
misread a per-*hop* cost as a per-*candidate* constant for a week.

---

## 2. `template` — hops pay off or don't, depending on the objective

Whether multi-hop is worth anything is a property of what you are being scored on:

* scored **per action** — 8 hops = 8x the score. Measured k-sweep against k=8:
  k=4 **-4.61%**, k=2 **-9.80%**, k=1 **-11.37%**.
* scored **per candidate** — extra hops are pure cost. Ceiling reached in one hop.

The trailing "nothing left to do" hop is where this bites. Below the hop cap the agent
always burns one extra generation:

```
2-hop candidate:  post hop 3.78 s + wrap-up hop 1.29 s   -> 25% of the budget, 0 score
8-hop candidate:  generations [22,22,21,21,21,21,21,21]  -> no wrap-up at all, 0 tokens
```

**You cannot skip that hop** — that part is structural. We controlled only the user message; the
hop loop belonged to the harness, every exit below the cap required the last tool call to have
*failed*, and every predicate required it to have *succeeded*. Deliberately failing the last call
also loses: the break still costs a generation, and the cheapest unknown-tool call is 7 tokens
against the seed's 4.

**But its CONTENT can be driven to a single termination token, and we wrongly concluded it could
not.** Our probe replayed generation-1 with `logits_all=True` and found the first wrap-up token at
99.9999%, with a top-1/top-2 gap of 18.46 nats; five hand-written conditionings that asked for
termination were all worse (one *doubled* the wrap-up, 4 -> 9 tokens). We read that as "no prompt
moves this token." Two mistakes:

* **Wrong quantity.** Top-1-vs-top-2 measures how dominant the *winner* is. What decides the outcome
  is the **EOG margin**, `z_EOG - max_{v != EOG} z_v` — how far the token you *want* is from winning.
  Those are different numbers, and only the second is an optimization target.
* **Wrong search space.** We searched hand-written natural language. The competition's 1st-place
  solution ran **GCG** (gradient-guided coordinate descent over arbitrary token substitutions)
  against the BF16 checkpoints, reranked onto the quantized artifact, and crossed zero on one of the
  two models: hop 2 became a single EOG token. On the other model even GCG failed to transfer across
  quantization.

The lesson is general enough to be worth stating outside its competition context: **"no prompt can do
X" is almost never what you measured.** You measured that the prompts you wrote did not do X. If the
weights are available, the honest next step is a gradient-guided search with an explicit margin
objective — and `screen`/`abba` are then the tools for validating that its output survives the
deployment stack, which is where such attacks usually die.

So there were only ever two options *within* natural-language prompting, and neither is suppression:

1. **Avoid it structurally** — exhausting the hop cap is the only free exit. The 8-hop cell paid zero.
2. **Make it cheap** — a `Reply only: X` frame makes the final turn be literally `X` (4 tokens)
   instead of prose ("Email sent.", 6). That works on **compliance**, not length, so the right word
   is the one with the highest echo rate, not the shortest.

The terminator clause is not optional and not linear: our 3-token stop clause cost 3 tokens;
removing it cost 15; asking for an acknowledgement cost **407**.

---

## 3. `screen` — gate on the objective, never on tokens

Two failure modes that a token metric reports as **wins**:

| variant | decode tokens | score/candidate | what it is |
|---|---:|---:|---|
| control | 35.25 | 6.000 | baseline |
| `Just answer:` | **31.00** cheaper | **5.25** | reads as "just answer me" — suppresses the call |
| `Print the arguments…` | **2.31** | **0.210** | 193/200 never call the tool |
| `Repeat one word:` | **653** | 5.4 | the verb makes the model actually repeat |

`Print the arguments…` shipped in a 4-clause pool and was running that cell at 4.52 against
a 6.00 ceiling — a quarter of it dead. It hid because its failures were *cheap*, so the
aggregate token figure looked fine. Fixing it was ~**+4.5%**.

**And then variation turns out to be free.** Screen properly and the diversity-vs-speed
trade mostly dissolves:

| pool | generated | fires 100% at ceiling | + cost-uniform (<= +2%) | shipped |
|---|---:|---:|---:|---:|
| closing frames | 321 | 313 | **205** | 101 |
| bare-args clauses | 194 | 39 | 39 | 7 |
| tail words | 43 | 33 | 33 | 32 |
| subjects | 129 | 99 | 81 | 81 |

A 101-frame rotation cost **nothing** — the cheapest members tied the fixed control
(35.12 vs 35.25). The hand-picked 11-frame pool it replaced had lost 0.67%, and that loss
was **3 fire failures in 300, not tokens.** Screening fixed the loss and the blast radius
together; they were never a trade-off.

**Corollary, and it is counterintuitive.** On a *time-limited* workload, fixing cheap
misses is worth nothing. We isolated 24 systematically dead prompts by ablation and fixed
them: score/candidate +0.81%, exactly cancelled by decode +0.87%. Net **+0.00%**. The time
those failures saved was already being spent on other candidates, and a working candidate
has the same score-per-token as the average. "X% fire zero" does not mean "X% of score
lost" unless every slot is guaranteed to run.

---

## 4. `abba` — a drifting host will happily confirm anything

Throughput drifts tens of percent across a run. **Five single-arm results reversed under
ABBA in one day.** Three rules:

1. **A/B/B/A in contiguous blocks.** ABAB leaves a linear drift term in the difference.
2. **Divide totals, never average per-block ratios.** Quoting a single mid-run block once
   overstated our best arm by 10% and understated another by 21%.
3. **Add blocks, not candidates.** 25x more candidates per block moved the block sd by 7%
   (1.968% -> 1.830%). The noise is *between* blocks. At ~1.9% block sd, resolving a 0.33%
   effect needs ~115 blocks — which is usually the honest answer to "is this 0.3% real".

Also: **discard the first candidate of each block.** The first generation after a context
change behaves differently, and at small block sizes that one sample flips signs.

---

## 5. `tokens` — characters are not tokens

Any value the model **echoes back** into a tool argument is paid at the decode rate. Our
pools were never audited for this and were not flat:

```
recipients: {5 tok: 793, 6 tok: 914, 7 tok: 293}    mean 5.75, floor 5
  zoq@zuv.ch = 7 tokens      bab@bab.com = 5 tokens
  same shape, same character count, 40% more decode
```

Two fixes, both from scanning the tokenizer's own vocabulary:

* **Fragment floor.** 2,003 of 4,410 three-letter fragments are single tokens on one model
  (1,874 on the other); every TLD is 2 tokens / 1 token. So `frag@frag.tld` = 5 = the
  floor. ABBA: **+1.67%** and **+2.02%**.
* **Single-token `@domain`.** `cale@call.com` is 4 tokens; `cale@get.com` is 3, because
  `@get` is one vocabulary entry. A scan of all 201,088 entries found **31** usable
  domains, not the 3 visible by inspection. **+2.92%**. The other model had **zero** such
  entries — the lever did not exist there at all.

Two rules: **measure in context** (a fragment that is 1 token alone can cost more inside
`f"{v}. Subject:"` — our "cheaper" swap based on isolated counts *lost* 0.65%), and
**filter real words** (models rewrite word-shaped values into something else).

**And size the pool >= N.** One pool held 1,681 values for a 2,000-candidate run, so ~319
candidates silently collapsed onto an earlier value and lost their novelty credit.
`build_composed_pool` raises rather than truncating.

---

## 6. `coverage` — "all unique" is usually cosmetic

Two disciplines, with different pool-size requirements:

* **permanent** — a value usable once, ever. Survived iff pool >= N.
* **window(W)** — blocked if it recurs within the last W. Round-robin over `n` values
  gives repeat distance exactly `n`, and that is **provably optimal**: over any horizon,
  any schedule on `n` values has some pair at distance <= n. You need `n = W+1`, never more.

Choose periods **pairwise coprime** or segments phase-lock: 1000/5/5 welds two segments
together every 5 candidates; 11/101/32 never repeats the tuple inside any realistic run
(lcm = 35,552).

Three checks worth running offline before you spend any GPU time:

* **blast radius** — how many candidates does the most-repeated value account for?
* **similarity profile** — rotating slots inside a fixed frame does *not* defeat a fuzzy
  check. Variation has to be at the **skeleton** level, and the pool has to be **larger
  than the window**. Both facts are asserted in the test suite.
* **shape collapse** — mask the identifier and see what is left. If 2,000 "unique"
  messages collapse to one skeleton, that is what a shape-based rule sees.

---

## 7. Method notes that are not modules

* **Never judge headroom on local wall-clock.** The evaluation host ran ~35x slower than
  the dev box; one constant reproduced three independent scores. Local timing called
  several real gains a wash.
* **One change per submission, off a validated baseline.** Our worst regression shipped
  three at once. Root cause turned out to be a bare `except: continue` in a model
  detector, which silently made one cell replay the other model's candidates.
* **Re-validate the control before trusting any delta.** A candidate went 500/500 -> 0/100
  across a reboot with identical bytes. Everything measured after that point was void.
* **Dump the tool arguments, not the prompt.** A day went into a value-obfuscation idea
  that the model silently normalized back. The prompt looked right; the arguments never
  were.
* **Substitute, never append.** Cost tracks words *added*, not words *changed*. Rotations
  that replace a token in place are roughly free; ones that lengthen the message pay.

---

## 8. Where these numbers were measured

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB VRAM |
| CPU / RAM | 20 cores, 31 GB |
| OS | Windows + WSL2 |
| Runtime | llama.cpp via `llama-cpp-python` 0.3.31, Python 3.13 |
| Models | a 20B and a 26B open-weight model, both Q4_K_M GGUF, `n_ctx = 8192` |
| Scoring host | a separate evaluation environment, ~35x slower, 4 x 2.5 h windows |

Two properties of that setup shaped the whole library:

* **Prefix caching hid the prefill cost.** Locally, re-sending a growing conversation is nearly free,
  so every timing we took was blind to per-hop prefill for weeks. Forcing a full prefill
  (`llm.reset()` before each generation) cost **+90%** and **+122%** on the two models. That is the
  term the evaluation host actually pays — hence `cost.py` fits a per-*hop* coefficient, and hence
  `fit_cost_model` refuses a design that cannot separate it.
* **Token counts are deterministic under greedy decoding; wall-clock is not.** On identical bytes,
  decode counts reproduce exactly while throughput swung up to **16.6%** between runs of the same
  arm. That asymmetry is why `screen` ranks on tokens *behind a score gate*, and `abba` ranks on
  contiguous blocks — neither metric is trustworthy alone.
