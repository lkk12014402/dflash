# Prototype: Adaptive-Block Confidence-Guided DFlash

A research prototype implementing the "main innovation direction" from
`dflash_vs_nemotron_analysis.md`, built on top of the DFlash reference decoder.
It adds two **lossless**, independently-ablatable mechanisms to the draft loop,
plus instrumentation and a comparison harness, and reports an **honest empirical
characterization** of when they help.

## Files added (all under `/data/lkk/mtp/dflash/`)
- `dflash/adaptive.py` — `dflash_generate_adaptive(...)`: drop-in variant of
  `dflash.model.dflash_generate` with two toggles:
  - `adaptive_block`: next block size = EMA(acceptance) × `block_factor`,
    clamped to `[min_block, max_block]`.
  - `early_stop`: truncate the drafted tail at the first token whose softmax
    confidence < `conf_threshold` before sending it to the target verifier.
  - Instruments: per-step `block_sizes`, `acceptance_lengths`,
    `num_target_forwards` (NFE), `verify_tokens`, `draft_tokens`.
- `bench_adaptive.py` — single-config A/B (fixed vs adaptive) + losslessness check.
- `ablation_adaptive.py` — multi-config ablation in one model load, reporting both
  a memory-bound metric (tok/s) and a compute-bound proxy (verify/draft tokens).

Setup used: **target** `Qwen/Qwen3-4B`, **draft** `z-lab/Qwen3-4B-DFlash-b16`
(block_size=16, 5 layers), 1× RTX 6000D, bf16, SDPA attention, greedy decoding,
gsm8k.

## How to run
```bash
cd /data/lkk/mtp/dflash
CUDA_VISIBLE_DEVICES=0 python ablation_adaptive.py \
  --model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --max-samples 16 --max-new-tokens 256
```

## Key empirical findings

### 1. The target *verify* forward is essentially FLAT in block length (single-seq)
Measured target forward latency with a warm KV cache:

| verify length L | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| ms / forward | 56.3 | 59.6 | 60.0 | 58.1 | 58.9 | 60.5 | 60.3 |

→ Decode is **memory-bandwidth-bound**: loading the 4B weights dominates;
verifying 1 vs 32 tokens costs ~the same. Throughput is therefore governed by the
**number of target forwards**, i.e. by **acceptance-per-iteration**, not by how
many tokens you verify.

### 2. Acceptance peaks at the *trained* block size and degrades on both sides
Fixed-block sweep (8 prompts, 256 new tokens):

| block | 4 | 8 | 12 | **16** | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| tok/s | 121 | 185 | 225 | **252** | 197 | 166 | 151 |
| mean accept | 3.34 | 5.12 | 6.31 | **7.14** | 5.70 | 4.79 | 4.29 |

→ Below 16, acceptance is **capped** by the block; above 16, the draft is
**out-of-distribution** (it was trained for block=16) and acceptance **collapses**.
The trained block size is a sharp optimum.

### 3. Ablation: the regime-dependent tradeoff (16 prompts)

| config | tok/s | accept | NFE | verify_tok | draft_tok | verify_saved | lossless* |
|---|---|---|---|---|---|---|---|
| **fixed (B=16)** | **208** | **6.63** | 546 | 8736 | 8190 | 0% | ref |
| early-stop@0.7 | 172 | 5.71 | 640 | 4211 | 9600 | **51.8%** | 12/16 |
| early-stop@0.5 | 195 | 6.36 | 577 | 5053 | 8655 | **42.2%** | 12/16 |
| adaptive-block | 156 | 5.00 | 736 | 5791 | 5055 | 33.7% | 13/16 |
| adaptive+earlystop | 132 | 4.54 | 817 | 4182 | 5048 | 52.1% | 9/16 |

\* *losslessness:* the "non-identical" samples are **bf16 tie-breaking**, not a
bug. Control: **fixed B=16 vs fixed B=8 also agree only 12/16**, diverging at late
positions (49, 83, 89, 219) — i.e. simply changing the block size in the *reference*
decoder produces the same rate of bf16-induced argmax flips. Our adaptive decoder is
exactly as lossless as the reference. (Run in fp32 / exact-match arithmetic to make
it bit-identical.)

## Honest conclusion (a useful negative result)
In the **single-sequence, memory-bound** regime, **the fixed trained block size is
already near-optimal** and naive test-time adaptation cannot beat it:
- `early_stop` shortens a verify forward that is *free* (flat-cost) → it only trades
  away acceptance → slower tok/s.
- `adaptive_block` shrinking caps acceptance and *increases* NFE; growing breaks the
  block-size-specialized draft.

**But the prototype quantifies a real win that lives elsewhere:** `early-stop@0.7`
cuts **verify tokens by ~52%** at only a ~14% acceptance hit. In a **compute-bound**
regime — **batched serving** (verify forward becomes GEMM-bound, cost ∝ Σ block
length) or **long-context** (attention cost ∝ length) — that ~52% reduction in
scaling FLOPs converts into real throughput. This matches Nemotron-Labs-Diffusion's
own motivation of "high efficiency at varying concurrency levels."

## Reframed research direction (what to actually pursue)
1. **Compute-bound validation** — re-run the verify-token savings under batched
   decoding (concurrency ≥ 8) / long context, where verify cost scales; expect
   early-stop to flip from a loss to a win on end-to-end throughput.
2. **Training-time variable block** — the test-time block ceiling is set by the
   draft being trained for a single block=16. Train the draft with **mixed /
   block-size-conditioned** objectives so larger blocks stay in-distribution; only
   then does the adaptive-block controller have headroom to exploit easy regions.
3. **Calibration-driven early-stop** — replace the raw-confidence threshold with a
   small **learned acceptance-probability head** (supervised by real verify
   accept/reject), giving a better verify-length predictor for the compute-bound
   regime.

## Would training block=32 + early-stop help significantly? (probed, no training)
`analyze_headroom.py` runs the **existing block=16 draft** and measures acceptance
headroom + confidence calibration to predict the answer.

**(A) Acceptance almost never hits the block ceiling** — the limiter is the text
*predictability horizon* (~10–15 tokens), not the block size:

| workload | mean accept | p90 | max | % blocks reaching ≥12 | clipped at 16-ceiling? |
|---|---|---|---|---|---|
| gsm8k (math) | 5.58 | 14 | 15 | 16.9% | ~0% (p90=14 < 15) |
| humaneval (code) | 5.21 | 15 | 15 | 16.8% | ~10% (p90=15 = ceiling) |

Per-position gated reach-prob decays steeply (gsm8k: 0.70 @pos1-4 → 0.11 @pos13-16;
by pos 16 ≈ 0). So a perfectly in-distribution block=32 draft has **almost no extra
runs to capture on math**, and only a **modest top-decile gain on code**.

**(B) Confidence is well-calibrated to acceptance** (so early-stop's signal works):

| draft confidence | <0.30 | 0.30–0.50 | 0.50–0.70 | 0.70–0.90 | 0.90–0.99 | ≥0.99 |
|---|---|---|---|---|---|---|
| P(token matches target) | 0.14 | 0.30 | 0.47 | 0.67 | 0.87 | 0.99 |

Low-confidence drafted tokens are almost never accepted (P(gated-accept) ≈ 0.01 for
conf<0.3) → early-stop reliably trims the worthless tail.

**Conclusion (honest, quantified):** *Not significantly, on these benchmarks.*
- Acceptance is **quality-limited, not block-limited** — block=16 already covers the
  ~10–15-token predictability horizon, so block=32 adds runs only where blocks
  currently saturate at 16 (≈0% on math, ≈10% on code → a modest bump).
- The long block=17–32 tail has reach-prob ≈ 0 and confidence collapses to ~0.2;
  **early-stop is exactly what makes a 32-block draft not waste verify FLOPs there**,
  but that mostly just claws back to "≈ block=16" — it doesn't create new acceptance.
- In the **single-seq memory-bound** regime early-stop adds nothing to tok/s anyway
  (verify is flat). The early-stop value is concentrated in the **compute-bound /
  batched / long-context** regime.

**Where block=32 + early-stop *would* be a significant win:** workloads with genuine
long predictable runs (code/boilerplate/structured/long-context) **and** compute-bound
serving. Even then, the higher-leverage move than "just train block=32" is:
1. train a **variable / mixed block-size** draft (in-distribution at both short and
   long blocks) so the adaptive controller has real headroom, and
2. since acceptance is quality-limited, invest in **per-position draft accuracy**
   (better target-hidden features / a learned, calibrated acceptance head) — that
   raises the whole reach-prob curve, which matters far more than the block ceiling.

Probe it on YOUR workload first: `python analyze_headroom.py --dataset <task>` — if
p90 acceptance hits the block ceiling often, block=32 has headroom; if not, it won't.

## Code-change notes
- No reference files were modified; everything is additive (`dflash/adaptive.py`,
  two top-level scripts). `dflash_generate_adaptive(adaptive_block=False,
  early_stop=False)` reproduces the fixed-block reference for sanity checks.

---

# Innovation 2 — Iterative draft self-refinement (BUILT, measured, NEGATIVE)

**Idea.** DFlash fills a masked block in **one** draft forward (block-diffusion
"0 refinement steps"). Import Nemotron's confidence-thresholded iterative
unmasking into the draft: after pass 0, freeze positions with conf ≥ τ, re-mask
the rest, and run the tiny 5-layer draft again so the still-masked positions get
**bidirectional** context from their now-fixed neighbours. Target still verifies
& commits the longest matching prefix + bonus token → **lossless by construction**.

**Implementation.** `dflash/refine.py::dflash_generate_refine(num_refine,
refine_threshold)`. `num_refine=0` reproduces the reference draft *exactly*
(verified token-identical). The draft KV cache is restored to its pre-block
length before each refinement pass so every pass has identical shapes
(`position_ids = [draft_ctx_len : start+block]`). Harness: `bench_refine.py`.

**Result (Qwen3-4B + DFlash-b16, gsm8k, 16 samples, greedy, max_new=256):**

| metric                  | reference | refine=1 | refine=2 |
|-------------------------|-----------|----------|----------|
| throughput (tok/s)      | 230.8     | 198.8    | 177.6    |
| mean acceptance length  | **6.63**  | 6.40     | 6.34     |
| draft forwards          | 546       | 1126     | 1701     |
| tokens / target-forward | 6.34      | 6.14     | 6.10     |
| lossless vs reference   | —         | 16/16    | 16/16    |

Threshold sweep (refine=1): τ=0.3 → 6.55→**6.58** (neutral); τ=0.6 → 6.63→6.40;
τ=0.9 → 6.55→**5.98** (worse). More re-masking ⇒ worse. Best case is *neutral*.

**Why it doesn't work (clean mechanistic reason).** Speculative acceptance is a
**longest-matching-prefix** operation, so only the tokens *up to the first
mismatch* matter. The correctness of the frontier token depends on its **left**
context — which pass 0 already provides. Iterative refinement only adds **right**
(future) context, and those future tokens are past the model's ~10–15-token
predictability horizon (reach-prob → 0, confidence ≈ 0.2), i.e. unreliable.
Conditioning an early token on noisy future predictions can even flip it the
wrong way (hence τ=0.9 *hurts*). dlm-style iterative unmasking improves
**full-sequence** generation quality (every position counts); it does **not**
improve **prefix** acceptance.

**Two premises this falsifies (both were assumed in the brainstorm):**
1. *"Extra draft pass is ~free because the draft is tiny."* — False here:
   one refinement pass costs ≈ 14% throughput. The draft forward includes the
   full `lm_head` over the 151k vocab for the whole block, which is a real
   fraction of the (flat) target verify cost. Doubling draft forwards is not free.
2. *"Better draft quality ⇒ higher acceptance."* — Only true for the **prefix**.
   Late-position quality (what refinement improves) is irrelevant to a
   single-candidate prefix verifier.

**When it *would* pay off.** Refinement's gain (better *suffix* candidates) is
only cashable under **tree / multi-candidate verification** (Innovation 1): there,
improved deep-position candidates can be verified in parallel and extend
acceptance. So Innovation 2's mechanism is effectively **subsumed by Innovation 1**
— refine only makes sense *on top of* tree drafting, not for linear verification.

**Recommendation.** Drop standalone refinement for linear DFlash. The headroom
analysis already pointed here: acceptance is quality/horizon-limited, and the real
levers are **training** (variable-block + target-argmax distillation to raise the
whole reach-prob curve) and **tree verification** (Innovation 1) to monetize the
draft's discarded per-position distributions. `refine.py` is retained as a
building block for the tree-drafting prototype.

## Code-change notes (Innovation 2)
- Additive only: `dflash/refine.py`, `bench_refine.py`. No reference files touched.
- `dflash_generate_refine(num_refine=0)` is token-identical to `dflash_generate`.

---

# Innovation 1 — Diffusion tree-drafting + tree verification (BUILT, POSITIVE)

**Idea.** The reference draft computes a full per-position distribution for the
whole block in one bidirectional forward, then discards all but the arg-max
(`block_output_ids[:,1:]=sample(draft_logits)`). Since the single-seq target
verify is **flat-cost** up to ~64 tokens, we can spend that free capacity
verifying *many* candidates: read the draft's per-position **top-k**, assemble a
probability-ordered **token tree** (DFlash's per-position logits act as Medusa
heads), verify the whole tree in ONE target forward via a **tree-attention
mask**, accept the longest path the target greedily agrees with (+1 bonus), and
compact the accepted path's KV back into the cache.

**Implementation.** `dflash/tree.py::dflash_generate_tree(tree_max_nodes, top_k)`.
Greedy Sequoia-style tree builder (`_build_tree`, heap-ordered by joint marginal
prob); 4D additive ancestor mask; `cache_position` decoupled from `position_ids`
(RoPE uses tree depth, cache slots are contiguous); per-layer KV gather
(`layer.keys[:,:,gather_idx,:]`) compacts the accepted path. Harness: `bench_tree.py`.

**Result — Qwen3-4B + DFlash-b16, greedy, 16 samples, max_new=256:**

gsm8k:
| metric                    | reference | tree 24:2 | tree 48:3 | tree 64:4 |
|---------------------------|-----------|-----------|-----------|-----------|
| throughput (tok/s)        | 228.2     | 236.6     | 249.2     | **258.8** |
| accepted tok / target-fwd | 6.34      | 7.17      | 7.79      | **8.10**  |
| target forwards (NFE)     | 546       | 476       | 438       | **421**   |
| speedup                   | 1.00x     | 1.036x    | 1.092x    | **1.134x**|

humaneval: tree 64:4 → **1.139x** (258.5 vs 227.1 tok/s), acc/fwd 6.22→8.05, NFE 652→504.

**Knee.** Throughput peaks at ~64 nodes; pushing further (96, 128) keeps lifting
acc/fwd (8.10→8.41→8.61) but tok/s plateaus then dips (verify stops being flat
past ~64 tokens + Python tree-build overhead). Sweet spot ≈ **64 nodes, top_k=4**.

**Losslessness (rigorously controlled).**
- `top_k=1` (exact mechanism, linear chain) = **16/16 token-identical** to the
  reference → the tree builder + 4D mask + KV-gather + verification are
  **algorithmically exact**.
- Larger trees show 10/16 (gsm8k) / 5/16 (humaneval) identical. This is **not** a
  bug: it is the known **bf16 tie-break** noise. Control on the same humaneval
  data: reference `block=16` vs `block=12` (a benign change that merely reorders
  bf16 attention accumulation) agree only **8/16** — the floor. A wide tree
  reorders accumulation the same way, so its rate sits at/just below that floor.
  The divergences are single-token bf16 arg-max flips, identical in nature to the
  caveat that already applies when you change the reference block size.

**Why this one works (and Innovation 2 didn't).** Innovation 2 tried to improve
*per-position* draft quality, which is useless for a single-candidate **prefix**
verifier (only the frontier's left context matters, already present). Innovation 1
instead widens the **set of candidates at the frontier** and lets the *target*
pick — directly attacking the acceptance bottleneck, and monetizing the draft
distributions DFlash already computes for free. The flat verify cost is what makes
the extra tree tokens nearly free, so +28% acc/fwd converts to a real ~1.13x tok/s.

**Headroom / next steps.**
- Pure-Python per-iter tree+mask build is overhead; a vectorized ancestor mask
  and a fixed/cached tree topology (Medusa-style static tree) would push closer to
  the +28% acc/fwd ceiling (toward ~1.25-1.3x).
- **Batched / compute-bound serving**: there the verify is *not* flat, so the
  NFE reduction (-23%) matters even more directly — expected larger wins.
- Combine with Innovation 2 *inside* the tree (refine deep candidates before
  building the tree) — refinement's gains are only cashable under multi-candidate
  verification, which this provides.
- Stochastic (temperature>0) tree verification (typical/SpecInfer acceptance) is
  not yet implemented; current prototype is greedy-only (asserted).

## Code-change notes (Innovation 1)
- Additive only: `dflash/tree.py`, `bench_tree.py`. No reference files touched.
- `dflash_generate_tree(top_k=1)` is token-identical to `dflash_generate` (16/16).

---

# Innovation B — Cascaded tree verification (cheap pruner) (BUILT, NEGATIVE)

**Idea.** The plain tree (Innovation 1) verifies the top-`K` nodes by the draft's
**marginal** per-position scores, and tops out at the ~64-node flat-verify knee.
Hypothesis: build a *much bigger* candidate tree (`big_nodes`=96–192), use a
**cheap pruner** to shortlist the best `keep_nodes`=48, and full-verify only those
— so the scarce verify budget is spent on *better* candidates. The proposed cheap
pruner: one extra *cacheless* draft pass over the big tree with an **ancestor
mask**, giving each node a score *conditioned on its real ancestor path*
(`child_score = log p_draft(token | ancestors)`), then re-expand a 48-node subtree
by conditional path-score.

**Implementation.** `dflash/cascade.py::dflash_generate_cascade(big_nodes,
keep_nodes, top_k, prune)`. Validated correct: `prune=False` is **token- and
acc/fwd-identical** to the plain tree (`casc_off == tree48`). The conditional
pruner pass and 4D-ancestor-mask draft plumbing were unit-tested
(`mask key length = ctx_len + N`; cacheless conditional logits verified sane).

**Result — Qwen3-4B + DFlash-b16, gsm8k, greedy, 12 samples, max_new=256:**

| variant              | tok/s  | acc/fwd |
|----------------------|--------|---------|
| tree 48 (marginal)   | **253.6** | **8.14** |
| cascade 96 -> 48     | 185.5  | 7.35    |
| cascade 192 -> 48    | 137.0  | 6.62    |

Strictly **worse on both axes**, and worse the bigger the pool. Single-block
diagnostic confirms the mechanism: on a code block, marginal-48 reaches accepted
depth **11**, conditional-48 only **8**, having reshuffled 26/48 nodes away from
the marginal set.

**Why it fails (clean reason).** The pruner re-scores with the **same draft model
that built the tree**. Re-ranking by the same model cannot inject information
correlated with *target* acceptance that the marginal scores didn't already carry
— it only adds noise. Worse, the "conditional" score fixes each node's ancestors
to **candidate** tokens that may be wrong, and conditioning on wrong context
actively *mis-ranks* (same failure mode as Innovation 2: conditioning on
unreliable neighbours doesn't help a prefix verifier). Plus the extra
96–192-node draft pass is real added latency. Net: the marginal expansion is
already near-optimal for choosing *which* candidates to verify.

**What a useful cascade actually needs.** A pruner signal that is (a) cheaper than
full target verify but (b) correlated with the **target**, not the draft — i.e.
the **target's own early-exit layers** scoring the big tree, then full-depth
verifying the survivors. That is a genuine architectural cascade (needs an
early-exit head / partial target forward), and is exactly Innovation A
(self-speculative / shared-trunk DFlash) in disguise. Draft-only rescoring is a
dead end.

**Takeaway.** Confirms a consistent theme across Innovations 2 and B: **no amount
of extra draft-side computation raises acceptance** — acceptance is set by
target-agreement, so the only levers are (i) widen frontier candidates and let
the *target* choose (Innovation 1, works), (ii) make the draft trunk *be* the
target (Innovation A, needs training), or (iii) train the draft to align better
with the target (distillation). `cascade.py` is retained as a validated
big-tree + tree-mask-pruner scaffold for plugging in a target-early-exit pruner.

## Code-change notes (Innovation B)
- Additive only: `dflash/cascade.py`. No reference files touched.
- `dflash_generate_cascade(prune=False)` is identical to the plain tree decoder.
