# V4 winning-config **scale ladder** — tokens-to-target across model sizes

**Status: COMPLETE — all three rungs (13M / 42M / 110M) run, every per-family LR optimum
bracketed.** Headline: the V4-sweep's *optimizer-dominance* holds at every scale, but its
*mHC-is-redundant-with-Muon* finding **inverts at the 110M rung** (mHC×2 beats bare Muon by
−0.024) — with a size-vs-head_dim confound flagged honestly below. The floor rung reproduces
the V4 sweep to 4 sig figs (harness-faithfulness check).

## The question

[`results/v4_sweep.md`](v4_sweep.md) ranked the DeepSeek-V4 levers on a **single rung**
(r2_small, ~13M, seq256, 800-step screen). Its verdict: **optimizer ≫ architecture (~5–8×)**
— plain Muon @ lr≈1e-2 beats fair AdamW by −0.88, while the best architecture lever (mHC)
buys −0.11 on AdamW and **~0 on Muon** (mHC↔Muon are redundant), and "everything-on" (`max`)
loses to bare Muon. The V4 mechanisms target **3B+ / long context**. The open question this
ladder answers, on the one axis we can move cheaply on a single ANE — **model size**:

> Does optimizer-dominance and the mHC↔Muon redundancy **hold, shrink, or invert** as the
> dense model grows from 13M → 42M → 110M (seq fixed at 256)?

Design + the headline-metric derivation: [ADR 0003](../docs/adr/0003-v4-winner-scale-ladder.md).

## Headline metric: tokens-to-target

Held-out val loss vs **training tokens** (not wall-clock / tokens-sec / inference memory).
The ANE trainer processes **b=1 sequence per micro-batch** (`train.m:619`), so
**tokens = `step` × `SEQ`**, and with **`SEQ=256` held across every rung the token axis is
identical** — comparing val-vs-step curves *is* comparing tokens-to-target. Budget is the
sweep's 800-step / accum-4 screen; val is logged every 50 steps, so the curve spans
**0 → 192,000 tokens** (16 points; cadence stops at step 750).

## The ladder

| rung | dims | params | head_dim | role |
|---|---|---|---|---|
| `r2_small` | d256 / 6L / 8H | **13.3M** | 32 | floor (smallest real-text 32K rung) |
| `r2_mid` | d512 / 8L / 16H | **42.1M** | 32 | intermediate (width+depth at *same* head_dim) |
| `r3_110m` | d768 / 12L / 12H | **109.5M** | 64 | ceiling (largest that runs on the ANE) |

~3.2× then ~2.6× params — even log-spacing. r2_small→r2_mid isolates width+depth (head_dim
held at 32); r3_110m is the rung that also widens head_dim. **Ceiling confirmed by smoke
test:** r3_110m compiles (10 kernels/492ms) and trains at **~100 ms/step** all-ANE — no
fallback to the hand-written `stories110m.h` needed.

## Method (per rung)

- **4 config families**, each at its own **≥3-point LR grid** (mandatory — optimal LR shifts
  with scale): `muon`/`mhc2`/`mhc4` @ `{3e-3, 1e-2, 3e-2}`, `adamw` @ `{3e-4, 1e-3, 3e-3}`,
  bracketing the r2_small optima. **Grids are extended if a rung's optimum lands on an edge.**
- **3 distinct builds** R0-gated (optimizer is a runtime flag → muon & adamw share the
  no-knob build; only `N_HC` is compile-time): `plain`, `mhc2`, `mhc4`.
- **R0 correctness gate at the rung's own dims** — overfit one pinned real batch to <0.1
  (lr 1e-3, clip 1.0; the hot lr-2e-3 overfit explodes in fp16 at d768 — ADR 0003). A build
  whose R0 does not collapse is not trusted at R2.
- Held fixed across rungs: data (`data00` train / `data01` val), seq 256, b=1, accum 4, seed,
  wd 0.1, warmup 40, 800-step budget.

---

## Rung 1 — `r2_small` (13.3M, floor) ✅

**R0 gate:** `plain` 0.0000 · `mhc2` 0.0000 · `mhc4` 0.0001 — all **PASS**.

### Harness faithfulness — reproduces the V4 sweep to 4 sig figs

The generalized `MODEL=`-parameterized harness re-derives the sweep's numbers exactly, so the
larger rungs are trustworthy:

| cell | ladder re-run | `v4_sweep.md` | | cell | ladder re-run | `v4_sweep.md` |
|---|---|---|---|---|---|---|
| muon@1e-2 | **3.5660** | 3.566 | | adamw@1e-3 | 4.4427 | 4.443 |
| muon@3e-3 | 3.7336 | 3.734 | | adamw@3e-4 | 5.2146 | 5.215 |
| muon@3e-2 | 3.8193 | 3.819 | | adamw@3e-3 | 5.5900 | 5.590 |
| mhc4@1e-2 | 3.5776 | 3.578 (Stage E) | | | | |

### Leaderboard (val@750 = 192k tokens, each family at its best LR; full 12-cell grid in TSV)

| # | family | best LR | val@192k | Δ vs muon | tokens-to-≤4.5 | wall/cell |
|---|---|---|---|---|---|---|
| 1 | **muon** | 1e-2 | **3.566** | — | **64,000** | 77s |
| 2 | mhc4 (muon+mHC×4) | 1e-2 | 3.578 | +0.012 | 64,000 | 363s |
| 3 | mhc2 (muon+mHC×2) | 1e-2 | 3.590 | +0.024 | 64,000 | 144s |
| 4 | **adamw** | 1e-3 | 4.443 | +0.877 | **179,200** | 35s |

All four optima are **interior** to their grids (no edge → no extension needed). Optimizer
gap (muon−adamw = **−0.877**) reproduces the sweep's −0.88.

### Reading (floor)

- **Optimizer dominance, in tokens:** Muon reaches val ≤ 4.5 in **64k tokens vs AdamW's 179k
  (~2.8× fewer)**, and reaches ≤ 4.0 (115k tokens) where **AdamW never gets within budget**.
- **mHC is redundant on Muon** — mHC4 +0.012, mHC2 +0.024 vs plain Muon (both *worse*), at
  ~2–5× the wall-clock. Reproduces the sweep's "mHC redundant with Muon" verdict, now with a
  full 3-pt LR sweep on each.
- **Nuance — mHC front-loads, Muon's edge is asymptotic:** at 25k tokens mHC4 (4.84) and mHC2
  (4.90) *lead* plain Muon (5.08); Muon overtakes by ~150k and finishes ahead. So mHC's
  conditioning helps the steep early descent but Muon catches up — consistent with the two
  being partially redundant update-conditioners.

---

## Rung 2 — `r2_mid` (42.1M) ✅

**R0 gate:** `plain`/`mhc2`/`mhc4` all 0.0000 — **PASS**.

**LR optima moved — and the grid had to be extended.** Two families' optima fell to a grid
edge (the handoff's #1 lesson: a fixed LR gives a wrong verdict), so the grids were extended
down and re-bracketed before trusting any number:

| family | grid (final val per LR) | optimum | vs r2_small |
|---|---|---|---|
| **muon** | 3e-3=3.552 · **1e-2=3.528** · 3e-2=4.212 | **1e-2** | **held** (interior both rungs) |
| adamw | 3e-5=6.746 · 1e-4=5.210 · **3e-4=4.440** · 1e-3=4.825 · 3e-3=6.033 | **3e-4** | 1e-3 → 3e-4 (**down ~3×**) |
| mhc2 | 1e-3=3.800 · **3e-3=3.580** · 1e-2=3.738 · 3e-2=5.019 | **3e-3** | 1e-2 → 3e-3 (**down ~3×**) |
| mhc4 | 1e-3=3.792 · **3e-3=3.574** · 1e-2=3.768 · 3e-2=4.599 | **3e-3** | 1e-2 → 3e-3 (**down ~3×**) |

All four optima are now interior. **Methodological headline:** AdamW's and mHC's optimal LR
both slid **down ~3×** as params grew ~3×, while **Muon's optimum held at 1e-2** — Muon is
markedly more LR-robust to scale (consistent with Newton-Schulz producing a scale-normalized,
≈unit-spectral-norm update, so the effective step is far less width-dependent). A single-LR
sweep would have mis-ranked everything at 42M.

### Leaderboard (val@750 = 192k tokens, each family at its **bracketed** best LR)

| # | family | best LR | val@192k | Δ vs muon |
|---|---|---|---|---|
| 1 | **muon** | 1e-2 | **3.528** | — |
| 2 | mhc4 | 3e-3 | 3.574 | +0.046 |
| 3 | mhc2 | 3e-3 | 3.580 | +0.052 |
| 4 | adamw | 3e-4 | 4.440 | +0.912 |

### Reading (42M)

- **Optimizer dominance GROWS:** the muon−adamw gap widens from −0.877 (13M) to **−0.912**
  (42M). Muon still reaches val ≤ 4.5 in **64k tokens**; AdamW needs 179k and never reaches
  ≤ 4.0 within budget. Tokens-to-target table (below) confirms the optimizer is *the* lever.
- **mHC stays redundant with Muon — and falls slightly further behind:** mhc4 +0.046, mhc2
  +0.052 vs bare Muon (both worse, both bigger gaps than at 13M). mHC is **not** catching up
  to Muon as size grows over this range.
- **Floor drops with size** (Muon 3.566 → 3.528) — the bigger model reaches a lower asymptote,
  as expected; the ladder's question is whether the *ranking* survives, and so far it does.

## Rung 3 — `r3_110m` (109.5M, ceiling) ✅

**R0 gate:** `plain`/`mhc2`/`mhc4` all 0.0000 — **PASS** (clean collapse at d768/head_dim-64;
the tamed lr-1e-3/clip-1.0 recipe, since the hot lr-2e-3 overfit explodes in fp16 at d768).

**LR optima moved again — and Muon finally moved too.** Grids were re-centered on the
extrapolated optima, then **two families still landed on edges and had to be extended**:

| family | grid (final val per LR) | optimum | vs r2_mid |
|---|---|---|---|
| muon | 3e-4=4.023 · 1e-3=3.675 · **3e-3=3.461** · 1e-2=3.730 · 3e-2=4.453 | **3e-3** | 1e-2 → 3e-3 (**finally dropped**) |
| adamw | 3e-5=5.692 · 1e-4=4.681 · **3e-4=4.222** · 1e-3=5.968 | **3e-4** | held at 3e-4 |
| mhc2 | 1e-3=3.643 · **3e-3=3.437** · 1e-2=4.005 | **3e-3** | held at 3e-3 |
| mhc4 | 1e-3=3.623 · **3e-3=3.456** · 1e-2=3.947 | **3e-3** | held at 3e-3 |

All four optima are now interior/bracketed. **Muon's optimum, rock-solid at 1e-2 through 13M
and 42M, dropped to 3e-3 at 110M** — and crucially the extension confirmed it *bottoms* there
(1e-3=3.675 is worse), so **Muon's true floor is 3.461**, not lower. This bracketing is what
makes the headline below trustworthy rather than a grid artifact.

### Leaderboard (val@192k, each family at its **bracketed** best LR)

| # | family | best LR | val@192k | Δ vs muon |
|---|---|---|---|---|
| 1 | **mhc2** (muon+mHC×2) | 3e-3 | **3.437** | **−0.024** |
| 2 | mhc4 (muon+mHC×4) | 3e-3 | 3.456 | −0.005 |
| 3 | muon | 3e-3 | 3.461 | — |
| 4 | adamw | 3e-4 | 4.222 | +0.761 |

### Reading (110M) — the inversion

- **mHC inverts to *helping* Muon.** For the first time on the ladder, mHC beats bare Muon:
  mHC×2 by −0.024 (signal — the sweep's noise floor is ~0.01), mHC×4 ties (−0.005). At 13M/42M
  mHC was strictly *worse* than Muon (redundant); at 110M it is better. tokens-to-target agrees:
  mhc2 reaches val ≤ 3.7 in **140.8k tokens vs Muon's 153.6k**, and ≤ 4.0 in 102.4k vs 115.2k.
- **Optimizer gap narrows but Muon still dominates.** adamw−muon = +0.761 (vs +0.912 at 42M) —
  the gap stopped growing and edged down, mostly because AdamW improved at 110M (4.44 → 4.22)
  while Muon was ~flat. Muon (or Muon+mHC) is still the biggest single lever by far.
- **`mhc2 < mhc4`.** Two streams beat four here — the paper's `n_hc=4` anchor is *not* the
  small-dense optimum; the extra streams cost ~2× the wall-clock for a worse result.

> **⚠️ Confound — size vs head_dim.** r3 differs from r2_mid in **two** ways: params (42M→110M)
> *and* head_dim (32→64). The one controlled size step (r2_small→r2_mid, both hd32) shows mHC
> getting **more** redundant with scale (+0.024→+0.052), not less. So the inversion appears
> **only at the rung that also widens head_dim** — it cannot be attributed to scale alone. The
> data is equally consistent with "head_dim 64 is what lets mHC help." Disentangling needs a
> d768/hd32 (or d512/hd64) rung — see Open questions.

## Cross-size synthesis — hold / shrink / invert?

Best-LR val@192k per (family × rung), all optima bracketed:

| | r2_small (13M) | r2_mid (42M) | r3_110m (110M) |
|---|---|---|---|
| muon | 3.566 | 3.528 | 3.461 |
| mhc2 | 3.590 | 3.580 | **3.437** |
| mhc4 | 3.578 | 3.574 | 3.456 |
| adamw | 4.443 | 4.440 | 4.222 |
| **optimizer gap** (adamw−muon) | +0.877 | +0.912 | +0.761 |
| **mHC×2** (mhc2−muon) | +0.024 | +0.052 | **−0.024** |
| **mHC×4** (mhc4−muon) | +0.012 | +0.046 | −0.005 |

**Verdict:**
1. **Optimizer dominance HOLDS at every scale** (the V4-sweep's main finding survives): Muon is
   the biggest single lever at all three sizes (+0.76 to +0.91 over fair AdamW). It does **not
   widen** with scale — it peaked at 42M and narrowed at 110M — but it never stops being #1.
2. **mHC↔Muon redundancy INVERTS at the 110M/hd64 rung** — the V4-sweep's "mHC is redundant
   with Muon" finding *breaks* here: mHC×2 goes from −worse to +better (+0.024 → −0.024). This
   is the headline the ladder was built to test — but see the size-vs-head_dim confound above:
   under *pure* scaling (constant hd32) the redundancy actually deepened; the reversal needs
   the head_dim widening too.
3. **The absolute floor drops monotonically with size** (3.566 → 3.528 → 3.437) even on the
   192k-token screen — bigger model, lower loss, as expected. The best config at the ceiling is
   **Muon + mHC×2 @ 3e-3**, not plain Muon.

## Caveats (carried from the sweep, plus ladder-specific)

- **Size vs head_dim confound (the big one).** The inversion lands on the rung that widens
  head_dim 32→64 *and* triples params. Under pure scaling (hd32, r2_small→r2_mid) mHC got more
  redundant, not less. The reversal therefore can't be attributed to scale alone — a
  head_dim-controlled rung is needed to separate the two (Open questions).
- **Single seed.** The inversion (−0.024) clears the ~0.01 noise floor (established by the
  sweep's exact `min`≡`swiglu` tie) but is **one seed**. The trainer seed is fixed (LR is the
  variance lever, per ROADMAP) — there is **no `--seed` flag**, so a seed sweep needs a code
  change. A −0.024 single-seed effect is *suggestive*, not bulletproof; treat it as a
  hypothesis the next rung should confirm, not a settled number.
- **Screen budget.** 192k tokens is the steep early descent — enough to *rank*, not to call
  converged frontiers. The ranking is consistent and mechanism-plausible, but the absolute
  values are far above a converged run (cf. Karpathy's llama2.c stories110M ≈ 0.76 val,
  converged at seq 1024 — a different, frontier-reaching experiment).
- **fp16 ANE, no gradient oracle.** R0 collapse (now at each rung's *own* dims) is the
  behavioral correctness proxy.
- **Seq fixed at 256.** This ladder moves *size*, not context length; the long-context V4
  levers (CSA/HCA, partial-RoPE — which *hurt* at 256) are out of scope by construction.
- **Ceiling = 110M on one ANE**, ~30× below the 3B+ the mechanisms target. The ladder tests
  the *trend*, not the destination.

## Open questions (what would settle the confound)

- **A head_dim-controlled rung** — `d768/hd32` (12 heads → 24 heads) or `d512/hd64` — to test
  whether the mHC inversion tracks *params* or *head_dim*. This is the single most valuable
  follow-up; everything else is secondary to it.
- **A seed sweep** (needs a `--seed` flag in `train.m`) to put an error bar on the −0.024.
- **Convergence run** of the best ceiling config (Muon+mHC×2 @ 3e-3) to a real frontier, for
  an apples-to-apples vs llama2.c stories110M (and to check the ranking survives past the knee).
- **`max` at the ceiling** — does "everything-on loses" still hold now that mHC has flipped to
  helping at 110M? (Deferred this study; partial-RoPE still hurts at seq 256, so likely yes.)

## Reproduce

```bash
# one rung end-to-end (R0-gate 3 builds -> R2 the 12 cells); artifacts in /tmp/ladder/<rung>
MODEL=r2_small zsh training/sweep/run_ladder_rung.zsh
MODEL=r2_mid   zsh training/sweep/run_ladder_rung.zsh
MODEL=r3_110m  zsh training/sweep/run_ladder_rung.zsh
# curves + optimum-LR + tokens-to-target tables across whatever rungs are present
python training/sweep/analyze_ladder.py --targets 3.7,4.0,4.5,5.0
```
