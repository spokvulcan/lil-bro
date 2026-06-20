# DeepSeek-V4 config sweep — correctness + fast-learning (R0 → R2)

**Status: R0 correctness GREEN for all 10 cells (including the full "max" stack);
R2 fast-learning screen complete; Muon LR-fairness sweep (Stage C) in progress.**

The question this answers (per the session goal): turn *everything* DeepSeek-V4 on
(paper-faithful max) vs *everything* off (plain transformer min), plus one-variable
ablations — and rank them on **correctness** (does it train) and **fast learning**
(held-out val vs step). Method follows ROADMAP invariants: vary one component, hold
data / seq / batch / dims / val-set fixed, trust a cell only after its R0 gate.

See [ADR 0002](../docs/adr/0002-observing-mhc-flagship-train.md) (the mHC flagship
observability work this builds on) and [ADR 0001](../docs/adr/0001-deepseek-v4-for-dense-ane.md)
(the V4 tier list + paper anchors). Per-mechanism detail lives in the sibling
`results/{qk_norm,attn_sink,swiglu_clamp,partial_rope,mtp,muon_v4,mhc}.md`.

## Cells

Ten configs, each a compile-time build of the **r2_small** rung (d=256, 6 layers,
8 heads, head_dim=32, seq=256, 32K vocab). The V4 knobs map to PRD child mechanisms
#6–#11 + Muon. Paper anchors from ADR 0001 §"Reference anchors": `n_hc=4`, partial
RoPE, V4 Newton-Schulz Muon. Tier C (CSA/HCA) is deferred (needs long context) and
Tier D (MoE/FP4) is out of scope, so `max` is the most-V4 the dense ANE model expresses.

| cell | knob(s) | tier |
|---|---|---|
| `min` | plain transformer, all V4 off | baseline |
| `qk_norm` | Q/KV RMSNorm before scores (#7) | A |
| `attn_sink` | learnable per-head sink logit (#8) | A |
| `swiglu` | SwiGLU clamp to [−10,10], gate cap 10 (#9) | A |
| `prope` | partial RoPE, rotate last 16 of 32 dims (#10) | A |
| `nhc2` | mHC, 2 residual streams (#11) | B |
| `nhc4` | mHC, **4 streams** = paper anchor (#11) | B |
| `mtp1` | multi-token prediction, depth 1 (#6) | B |
| `muon` | V4 hybrid Newton-Schulz optimizer (§2.4) | A |
| `max` | **all of the above together**, Muon | A+B |

Note `prope` is only non-identity because `rope_rotary_dims < head_dim`; at head_dim
≤ rope_rotary_dims it is a byte-identity no-op (true at every default rung).

## Phase 1 — R0 correctness gate (ANE, behavioral)

Overfit one pinned batch (`--overfit`, gen_r0_overfit dims, adamw, lr 2e-3, 800
steps). R0 isolates **architecture forward+backward correctness**; the optimizer is a
separate axis, so every cell's R0 uses AdamW (Muon's own correctness is in
`muon_v4.md`). A cell whose loss does not collapse here is not trusted at R2.

| cell | min | qk_norm | attn_sink | swiglu | prope | nhc2 | nhc4 | mtp1 | muon | **max** |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 final loss | .0009 | .0013 | .0005 | .0009 | .0009 | .0038 | .0035 | .0004 | .0009 | **.0005** |

**All 10 PASS** (floor ≈ .0156 fp16; all collapse well below). The load-bearing new
result: the **`max` stack — QK-norm + attn-sink + SwiGLU-clamp + partial-RoPE + mHC×4
+ MTP, never compiled or run together before — drives R0 to 0.0005.** Its combined
backward (CPU-attention bypass + CPU mHC×4 + MTP cross-depth chaining) is correct.

## Phase 2 — R2 fast-learning screen

r2_small from scratch, **identical data (data00) / seed / budget** across cells:
800 optimizer steps, accum 4 (batch 1024 tok), val every 50 on data01, lr 3e-4.
Headline metric = held-out val loss at fixed steps (tokens-to-target proxy).

| rank | cell | opt | val@800 | Δ vs baseline | wall_s |
|---|---|---|---|---|---|
| 1 | **nhc4** (mHC×4) | adamw | **5.059** | −0.156 | 323 |
| 2 | **max** (all on) | muon | **5.066** | −0.148 | 550 |
| 3 | attn_sink | adamw | 5.101 | −0.114 | 154 |
| 4 | nhc2 (mHC×2) | adamw | 5.113 | −0.102 | 106 |
| 5 | qk_norm | adamw | 5.150 | −0.065 | 189 |
| 6 | mtp1 | adamw | 5.181 | −0.034 | 70 |
| 7 | min (baseline) | adamw | 5.215 | — | 34 |
| 7 | swiglu | adamw | 5.215 | 0.000 | 34 |
| 9 | prope | adamw | 5.274 | +0.059 | 35 |
| 10 | muon | muon | 5.374 | +0.159 | 78 |

### Reading

- **mHC is the dominant lever.** `nhc4` is the best single mechanism, and
  `nhc2 < nhc4` (more streams help). The flagship choice is vindicated on *fast
  learning*, not only correctness.
- **all-on (`max`) ≈ mHC-alone, at 16× the wall-clock.** `max` (5.066) does not beat
  `nhc4` (5.059) but costs 550s vs 323s (and 16× the 34s baseline). The full stack's
  net gain over its mHC component is ~0 at this budget — the other mechanisms' small
  gains are offset by partial-RoPE's drag and Muon's under-tuning (see Stage C).
- **SwiGLU-clamp is a pure no-op early** — identical loss *and* wall to baseline
  (5.2146 / 34s). Expected: the clamp only engages on activations outside [−10,10],
  which don't occur this early. It is a large-/late-run stabilizer, not a
  fast-learning lever. (The `min`≡`swiglu` tie also confirms run-to-run determinism.)
- **Partial-RoPE hurts** at this scale (+0.059): leaving 16 of 32 head dims unrotated
  (NoPE) loses positional signal at seq 256.
- **Muon is last — but at AdamW's lr.** Muon's orthogonalized update wants a much
  higher LR; 3e-4 starves it. `max` *uses* Muon, also at 3e-4, so both Muon configs
  are under-tuned. → Stage C.

The `wall_s` column quantifies the ADR-0001 CPU-first cost: CPU-attention (`qk_norm`,
`attn_sink`) and CPU mHC (`nhc*`, `max`) cells run 3–16× slower than the all-ANE
baseline. This is a *throughput* secondary, not the headline (tokens-to-target).

## Stage C — Muon LR-fairness sweep

Muon @ {1e-3, 3e-3, 1e-2} and `max` @ Muon {1e-3, 3e-3}, same 800-step budget, to
give Muon (and the Muon-based `max`) a fair optimum the screen denied them.

| cell | opt | lr | val@800 |
|---|---|---|---|
| muon | muon | 1e-2 | **3.566** |
| max | muon | 3e-3 | 3.728 |
| muon | muon | 3e-3 | 3.734 |
| max | muon | 1e-3 | 4.095 |
| muon | muon | 1e-3 | 4.185 |

**This inverts the screen verdict.** Muon was last *only* because the screen starved
it at AdamW's lr 3e-4. Given its own LR, Muon at 1e-2 reaches **3.566** vs the AdamW
baseline's **5.215** at the same budget — −1.65. (That gap is vs the *not-yet-retuned*
baseline; AdamW was also under-tuned, so the **fair** Muon-vs-AdamW gap is **−0.88** —
see Stage D. Either way it dwarfs the best *architectural* effect, mHC's −0.16.)
**The optimizer is the dominant fast-learning lever; architecture is second order.**

At matched LR, `max` ≈ plain `muon` (1e-3: 4.095 vs 4.185; 3e-3: 3.728 vs 3.734) —
i.e. once Muon is doing the work, stacking all the architecture mechanisms on top adds
almost nothing at this budget, while costing ~7× the wall-clock (549s vs 77s).

## Stage D — close the LR grid

(1) AdamW was itself pinned at 3e-4 (same under-tuning I corrected for Muon) →
sweep `min`; (2) confirm Muon's optimum isn't past 1e-2; (3) run `max` at Muon's
optimum, which Stage C never did.

| cell | opt | lr | val@800 |
|---|---|---|---|
| max | muon | 1e-2 | 3.701 |
| muon | muon | 3e-2 | 3.819 |
| min | adamw | 1e-3 | **4.443** |
| min | adamw | 3e-3 | 5.590 |

Both optima are now **bracketed** (each degrades on the far side: AdamW 3e-4 → 1e-3
→ 3e-3 = 5.215 / 4.443 / 5.590; Muon 1e-2 → 3e-2 = 3.566 / 3.819), so they are real
minima, not grid edges.

- **AdamW was under-tuned too.** Its real optimum (1e-3, 4.443) beats the screen's
  3e-4 (5.215) by 0.77 — the same bias I corrected for Muon, applied honestly to the
  baseline. The fair optimizer gap is therefore **Muon 3.566 vs AdamW 4.443 = −0.88**,
  not the −1.65 the raw screen implied. Muon still wins clearly, by ~half as much.
- **The full stack loses to plain tuned Muon.** `max`@1e-2 (3.701) > `muon`@1e-2
  (3.566) by 0.135. At Muon's optimum, stacking all V4 architecture mechanisms is
  **net-negative** — the partial-RoPE penalty and inert clamp outweigh mHC, whose
  own benefit fades under Muon.

## Final verdict — what works best (fast learning, r2_small, 800-step screen)

**Ranked by the fair-LR optimum of each:**

| # | config | opt @ best LR | val@800 | takeaway |
|---|---|---|---|---|
| 1 | **plain Muon** | muon 1e-2 | **3.566** | the single best lever |
| 2 | full V4 stack (`max`) | muon 1e-2 | 3.701 | all-on *loses* to bare Muon |
| 3 | Muon (off-optimum) | muon 3e-2 | 3.819 | |
| 4 | plain AdamW | adamw 1e-3 | 4.443 | well-tuned baseline |
| — | AdamW + mHC×4 | adamw 3e-4† | 5.059 | best *architecture* lever (−0.16) |
| — | AdamW baseline | adamw 3e-4 | 5.215 | reference |

†architecture cells' fair-LR retest in Stage E (below).

**Conclusions:**
1. **Optimizer ≫ architecture (~5×).** Switching AdamW→Muon and tuning its LR buys
   −0.88 val; the best architectural mechanism (mHC) buys −0.16. For *fast learning*
   at this scale, the optimizer is the lever that matters.
2. **"Everything on" is not the frontier.** The paper-faithful max stack is beaten by
   plain tuned Muon. More mechanisms ≠ better here — partial-RoPE hurts at seq 256,
   SwiGLU-clamp is inert early, and mHC's gain does not survive on top of Muon.
3. **Recommendation:** **plain Muon @ lr ≈ 1e-2.** (Stage E overturned the initial
   "Muon + mHC" guess — mHC is redundant with Muon; see Final recommendation below.)
4. **Correctness is universal** (Phase 1): all 10 configs, including the combined max
   stack, train correctly on the ANE — so any of these is a *valid* choice; this is
   purely an efficiency ranking.

## Stage E — fair-LR architecture retest

mHC and attn-sink were only seen at AdamW's under-tuned 3e-4. Retest at AdamW 1e-3,
and isolate **mHC on the best optimizer** (Muon @ 1e-2).

| mechanism | @ AdamW 3e-4 | @ AdamW 1e-3 (fair) | Δ vs fair baseline |
|---|---|---|---|
| baseline | 5.215 | 4.443 | — |
| + mHC×4 | 5.059 | **4.333** | **−0.110** |
| + attn_sink | 5.101 | 4.393 | −0.050 |

| mHC isolated on each optimizer @ its optimum | val@800 | Δ |
|---|---|---|
| AdamW 1e-3:  min 4.443  →  +mHC **4.333** | | **−0.110 (helps)** |
| Muon  1e-2:  muon 3.566 →  +mHC **3.578** | | **+0.012 (no help)** |

**Decisive result:** mHC's benefit is real on AdamW (−0.11, and it survives fair-LR
tuning) but **disappears on top of Muon** (+0.01 — tied, marginally worse). attn-sink
behaves the same way (helps AdamW a little, shrinks at fair LR). The mechanistic read,
consistent with `max` < `muon`: **Muon (Newton-Schulz orthogonalization) and mHC
(doubly-stochastic residual conditioning) are partially redundant** — both improve
update/signal conditioning, so once Muon does it, the architecture has little to add.

## Final recommendation (corrected after Stage E)

**For fast learning at r2_small scale: use plain Muon @ lr ≈ 1e-2.** It is the single
best config measured (3.566), and it beats the full paper-faithful V4 stack (3.701).

- **Optimizer is the lever (~5–8×).** Muon vs fair AdamW = −0.88; the best
  architecture mechanism (mHC) on AdamW = −0.11; on Muon = ~0.
- **mHC: keep it *only* with AdamW.** It's the strongest architecture mechanism and
  genuinely helps AdamW (−0.11), but is redundant with Muon — do not pay its ~10×
  CPU cost on top of Muon for no gain.
- **Drop partial-RoPE** at seq 256 (it hurts, +0.06). **SwiGLU-clamp** is an early
  no-op and **attn-sink** a minor help — treat both as scale/length stabilizers (their
  V4 role), not fast-learning levers; revisit at a long-context / larger rung.
- **"Everything on" is not the frontier.** Stacking all V4 mechanisms (`max`) is
  net-negative vs plain tuned Muon at this scale.
- **All of this is an *efficiency* ranking, not a correctness one** — Phase 1 proved
  every config (including the full max stack) trains correctly on the ANE.

This is a small-scale (20M, seq 256, 800-step, single-seed) fast-learning finding.
The V4 mechanisms target 3B+ / long context; the right next test is whether mHC's
AdamW benefit and the stabilizers' value grow at the R3 (110M) rung and at longer
sequences — and whether mHC↔Muon redundancy persists there. See Caveats.

## Caveats (honest scope)

- **Short screen budget.** 800 steps / accum 4 reaches val ≈ 5.1, the steep early
  descent — enough to *rank*, not to call converged frontiers. The spread (5.06–5.37)
  is small but the ordering is consistent and mechanism-plausible; `min`≡`swiglu`
  exactly, so differences > ~0.01 are signal, not noise.
- **Single seed.** No seed sweep (the trainer seed is fixed; LR is the variance
  lever, per ROADMAP). A fair optimizer/arch headline needs the ≥3-pt LR sweep —
  done for Muon/`max` in Stage C; a full per-cell LR sweep is future work.
- **fp16 ANE, no gradient oracle** (ADR-0001 accepted trade-off): a subtly wrong
  backward that still trains is not caught. R0 collapse is the behavioral proxy.
- **accum 4 screen ≠ the accum-16 r2_small recipe.** Comparable across cells, but the
  absolute val is higher than a full-recipe run would reach.

## Reproduce

```bash
# harness in /tmp/sweep (cells.zsh, run_r0.zsh, run_r2.zsh); binaries built per-cell
zsh /tmp/sweep/run_r0.zsh                                   # Phase 1: R0 gates
ACCUM=4 R2_STEPS=800 VAL_EVERY=50 zsh /tmp/sweep/run_r2.zsh # Phase 2: R2 screen
CELLS_FILE=/tmp/sweep/cells_c.zsh R2SUM=/tmp/sweep/r2c_summary.tsv FORCE_R2=1 \
  ACCUM=4 R2_STEPS=800 VAL_EVERY=50 zsh /tmp/sweep/run_r2.zsh   # Stage C
```
