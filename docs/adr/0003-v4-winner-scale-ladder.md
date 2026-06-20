# ADR 0003 — Scale ladder for the V4 winning configs: tokens-to-target across model sizes

- **Status:** Accepted + **executed** (all 3 rungs run; results in
  [results/scale_ladder.md](../../results/scale_ladder.md)). Proceeded on the handoff defaults
  under autonomous direction (not `/grill-with-docs`-ratified; the operator asked to proceed
  without pausing). Decisions table below records what was locked and why (cf. ADR 0002).
- **Outcome (one line):** optimizer-dominance **holds** at every scale; the mHC↔Muon
  **redundancy inverts at the 110M/hd64 rung** (mHC×2 beats bare Muon −0.024) — confounded with
  the head_dim 32→64 widening, so a head_dim-controlled rung is the key follow-up.
- **Date:** 2026-06-20
- **Context docs:** [results/v4_sweep.md](../../results/v4_sweep.md) (the verdict being scaled) ·
  [ADR 0001](0001-deepseek-v4-for-dense-ane.md) (V4 tier list/anchors) ·
  [ADR 0002](0002-observing-mhc-flagship-train.md) (faithful sampler + v5 ckpt) ·
  [ROADMAP.md](../../ROADMAP.md) (ladder + method invariants) ·
  [training/sweep/](../../training/sweep/) (the harness, now `MODEL`-parameterized).
- **Supersedes nothing.** Extends the completed r2_small V4 sweep up the size ladder.

---

## Context

The V4 config sweep ([results/v4_sweep.md](../../results/v4_sweep.md)) is a **single-rung**
result: r2_small (d256/6L, ~13M params, seq256, 800-step/accum-4 screen, single seed). Its
verdict, at fair per-config LR:

- **Optimizer ≫ architecture (~5–8×).** Plain Muon @ lr≈1e-2 (val 3.566) beats fair AdamW
  @ 1e-3 (4.443) by −0.88; the best *architecture* lever, mHC, buys −0.11 on AdamW.
- **mHC ↔ Muon are redundant.** mHC helps AdamW (−0.11) but is ~0 on top of Muon (+0.01).
- **"Everything on" (`max`) loses** to plain tuned Muon (3.701 vs 3.566).

The V4 mechanisms target **3B+ / long context**. The open question the sweep itself flags:
**does optimizer-dominance and the mHC↔Muon redundancy hold, shrink, or invert as the model
grows?** This ADR is the design to answer it on the only axis we can move cheaply on one ANE:
**model size**, holding sequence length fixed.

## The headline metric (operator-confirmed): tokens-to-target

Held-out val loss as a function of **training tokens** — *not* wall-clock, tokens/sec, or
inference memory. The decisive simplification, verified in `train.m`:

> The ANE trainer processes **b=1 sequence per micro-batch** (`train.m:619`; the loop at
> `train.m:1237` draws one `SEQ`-token window per `step`, optimizer update every `--accum`
> steps at `train.m:1665`). So **training tokens = `step` × `SEQ`**. With `SEQ=256` held
> across *every* real-text rung, the token axis is **identical** across sizes — comparing
> val-vs-`step` curves *is* comparing val-vs-tokens. (`Config.batch_size`=16 is the MLX
> twin's notion, unused by the ANE trainer.)

We therefore hold **seq=256, b=1, accum=4 fixed across all rungs** so the x-axis is shared,
and read tokens-to-target off the val curves post-hoc for a target reachable by all cells.

## Ratified decisions

| Decision | Choice | One-line rationale |
|---|---|---|
| **Ladder rungs** | `r2_small` (d256/6L, 13.3M) → **`r2_mid`** (d512/8L, 42.1M, NEW) → `r3_110m` (d768/12L, 109.5M) | ~3.2× then ~2.6× params — even log-spacing. Floor = smallest real-text (32K) rung; ceiling = largest that runs (smoke-confirmed below). |
| **`r2_mid` shape** | d512 / 8L / **16H × hd32** / hidden1408 | Keep `head_dim=32` (= r2_small) so r2_small→r2_mid isolates width+depth; r3_110m is the rung that *also* widens head_dim (hd64). |
| **Configs per rung** | `muon`, `adamw`, `muon+mHC×2`, `muon+mHC×4` (`max` = optional secondary) | The sweep winners + the headline **redundancy test at scale** (does mHC stay redundant with Muon as size grows?). `max` answers "does everything-on still lose?" if time allows. |
| **Builds per rung** | **3 distinct**: `plain`, `mhc2` (`-DN_HC=2`), `mhc4` (`-DN_HC=4`) | Optimizer is a *runtime* flag (`--opt`), so muon & adamw share the no-knob build; only `N_HC` is compile-time. R0-gate the 3 builds, not 12 cells. |
| **LR grid (≥3 pts)** | muon/mHC `{3e-3, 1e-2, 3e-2}`; adamw `{3e-4, 1e-3, 3e-3}` | Brackets each family's *r2_small* optimum (muon 1e-2, adamw 1e-3 — both bracketed in the sweep). **Extend a grid if a larger rung's optimum lands on an edge** (LR is mandatory per ROADMAP; optima shift with scale). |
| **Token budget** | 800 steps × accum4 = **204,800 tokens**, val every 50 (17 curve points) | Matches the sweep screen → r2_small re-run reproduces it (a harness-faithfulness check) and r3/r2_mid are directly comparable. Pass-1; extend the winners if the big rungs haven't reached their knee. |
| **R0 gate** | Overfit each build on one pinned **real** batch **at the rung's own dims** (`run_r0` MODEL=`<rung>`), pass < 0.1 | The (size×config) gate the larger builds need: catches d512/d768-specific kernel/backward issues the tiny-dims sweep R0 can't. Architecture knobs were already R0-green at tiny dims. |
| **R0 recipe at scale** | adamw, **lr 1e-3, clip 1.0**, warmup 20, ~400 steps | The hot lr-2e-3 overfit *explodes in fp16 at d768* (x ~ ±124, attn grads → 0; clipping at the default 1.0 alone doesn't tame it). Lower lr + more steps drives a clean collapse. |
| **Checkpoints** | Distinct `--ckpt` per cell (harness already does `r2_<name>.bin`); mHC auto-stamps **v5** | No collision; v5 (`#if N_HC>1`) carries the mHC maps+Adam (ADR 0002). Plain/AdamW stay v4-shaped. |
| **Sequence length** | **Fixed at 256 across the whole ladder** | Keeps the token axis shared (above). Long-context V4 levers (CSA/HCA, partial-RoPE) are out of scope — they need seq≥2–8K (ADR 0001) and the sweep showed partial-RoPE *hurts* at 256. |

## Ceiling smoke test (r3_110m) — result

`emit_c r3_110m` → build plain (adamw) → overfit. **PASS**: d768/12L compiles
(10 dynamic kernels in 492ms), 109.5M params (85.0M transformer + 24.6M embed),
**~100 ms/step** all-ANE, loss 9.14 → 4.09 monotone with gradients flowing. The ceiling is
`r3_110m` — no fallback to the hand-written `stories110m.h` needed. At ~100ms/step a plain
800-step cell is ~80s; mHC cells (CPU residual conditioning) run slower, but the full
3-rung ladder is hours, not days — feasible sequentially on one ANE.

> fp16 caveat surfaced by the smoke: under an aggressive overfit the d768 residual stream
> grows large and attention grads underflow. This is an *optimization-stability* artifact of
> the hot overfit recipe, **not** a backward-correctness bug (loss decreases, grads are
> non-zero, embed path learns) — hence the tamed R0 recipe above. R2 training (lower lr,
> wd 0.1, warmup, random batches) does not hit it.

## What this measures (and what it does not)

- **Measures:** how the *r2_small efficiency ranking* (optimizer ≫ architecture; mHC↔Muon
  redundant; max loses) moves along the model-size axis at a fixed 256-token context and a
  fixed 205k-token budget. The interesting plots: (1) val-vs-tokens per (size×config) with
  the optimizer gap and the muon-vs-muon+mHC gap as functions of size; (2) tokens-to-target
  per size — bigger models should reach a lower target in fewer tokens *if* the levers hold.
- **Does not measure:** long-context behavior (seq fixed 256), 3B+ scale (ceiling 110M on
  one ANE), multi-seed variance (seed fixed; LR is the variance lever per ROADMAP), or a
  converged frontier (205k-token *screen*, the steep-descent regime — enough to *rank*).

## Consequences

- **One harness, any rung.** `MODEL=<rung>` selects `gen_<rung>.h` in both `run_r0`/`run_r2`;
  the zsh fixes (`${=var}` splitting, `printf '%s\t'`, `date +%s`) are preserved. r2_small
  with no `MODEL` still behaves exactly as before.
- **r2_small is re-run, not reused.** Same budget through the generalized harness → it must
  reproduce the sweep (muon@1e-2 ≈ 3.566). A mismatch means the generalization broke
  something — a built-in correctness check before the expensive rungs.
- **Pass-1 may under-resolve the big rungs.** 205k tokens is the early descent; a 110M model
  may still be warming up and look *worse* than 13M at equal tokens. That crossover is itself
  a finding; if the curves haven't separated we extend the budget on the winning configs only.
- **Result lands in `results/scale_ladder.md`** with per-(size×config) val-vs-tokens curves,
  the tokens-to-target table, and the verdict on hold/shrink/invert.
