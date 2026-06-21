# V2 ANE residency — measured baseline + lever ranking

> Profiling note for the V2 ANE-resident trainer (issue #12, ADR 0004). This is
> the **before** row every migration must beat. Throughput-only: it changes
> *where* ops run, never the loss-vs-tokens curve.

## Setup

- **Hardware:** Apple M3 Max (16-core ANE, 12P+4E CPU, 48 GB), Darwin 25.5.
- **Model:** `stories110m` (12L, MHA, DIM 768, SEQ 256, 109M) — the PRD's headline
  "winning config at 110M". Built `make MODEL=stories110m`, plain Muon/AdamW, no
  V4 knobs (`ATTN_SINK=QK_NORM=N_HC-1=MTP_DEPTH=0`).
- **Command:** `./train --scratch --steps 30` on `tinystories_data00.bin`.
- **Compile:** 10 kernels, one-time ~0.4–0.6 s (no recompile — dynamic pipeline).

## Per-step buckets (steady state, step 10–20)

The `timing:` line at `train.m:1652`. Buckets are **sequential** — they sum to
the step (~102 ms), except `cblas_wait` which is the dW async barrier.

| Bucket | ms | Where it runs | Category |
|---|---:|---|---|
| `ane_bwd` | 31.2 | ANE | compute — already ANE |
| `ane_fwd` | 19.9 | ANE | compute — already ANE |
| `cls` | 15.0 | CPU (cblas + CE) | **migratable — classifier fwd GEMM + CE + bwd GEMM (P2)** |
| `io_bwd` | 9.8 | host↔ANE | IOSurface round-trip (fp16↔fp32 + pack) |
| `silu` | 6.5 | CPU (vDSP) | **migratable — SiLU/SwiGLU backward, elementwise (P3)** |
| `io_fwd` | 4.3 | host↔ANE | IOSurface round-trip |
| `rms_bwd` | 4.1 | CPU (vDSP) | **migratable — RMSNorm backward (P1)** |
| `dw_copy` | 2.6 | CPU | floor — capture-buffer churn for async dW |
| `rms` | 1.8 | CPU (vDSP) | **migratable — RMSNorm forward (P1)** |
| `cblas_wait` | **0.0** | CPU | **floor — dW already fully overlapped** |
| *(unaccounted)* | ~7 | CPU | embedding gather, optimizer, loss-scale, misc |

Measured sum of named buckets ≈ 95 ms; total step ≈ 102 ms.

## Findings that shape the plan

1. **`cblas_wait = 0.0` → the dW CPU floor is already hidden.** The async-GCD
   weight-gradient dispatch (`dw_grp`) fully overlaps with the next step's ANE
   forward. So *migrating the floor onto the ANE buys ~nothing in wall-clock* —
   it is already off the critical path. This is the concrete reason "100% ANE"
   is the wrong target (it would chase already-hidden work).

2. **The step is a hard sequential chain `fwd → cls → bwd`.** The only
   cross-step overlap (dW) is already taken. Therefore wall-clock can only fall
   if a bucket gets genuinely *faster* — real ANE migration (where the ANE beats
   CPU for that op) or a faster/fused CPU path. There is no free overlap left.

3. **The migratable CPU residual is `cls`(15) + `silu`(6.5) + `rms`+`rms_bwd`(5.9)
   ≈ 27 ms (~27% of step).** Plus the IOSurface round-trip `io_fwd`+`io_bwd`
   ≈ 14 ms (~14%) — overhead, not compute, targeted by the function-parameter
   IOSurface lever (upstream PR #22, measured −30%).

## Lever ranking (by measured wall-clock at risk)

| Lever | Target ms | Risk | Notes |
|---|---:|---|---|
| IOSurface function-param staging | ~14 (io) + kernel speedup | high (rewrite) | upstream −30%; biggest single lever |
| `cls` classifier-fwd + CE → ANE | ~7 (half of cls) | high (fp32-island LSE) | PLAUSIBLE-BUT-UNVERIFIED; prior art PR #19 |
| `silu` SiLU-bwd fold into FFN-bwd | ~6.5 + a round-trip | medium (elementwise) | ANE-friendly; fwd already runs SiLU on ANE |
| `rms`/`rms_bwd` RMSNorm → ANE | ~5.9 | medium (reduction) | pathfinder for fp32-island recipe; ANE reductions are the "wrong axis" — win uncertain |
| `dw_copy` buffer pooling | ~1–2 | none (identical math) | pre-allocate persistent capture buffers; safe momentum win |

## Honest ceiling

Zeroing **every** migratable + IOSurface bucket floors the step at
`ane_fwd + ane_bwd + irreducible-CPU` ≈ 51 + ~7 = ~58 ms, i.e. a hard ceiling
near **1.75×**. The realistic, measured target is the lower band **~1.3–1.5×**,
because (a) migrated ops add ANE time, not zero, and (b) reductions (RMSNorm, CE)
may not beat CPU on the ANE at all. Same loss-vs-tokens curve, reached faster in
wall-clock — never a token-efficiency claim.

## Migration log

| Date | Change | Gate | Before → after | Verdict |
|---|---|---|---|---|
| 2026-06-21 | baseline (stories110m, plain Muon) | R0 ✓ falls 9.14→7.07 | **~102 ms/step** | baseline |
| 2026-06-21 | fuse SiLU-bwd 9 vDSP passes → 1 loop | R0 ✓ / R1 cos 0.99944 | `silu` 6.5→5.8 ms (−0.7) | keep (minor) |
| 2026-06-21 | woFwd → function-param IOSurface (`WO_FUNCPARAM`) | R0 ✓ / R1 **cos 1.00000** | no Δ (smallest weight) | mechanism proven; default-off |
| 2026-06-21 | ffnBwdW2t → function-param (`W2T_FUNCPARAM`, W2=768×2048) | R0 ✓ / R1 **cos 1.00000** | `ane_bwd` 31.0→30.5, `io_bwd` 9.8→10.0 (noise) | **lever refuted here**; default-off |

**IOSurface function-param lever — REFUTED on this config (measured).** Removing the
in-kernel weight slice/reshape gives **no wall-clock delta** on either the smallest
weight (woFwd, 768×768) *or* the largest simple weight (ffnBwdW2t, 768×2048) — both
bitwise-correct, both ~0 ms. The per-kernel unpack is not a measurable fraction of
eval time on the M3 Max ANE (most likely the compiler already constant-folds the
static-offset slice). The upstream PR #22 −30% does **not** reproduce on
stories110m/M3 Max. The full multi-kernel rollout is therefore **not worth it** — the
measurement saved that effort. The multi-input *mechanism* (`compile_kern_mil_2in`,
`gen_matmul_2in`) is kept: the SiLU-fold and ANE-classifier need it to move *compute*
(not just weight layout) onto the ANE, a different value proposition.

**Revised hypothesis for the real wall-clock:** `ane_fwd`(20)+`ane_bwd`(31)=51 ms is
the ANE matmul datapath itself, and `gen_dyn_matmul` wraps every matmul in **two
transposes** (`reshape→transpose→matmul→transpose→reshape`) to hit the `[SEQ,IC]@[IC,OC]`
orientation. The ANE is natively a convolution engine; a **1×1 conv** consumes the
`[1,IC,1,SEQ]` layout directly with no transpose (PRD #26). That targets the *biggest*
bucket (51 ms) and is the next experiment.

**IOSurface lever — mechanism proven.** Multi-input MIL binding works: a
`func main(x, Wo)` with `requestWithInputs:@[act,Wo] inputIndices:@[@0,@1]` and `Wo^T`
staged as a contiguous `[1,1,Q_DIM,DIM]` surface gives **bitwise-identical** grads
(cos 1.0, rel_l2 0). woFwd alone is unmeasurable (Wo is 768×768, the smallest weight),
but this unlocks the −30% rollout to the big-weight kernels (FFN W1/W2/W3 = 768×2048,
SDPA Wq/Wk/Wv) and is the same multi-surface plumbing the SiLU-fold and classifier need.
Reusable: `compile_kern_mil_2in`, `make_request_2in` (`io.h`), `gen_wo_fwd_2in` (`mil_dynamic.h`).

**SiLU finding:** the bucket is *not* dominated by the 9 elementwise passes (fusing
them saved only 0.7 ms) — it's dominated by the **sigmoid setup** (`vvexpf` + `vvrecf`,
4 vectorized passes left intact). So the real lever for `silu` is the ANE's hardware
`sigmoid`, i.e. folding SiLU-backward into the FFN-bwd ANE kernel (P3), not CPU
micro-opt. R1 cos 0.99944 (not 1.0) is benign FMA contraction at -O2 — *more* accurate
than the separate-rounding vDSP, well inside the cos≥0.99 gate.
