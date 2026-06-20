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

| Date | Change | Gate | Before → after ms/step | Verdict |
|---|---|---|---|---|
| 2026-06-21 | baseline (stories110m, plain Muon) | R0 ✓ falls 9.14→7.07 | — / **~102** | baseline |
