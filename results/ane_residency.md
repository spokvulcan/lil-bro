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
| 2026-06-21 | fuse SiLU-bwd 9 vDSP passes → 1 loop | R0 ✓ / R1 **cos 0.99944** (clean re-verify) | `silu` 6.5→5.8 ms (−0.7) | **keep (verified, default-on)** |
| 2026-06-21 | ~~woFwd → function-param IOSurface (`WO_FUNCPARAM`)~~ | ~~R1 cos 1.00000~~ | ⚠️ **RETRACTED** | false pass — see Correction |
| 2026-06-21 | ~~ffnBwdW2t → function-param (`W2T_FUNCPARAM`)~~ | ~~R1 cos 1.00000~~ | ⚠️ **RETRACTED** | false pass — see Correction |
| 2026-06-21 | ~~conv datapath on ffnBwdW2t (`CONV_PROBE`)~~ | ~~R1 cos 1.00000~~ | ⚠️ **RETRACTED** | false pass — see Correction |
| 2026-06-21 | conv via SINGLE-input on wotBwd (`CONV1IN`, fixed gate) | R0 ✓ / R1 **cos 1.00000** (real clean-build diff) | `ane_bwd` 31.2→31.0 (=1) / 31.4 (=2) | **correct, but a WASH; transposes refuted as bottleneck** |

> ### ⚠️ CORRECTION (2026-06-21) — the func-param + conv "confirmations" were false passes
>
> Three of the rows above were **retracted after the equivalence gate was found
> broken**. Re-derived from primary evidence this session:
>
> 1. **The gate was comparing a binary against itself.** `gate_placement.zsh`
>    built `KNOB=0`, then `KNOB=1`, with **no `make clean` between** — and the
>    Makefile keys on source mtime, with `EXTRA` (`-D` flags) **not** a
>    prerequisite. So the second build was a no-op: both dumps came from the
>    `KNOB=0` binary. A self-compare returns *exactly* `cos 1.00000` — which is
>    why every "perfect" pass looked perfect. (Sanity control after the fix:
>    clean `KNOB=0` vs clean `KNOB=0` → `cos 1.00000, rel_l2 0`; clean `KNOB=0`
>    vs clean `W2T_FUNCPARAM=1` → **`cos 0.00000 @ L0.W1`**.) *Fixed:* `make
>    clean` before each build in `gate_placement.zsh`.
>
> 2. **The multi-input ANE binding does not execute on this hardware.** Every
>    func-param / conv kernel routed through `make_request_2in` (a 2-input
>    `requestWithInputs:@[act,W] inputIndices:@[@0,@1]`). That request is
>    **rejected at inference** on this M3 Max / Darwin 25.5:
>    `ANEProgramProcessRequestDirect() Failed with status=0x1d : statusType=0x9:
>    Program Inference error` (Code=8). The output surface stays zero → grads are
>    zero → `cos 0.0`. Compile succeeds; **eval fails**. Single-input
>    (`make_request`) evals fine — it is specifically the 2-input request.
>
> 3. **The failure was silent.** `ane_eval`/`ane_eval_req` discarded the
>    `evaluateWithQoS` `BOOL` and never printed the `NSError`, so a failed eval
>    looked identical to a successful one. *Fixed:* `ane_eval_check` now prints
>    the kernel + error and `abort()`s on `ok==NO` (`io.h`). Verified: a
>    `W2T_FUNCPARAM=1` build now aborts loudly with the status-0x1d error instead
>    of training on zeros.
>
> **Net:** the IOSurface function-param lever is **not "refuted by no-speedup"**
> — the mechanism never ran here (multi-input inference is the wall). The conv
> datapath is **neither confirmed nor refuted — it is UNTESTED**: the only conv
> path tried delivered its runtime weight as a 2nd input (broken). The "~1 ms
> faster" conv timing was the kernel *erroring out early*, not real work. The
> baseline buckets, determinism, and the SiLU fold are unaffected and stand.

### Conv datapath (PRD #26) — RESOLVED: correct on single-input, but a WASH

The conv reframe was retested honestly via the single-input binding (`CONV1IN`,
`gen_conv_1in_mil`), the only conv path that evals here. Result, with the **fixed**
gate (real clean-build `=0` vs `=1`/`=2` diffs) and independent timing:

- **Correct.** `cos 1.00000 / R1 PASS` for both `=1` (conv, one in-MIL weight
  transpose) and `=2` (conv, weight pre-transposed in CPU staging → ZERO in-MIL
  transposes). Conv compiles *and* evals on the single-input binding — MIL `conv`
  *does* accept a non-const weight sliced from the packed input. So PRD #26 was
  blocked only by the dead 2-in binding, not by conv itself.
- **But a wash.** `ane_bwd` median (stories110m, wotBwd, square IC=OC=768):
  matmul **31.2**, conv-1-transpose **31.0**, conv-0-transpose **31.4** ms — all
  inside run-to-run noise.
- **The transposes were never the bottleneck.** `=2` removes *all* transposes and
  is no faster than matmul. So `ane_fwd`+`ane_bwd` ≈ 51 ms is **ANE matmul
  compute + fixed per-eval dispatch**, not data layout. The "two transposes are
  the cost" hypothesis is **refuted by direct measurement.** Conv only wins where
  the activation dominates the weight (tall-skinny: large SEQ or IC≫OC) — and no
  current kernel is that shape (the classifier is the opposite, OC=32000≫).

**Strategic consequence — the easy/data-layout levers are exhausted.** Every
weight-layout idea (func-param IOSurface, conv, transpose elimination) is either
broken on this HW or a wash. The remaining wall-clock is real work in three
buckets, and the levers left are **architectural, not cosmetic**:

| Bucket | ms | Real lever (not yet tried) | Cost |
|---|---:|---|---|
| `ane_fwd`+`ane_bwd` | ~51 | **Fewer ANE evals** — fuse adjacent kernels so the ~96 eval/step dispatch count drops (per-eval overhead is now the suspect, not compute-per-eval); or genuinely less compute. | high (kernel fusion) |
| `io_fwd`+`io_bwd` | ~14 | **Keep activations ANE-resident** between kernels so the host↔ANE fp16 round-trip isn't paid per kernel. | high (residency rework) |
| `cls` | ~14 | classifier GEMM+CE → ANE (huge-vocab fp32-island LSE). | high |
| `silu`,`rms`/`rms_bwd` | ~11 | elementwise/reduction → ANE hardware `sigmoid` / reduction. | medium |

*Next experiment (recommended):* a cheap **dispatch-overhead probe** — measure how
`ane_bwd` scales with eval count (e.g. time one isolated kernel eval vs the full
chain) to confirm per-eval dispatch is the suspect *before* committing to kernel
fusion. If dispatch dominates → fusion is the big win; if compute dominates → the
51 ms is a hard floor and the reachable wins are the CPU buckets (cls, silu, rms).

**Why the multi-input request fails (open, for the next attempt).** Status
`0x1d` is opaque. Two hypotheses, untested: (a) the private
`requestWithInputs:inputIndices:…procedureIndex:` path genuinely doesn't honor a
2-input procedure on this OS; (b) the compiled MIL `func main(x, w)` signature /
input-index mapping doesn't match the request binding. Upstream PR #22 (the
function-param IOSurface lever) reportedly works elsewhere, so it may be OS/HW
version or a plumbing detail — but on *this* box, single-input is the only path
proven to eval.

**SiLU finding (stands, re-verified clean).** Pre-fusion (`561cb78`, 9-pass
vDSP) vs current fused default → **`cos 0.99944 @ L2.rms_ffn, R1 PASS`**
(separate clean builds, not a self-compare — a self-compare would be exactly
1.0). The bucket is *not* dominated by the 9 elementwise passes (fusing saved
only 0.7 ms) — it's the **sigmoid setup** (`vvexpf`+`vvrecf`). The real `silu`
lever is the ANE's hardware `sigmoid` (fold SiLU-bwd into the FFN-bwd ANE kernel,
P3), not CPU micro-opt. The 0.99944 (not 1.0) is benign FMA contraction at -O2 —
*more* accurate than separate-rounding vDSP, well inside the cos≥0.99 gate.
