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
| ~~`silu` SiLU-bwd fold into FFN-bwd~~ | ~~~6.5 + a round-trip~~ | done | **DONE (`FUSE_SILU_BWD`, −2.0 ms/step, R1 cos 0.99853)** — folded into `ffnBwdW13t`; the round-trip tax (read dh1/dh3 back for dW) ate ~1.8 ms, leaving net −2.0. Next: recompute dh1/dh3 in the async dW closure to drop the read-back (gated on serial `dw_q` slack). |
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
| 2026-06-21 | **kernel fusion**: re-fuse qBwd+kvBwd → 1 eval (`FUSE_QKVBWD`) | R0 ✓ / R1 **cos 0.99996** (fixed gate, both verified independently) | **`ane_bwd` 31.1→29.1 ms (−2.0/step, −6.3%)** | **WIN — first validated speedup; lever confirmed** |
| 2026-06-21 | **vertical fusion**: fold SiLU-bwd into `ffnBwdW13t` on ANE (`FUSE_SILU_BWD`); kernel outputs `concat(dx,dh1,dh3)`, `dsilu` flows W2t→W13t via strided ANE→ANE copy | R0 ✓ falls to 3.09 / R1 **cos 0.99853 @L9.Wq, rel_l2 0.054** (fixed gate; worst is a non-FFN weight = correct change propagating fp16-silu rounding upstream) | `silu` **5.6→0.0**, `ane_bwd` +1.8 (in-kernel silu ops + 6× bigger output), `io_bwd` +1.8 (dh1/dh3 read-back for dW) ⇒ **−2.0 ms/step bucket-sum, −2.7 step-median** (9 samples each) | **WIN — 2nd validated speedup; the *vertical* lever confirmed. Stacks with `FUSE_QKVBWD`.** |

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
| `ane_fwd`+`ane_bwd` | ~51 | **Fewer ANE evals (kernel fusion)** — MEASURED dispatch-bound: ~0.12 ms fixed overhead/eval, compute near-free to n=1024, fusion PoC saves 38% on 2 matmuls (see below). ~96 evals/step ⇒ ~11 ms is pure dispatch. **The top lever.** | high (kernel fusion) |
| `io_fwd`+`io_bwd` | ~14 | **Keep activations ANE-resident** between kernels so the host↔ANE fp16 round-trip isn't paid per kernel. | high (residency rework) |
| `cls` | ~14 | classifier GEMM+CE → ANE (huge-vocab fp32-island LSE). | high |
| `silu`,`rms`/`rms_bwd` | ~11 | elementwise/reduction → ANE hardware `sigmoid` / reduction. | medium |

### Dispatch-overhead probe — RESULT: the ANE is DISPATCH-BOUND; fusion is the lever

`probe_dispatch.m` (standalone) times real single-input matmul kernels in a tight
eval loop. Two measurements, both stable across runs:

**[1] Per-eval time is flat across 1000× of compute.** Square `ic=oc=n` matmul at
`seq=256`, per-eval ms:

| n | 32 | 64 | 128 | 256 | 512 | 768 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| FLOPs ×768 | .002 | .007 | .028 | .111 | .444 | 1.0 | 1.78 | 7.11 |
| ms/eval | ~0.30 | ~0.30 | ~0.30 | ~0.22 | ~0.16 | **0.15** | ~0.19 | ~0.46 |

From n=32 to n=1024 the FLOPs rise ~1000× yet per-eval time is **flat ~0.15–0.30 ms**
(n=512/768 are the *fastest*). It only rises at n=2048 (and that surface is 9.4 MB —
likely memory bandwidth, not compute). Caching is ruled out: a cache would keep
n=2048 flat too. ⇒ **matmul compute is nearly free; the per-eval cost is fixed
dispatch/overhead.**

**[2] Fusion PoC — 2 matmuls in one eval vs two evals (n=768):**

| | ms |
|---|---|
| 1 matmul / eval | 0.153 |
| 2 matmuls / eval (fused, single dispatch) | 0.189 |
| 2 separate evals | 0.307 |
| **fused saves** | **0.118 ms (38%)** |

The 2nd matmul added only **0.035 ms** of marginal compute; a separate eval would
have added 0.153 ms. So **~0.12 ms of every eval is pure dispatch overhead** that
fusion eliminates, and ~77% of a 768-eval's cost is dispatch, ~23% compute.

**THE LEVER (measured, not assumed): kernel fusion.** The step does ~96 ANE
evals (≈8 kernels × 12 layers); at ~0.12 ms dispatch each that's **~11 ms of pure
dispatch** in the 52 ms `ane_fwd`+`ane_bwd` bucket, *plus* the inter-kernel
host↔ANE staging fusion also removes from `io_fwd`+`io_bwd` (~14 ms). Fusing
adjacent matmuls (e.g. the per-layer QKV projections into one kernel, or the FFN
backward matmuls) cuts the dispatch count directly. This is the first lever this
session with measured headroom and a working mechanism — and it needs no
multi-input binding (a fused kernel is one single-input program with a bigger
packed surface, exactly the n=2048-class kernels that already eval fine).
*Incidental:* the ANE rejects very small/odd program shapes with the same
`status=0x1d` (e.g. 16×16×16) — so 0x1d is a general "program rejected", not
unique to multi-input; the multi-input wall may yet be a plumbing detail (below).

### Fusion lever VALIDATED — qkvBwd re-fusion, −2.0 ms/step (`FUSE_QKVBWD`)

First fusion applied and measured end-to-end. `qBwd` (dq@Wq → dx_attn) and
`kvBwd` (dk@Wk + dv@Wv → dx_kv, then CPU `dx_attn += dx_kv`) are independent
backward projections that sum into `dxnorm`; they were split from one `qkvBwd`
kernel in `475348a` purely for GQA (Q_DIM≠KV_DIM), not correctness. Re-fused on
MHA (stories110m) into ONE single-input kernel computing
`dx_attn = dq@Wq + dk@Wk + dv@Wv` (`gen_qkv_bwd_fused_mil`), removing 1 eval/layer
(×12) + the CPU add + one fp16 pack.

- **Correct** (fixed gate, verified twice independently): `R1 PASS, cos 0.99996
  @L0.rms_ffn` (rel_l2 0.0092). Not exactly 1.0 — and that's the *honest*
  signature of a real fused-vs-split diff: the 3-way in-kernel fp16 `add` reorders
  the reduction vs the split's 2-way+CPU-fp32 add. Same benign class as the SiLU
  fold (0.99944), well inside the cos≥0.99 gate. R0 overfit trains to loss 1.43.
- **−2.0 ms/step** (independently measured, 9 samples each): `ane_bwd` median
  31.1 → 29.1 (mean 32.7 → 30.4; min 30.3 → 28.8 — every order statistic shifts
  down ~2 ms). Matches the prediction: 12 evals × ~0.12 ms dispatch + the removed
  CPU add. Default-0 (opt-in); MHA-only (`#error` guards GQA).

**This converts the dispatch-bound hypothesis into a banked win and a rollout
rationale.** Caveat for the rollout: the *easy* horizontal fusions are limited —
the forward is already fused (sdpaFwd packs QKV+attention, ffnFused packs
W1+W3+SiLU+W2), and the remaining backward kernels are mostly *sequential*
(ffnBwdW2t→ffnBwdW13t; wotBwd→attn-bwd→qkvBwd), which can't share one eval. The
larger headroom is **vertical fusion** (folding a sequential chain into one
kernel, as sdpaFwd already does) — higher value but harder, and gated by whether
the intermediates (softmax-bwd, SiLU-bwd) are ANE-expressible. qkvBwd was the
cleanest first cut; it proves the mechanism and the measurement.

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
