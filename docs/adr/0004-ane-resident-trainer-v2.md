# ADR 0004 — V2 trainer: drive the winning config fully onto the ANE ("ANE-resident" training)

- **Status:** Proposed — forward-looking design, not yet implemented. Records the V2
  target and the evidence-ranked op-migration plan so the build order is decided before
  any kernel work. **Refined 2026-06-21** against external prior art, the upstream
  `maderix/ANE` PR/issue landscape, and a re-verification of every internal citation —
  full evidence in the companion [research dossier](0004-ane-resident-trainer-v2-research.md).
  (Created under autonomous direction, in the spirit of ADR 0003; ratify via
  `/grill-with-docs` before the first kernel lands if the priorities are contested.)
- **Date:** 2026-06-21
- **Context docs:** [research dossier](0004-ane-resident-trainer-v2-research.md) (cited
  evidence for every claim below) · [results/scale_ladder.md](../../results/scale_ladder.md)
  (the winner this targets) · [ADR 0003](0003-v4-winner-scale-ladder.md) (scale ladder +
  tokens-to-target) · [ADR 0001](0001-deepseek-v4-for-dense-ane.md) (V4 tier list; mHC
  CPU-first placement) · [ROADMAP.md](../../ROADMAP.md) (R-ladder + method invariants;
  "measure ANE utilization yourself") · [training/README.md](../../training/README.md)
  (trainer internals). **External:** Orion (arXiv 2603.06728) — an independent from-scratch
  ANE trainer with the *same* CPU/ANE split; Apple "Deploying Transformers on the ANE."
- **Supersedes nothing.** Extends the trainer that produced the ladder; it changes *where*
  ops run, not the math or the headline metric.

---

## Context

The scale ladder (ADR 0003) settled the winner: **plain Muon**, and it holds 13M → 110M.
The natural next question — the operator's — is **speed**: "make everything run on the ANE;
CPU is too slow for quick efficient learning." Before committing kernel work, profile the
*winner* to learn where the time actually goes. The per-step `timing:` instrumentation
(`train.m:1652`) measured this session, on the ladder logs, says the intuition is **half
right** — and the precise half matters.

### What's measured (per-step `timing:` buckets, plain Muon, this session)

| rung | ANE (`ane_fwd`+`ane_bwd`) | `rms`+`rms_bwd` | `cls` | `io_fwd`+`io_bwd` | `silu` | step |
|---|---|---|---|---|---|---|
| r2_small 13M | 18.3 ms | 1.1 ms | 9.9 ms | 2.2 ms | 1.2 ms | ~33 ms |
| r2_mid 42M | 40.9 ms | 2.7 ms | 12.0 ms | 6.7 ms | 3.0 ms | ~62 ms |
| r3_110m 110M | 52.4 ms | 5.8 ms | 11.7 ms | 14.6 ms | 6.3 ms | ~93 ms |

**The winning config is already ANE-bound** — ANE is the single largest bucket at every
size (~56% of the step at 110M). The CPU residue, in order: the **classifier head +
cross-entropy** (`cls`, `cblas_sgemm` at `train.m:1296` + `cross_entropy_loss` — the largest
CPU item, and *nearly constant* across sizes because it's fixed by the vocab width — the
dynamic trainer already compacts 32K→~9.2K active (`training/README.md:199`) — so it is
largely non-amortizing overhead), the **IOSurface host↔ANE copies + fp16↔fp32 conversion**
(`io`, which *grows* with size and with every op we offload), **RMSNorm** fwd/bwd
(`cpu_ops.h:7-54`, vDSP), **SiLU** backward (`train.m:1379`, vDSP), **RoPE** (`cpu_ops.h`),
**embedding** lookup/backward, and the **optimizer step** (Muon Newton–Schulz / AdamW,
`cpu_ops.h:56-145`, CPU, every `--accum`; not in the per-step bucket).

### The target, corrected: drive to the CPU *floor*, not to 100%

Two independent from-scratch ANE trainers — this one and **Orion** (arXiv 2603.06728,
Mar 2026; M4 Max, Stories110M, 1000 steps, 0 NaN) — land on the **same division of labor**:
forward + activation-gradient (`dx`) backward on the ANE; **`dW` weight-gradients,
cross-entropy/loss, classifier-backward, embedding gather, and the optimizer on CPU**
(Orion Table 5 ≡ lil-bro `train.m:1227-1666, 1814-1823`). No observed system has moved `dW`
or the optimizer onto the ANE, and **MIL has no loss / gradient / optimizer operator at all**
(coremltools iOS15 op set). That set is an **irreducible CPU floor**. So V2's goal is *not*
"~100% ANE" — it is **"shrink the CPU residual and the CPU↔ANE round-trips down to that
floor."** lil-bro already does the only thing available for the floor itself: it **overlaps
`dW` (async GCD `cblas`) with the next step's ANE forward** (`train.m:1227, 1644`).

### The one place CPU dominates catastrophically — mHC

Same dims, mHC on vs off, only difference is `N_HC`:

| rung | plain `rms`+`rms_bwd` | mHC×4 `rms`+`rms_bwd` | mHC step vs plain |
|---|---|---|---|
| r2_small 13M | 1.1 ms | **287 ms** | ~10× |
| r3_110m 110M | 5.8 ms | **1882 ms** | ~22× |

The mHC block (N_HC-stream RMSNorms + maps A/B/C + the iterative Sinkhorn) runs **entirely
on CPU** — `mhc.h:1` ("CPU-first per ADR 0001"), placed there because "the iterative 4×4
Sinkhorn fights fixed-shape MIL; lowest new-kernel risk" (ADR 0001 line 92). It lands in the
`rms` bucket and balloons it. Wall-clock per cell: mHC×4 `363s → 3016s` vs plain `77s → 1216s`
across the ladder. **And the ladder shows mHC is redundant with Muon** (≤0.05 nats, often
*worse*; at 110M a ≤0.024-nat hint of crossover, still being bracketed). So mHC is the worst
offender on *both* axes — most expensive on CPU, least proven in tokens-to-target.

### Why ops are on CPU today (cited, so V2 attacks the real blocker)

- **Attention is already fully ANE** — QKV, RoPE, GQA tiling, scores, **causal mask applied
  as an additive bias** (`mil_dynamic.h:188-190`, sidestepping the documented "ANE ignores
  the native SDPA mask" trap), softmax, and ×V all run in `gen_sdpa_fwd_dynamic`. It drops to
  CPU only via `ATTN_CPU (ATTN_SINK || QK_NORM)` (`config.h:61`) for the two V4 knobs the
  softmax kernel can't express — both default-off, both *hurt* at seq256 (sweep).
- **mHC** — CPU-first by ADR 0001 (above), not a hardware "no": a *risk* decision to reach an
  overfit-green gate. The Sinkhorn dynamic-shape concern is real but addressable
  (fixed-iteration unroll; `mhc_sinkhorn_spike.c` already exists).
- **RMSNorm / RoPE-bwd / SiLU-bwd** — vectorized-Accelerate conveniences, **not** hard ANE
  blocks; MIL has the primitives (`rsqrt`, `l2_norm`, `reduce_*`). (RoPE *forward* is already
  ANE inside SDPA; only the bwd re-rotation is CPU — `cpu_ops.h:480`.)
- **classifier+CE and the optimizer are a different category** — there is **no MIL
  cross-entropy / loss / gradient / optimizer operator** (coremltools iOS15), so these can't
  be a drop-in port: any ANE version must be **hand-built from forward MIL primitives**
  (`reduce_log_sum_exp` exists for a stable CE) or stay CPU. That IR gap — not laziness — is
  *why* every observed ANE trainer keeps them on CPU.
- **The binding hardware facts** (not "can't", but they shape every kernel): ANE wants
  `(B,C,1,S)` fp16 with the **small dim off the last axis** (64-B align; 32× memory penalty
  for a singleton last axis); its native primitive is **conv, not matmul** (lil-bro emits
  matmul today — `mil_dynamic.h:31,184` — so 1×1-conv is an *unexploited, unproven* datapath
  lever); and **fp16 backward gradients underflow to ~0** without loss-scaling — lil-bro
  already carries `loss_scale=256` + grad-clip (`train.m:898-899, 1318, 1667`), which the
  community confirms is load-bearing (training explodes ~step 130 without it).

### Honest framing of "quick efficient learning"

Moving an op CPU→ANE changes **throughput** (wall-clock tokens/sec), **not** tokens-to-target
(the loss-vs-tokens curve is where-agnostic — set by optimizer/architecture, ADR 0003). So V2
is: **same learning curve, reached much faster in wall-clock**, which is exactly "CPU is too
slow → make it fast," and *indirectly* improves the achievable frontier (more LR-sweeps and
tokens per human-hour; mHC viable enough to actually test long-context). V2 must not claim it
makes learning more token-efficient — only faster. And the ceiling is **bounded**: with `dW`
+ optimizer + CE + embedding on the CPU floor (above), V2 cannot reach "100% ANE" — the honest
target is the *floor*, and the honest metric is wall-clock at equal tokens-to-target, not an
ANE-residency percentage.

## Decision

Build **V2 = an ANE-resident trainer**: drive the **winning config (plain Muon)** down to the
**CPU floor** (not "100%") by (1) porting the hot, *expressible* CPU residuals onto the ANE —
**RMSNorm and classifier-forward**, reusing the proven static-trainer kernels
(`ane_rmsnorm_bwd.h`, `ane_classifier.h`) re-expressed in the dynamic MIL generator, each
behind an R0 fp16-correctness gate, **RMSNorm first as the lower-risk pathfinder** for the
fp32-island technique; (2) cutting the **IOSurface round-trip tax** with **function-parameter
IOSurfaces** (upstream PR #22, measured **−30%**), which delete the `slice→reshape→transpose`
weight-unpack lil-bro pays on every kernel — **not** const()-weight mega-kernel fusion, which
would re-introduce the per-step recompile lil-bro's compile-once design avoids and whose
"94% util" promise was refuted in review; and (3) leaving the **CPU floor** (`dW`, optimizer,
cross-entropy, embedding) on CPU but **overlapped**, since no MIL operator exists for it and
no system has moved it. mHC gets a **decision fork**, not an automatic port.

**Explicitly off the table** — they solve a problem lil-bro already solved (it is compile-once;
no `exec()` restart, no 119-compile limit; `training/README.md:38`): async double-buffering
(#23), ACCUM-for-compile-amortization (#33/#24), and Orion delta-compilation.

### Op-migration plan (ranked; feasibility-tagged — see dossier §6)

| P | Op | Today (file:line · measured) | V2 target | Rationale / risk · evidence |
|---|---|---|---|---|
| **P0** | **mHC block** | CPU `mhc.h:1` · `rms` 287→1882 ms (10–22× tax) | **Fork: ANE kernel *or* retire from hot path** | Biggest CPU cost *and* ladder says redundant w/ Muon. Do **not** burn kernel-risk until a **long-context rung** shows it earns value; until then it's opt-in CPU, off the winner's path. If ported: fixed-iteration unrolled Sinkhorn (`mhc_sinkhorn_spike.c` exists) kills the dynamic-shape blocker. *(No external evidence bears on mHC; verdict unchanged.)* |
| **P1** | **RMSNorm fwd+bwd** | CPU vDSP `cpu_ops.h:7-54` · `rms`≈1–6 ms plain (the bucket mHC explodes) | ANE kernel, **fused into the adjacent QKV/FFN entry kernel** | On the critical path; fusing removes an IOSurface hop. **Prior art:** `ane_rmsnorm_bwd.h` already does this on the *static* trainer → re-express in dynamic MIL. **Risk:** fp16 reduction variance + documented ANE `layer_norm`-overflow precedent → **fp32 island likely required**; R0 at d768 decides. *(PLAUSIBLE-BUT-UNVERIFIED — the pathfinder for the fp32-island recipe.)* |
| **P2** | **Classifier head + cross-entropy** | CPU `cblas_sgemm` `train.m:1296` + `cross_entropy_loss` `cpu_ops.h:151` · `cls`≈10–12 ms | ANE GEMM/conv for logits **forward** + fp32-island stable-LSE CE; **classifier-bwd stays CPU** | **Prior art:** `ane_classifier.h` has `gen_classifier_fwd` + a 32K `gen_softmax_vocab` (static). **But:** (a) the dynamic trainer already **vocab-compacts 32K→~9.2K** (`training/README.md:199`), so the ANE port must beat a *shrunk* CPU baseline, and the per-batch-dynamic active count fights fixed-shape MIL; (b) both Orion + maderix keep **classifier-backward on CPU** ("ANE matmul slower than cblas for this shape"); (c) fp16 32K-wide softmax/LSE correctness is **unverified** (MIL `reduce_log_sum_exp` exists; on-device fp16 is the open risk). R0 gate first. *(PLAUSIBLE-BUT-UNVERIFIED; demoted below RMSNorm.)* |
| **P3** | **SiLU/SwiGLU bwd** | partly fused fwd (`ffnFused`); bwd vDSP `train.m:1379` · `silu`≈1–6 ms | fold bwd into the FFN-bwd ANE kernel | Small, removes a round trip; the fwd is already ANE. Low risk. |
| **P3** | **RoPE bwd** | CPU `cpu_ops.h:480` | fold into the SDPA-bwd ANE kernel (cos/sin as kernel input) | RoPE **fwd is already ANE** (`mil_dynamic.h:131-168`); only the bwd re-rotation is CPU. *(Earlier `cpu_ops.h:401` cite was the MTP-only path — corrected.)* |
| **infra** | **IOSurface I/O + fp16↔fp32** | `io`≈2–15 ms, grows with size; spatial-pack unpack tax `mil_dynamic.h:18-34` | **Function-parameter IOSurfaces** (PR #22) + keep `io_copy` chaining | **The real lever.** PR #22 deletes the `slice→reshape→transpose` per-weight (≈12 ops/attn kernel) → **−30% (76.9 vs 110 ms/step)**. lil-bro already has host-memcpy chaining (`io_copy` `io.h:63`). **Risk:** PR #22 closed-unmerged → re-validate; runtime-weight op constraints (#47: inner-dim ×32, ascending/descending slots → *silent zeros*) must be asserted. On-device chaining (`_ANEChainingRequest`) is **BLOCKED** for in-mem MIL (#40). |
| **datapath** | **matmul → 1×1 conv** | matmul today `mil_dynamic.h:31,184,197` | experiment: emit conv on the hot GEMMs | ANE's native primitive is conv (Apple); one source claims 3×, but `smpanaro` found "no help" when bandwidth-bound. **Unproven at this scale → measure, don't commit.** |
| **floor** | **`dW` + optimizer (Muon NS / AdamW)** | CPU `cblas` `train.m:1322-1585`; `cpu_ops.h:56-147` | **keep on CPU, overlap** (already async GCD) | **CPU floor.** No MIL gradient/optimizer op; CPU-resident in Orion *and* lil-bro *and* maderix. `dW` already overlapped with next-step ANE forward (`train.m:1227, 1644`). Don't spend kernel risk here. |
| **floor** | **Embedding lookup + bwd** | CPU loops `cpu_ops.h:259-273` | **keep on CPU** | Gather/scatter is ANE-awkward; cost tiny. PR #39 offers a **~12×, bit-exact** CPU cache-opt if it ever surfaces in the timing. |
| **constraint** | **ATTN_CPU (sink / QK-norm)** | CPU bypass `config.h:61` (default off) | ANE kernel for sink-denominator + QK-norm, **or leave off the winner** | Plain attention (incl. causal mask + softmax) is **already ANE**; only these two knobs bypass. Both *hurt* at seq256 (sweep). Only worth a kernel if a long-context rung revives them. |
| ~~fusion~~ | ~~mega-kernel fusion~~ | — | **REMOVED as a residency lever** | const()-weight fusion (#24) forces the per-step recompile lil-bro avoids; "94% util" claim **refuted 0-3**. Activations stay resident via PR #22 + `io_copy`, not mega-kernels. |

### Method invariants (carried from ROADMAP / ADR 0003, sharpened by the evidence)

- **R0 correctness gate at the rung's own dims for every new ANE kernel** before its R2
  numbers are trusted. Not ceremony: upstream PR #26 shipped an ANE conv that **compiled and
  ran but returned numerically wrong output** (fell back to CPU BLAS), and ANE fp16
  `layer_norm` overflow is documented — "compiles ≠ correct" is the live failure mode. fp16
  reductions (RMSNorm, 32K CE) are the precision risk; a kernel that changes the math fails R0
  (overfit must still collapse to ~0).
- **Measure residency from the per-step `timing:` buckets — and cross-check.** V2 success =
  the plain-config CPU buckets (`rms`+`silu`+`cls`) and the `io` tax → **down to the CPU
  floor** (*not* "ANE fraction → 90%+", which the floor makes unreachable). Caveats the
  research surfaced: the ~56% figure is lil-bro's *own* instrumentation, externally
  unvalidated; community ANE-util% are *bogus* (hardcoded M4 denominator → 37000% values,
  #46); and because lil-bro drives the ANE through **private APIs, not CoreML**, the Xcode
  Instruments Core ML template **cannot see it**. Cross-check the buckets against
  `powermetrics` ANE *power* — "ANE-bound" should rest on two independent signals.
- **Tokens-to-target stays the headline** — V2 is graded on wall-clock at equal
  tokens-to-target, with R0-green correctness; it must not move the loss-vs-tokens curve.

## Consequences

- **Bounded, real ROI — but ~1.3–1.5×, not 3×.** At 110M the CPU/transfer on the critical
  path is `cls`(≈12) + `io`(≈15) + `rms`(≈6) + `silu`(≈6) ≈ **39 ms of a ~93 ms step**; the
  ANE compute (~52 ms) and the CPU floor (CE + a residual `io` boundary) **cannot** be removed.
  Even zeroing every migratable bucket floors the step near ~54 ms (a **~1.7× ceiling**);
  realistically, function-param IOSurfaces (−30% on the matmul kernels) + RMSNorm/SiLU-bwd
  folding + classifier-fwd (*if* fp16-correct) take ~93 ms toward **~65–70 ms (~1.3–1.5×)**,
  with no accuracy change. The original "3× step-time cut" was inconsistent with the bucket
  arithmetic and is **withdrawn**; each kernel still earns its place in *measured* ms.
- **mHC stays honest.** "Everything on ANE" does **not** mean "port the mechanism the data says
  doesn't pay." mHC is gated behind a long-context rung; on the size axis it's retired from the
  hot path. Keeps V2 from spending its hardest kernel (Sinkhorn) on a redundant lever.
- **Precision and the IR gap are the gating risks, not "feasibility".** The fp16 numerics (32K
  CE log-sum-exp, RMSNorm reductions → fp32 islands built by explicit `cast` ops, since MIL has
  no per-op precision attribute) are the live unknown — R0 at each rung is the guard. And
  `dW`/optimizer/CE have **no MIL operator**, so the CPU floor is structural, not a to-do.
- **Result lands as a V2 profiling note** (`results/ane_residency.md`): per-step buckets
  before/after each migration, the ANE-vs-CPU-floor split, and wall-clock tokens-to-target vs V1
  at equal loss — cross-checked against `powermetrics` ANE power, so each kernel earns its place
  in measured ms.

## Open questions (resolve before / during build)

1. **fp16 execution correctness (the deciding risk):** loss-scaling is *already in place*
   (`train.m:899`) and MIL exposes a stable `reduce_log_sum_exp`, so the math is expressible —
   but does a **32K-wide softmax/LSE and RMSNorm actually execute correctly on the ANE in
   fp16**, or overflow/underflow and need explicit fp32-`cast` islands? Documented ANE
   `layer_norm` overflow cuts against it. **R0 at d768 answers it** — RMSNorm is the pathfinder
   (its fp32-island recipe informs the CE).
2. **Fusion ceiling — partly answered:** native on-device chaining (`_ANEChainingRequest`) is
   **blocked** for in-memory MIL (#40), so activations stay resident only via (a) bigger single
   MIL programs within the runtime-weight op constraints (#47) and (b) host-memcpy `io_copy`.
   Open: how big can a single runtime-weight MIL program get before a constraint (inner-dim ×32,
   slot ordering) or a silent-zero bites? Sets the floor on `io`.
3. **mHC fork trigger:** which long-context rung (seq ≥ 2–8K, per ADR 0001) is the test that
   decides whether mHC gets an ANE kernel at all? *(Unchanged — no external evidence bears.)*
4. **The CPU floor — answered, keep as a watch-item:** `dW` + optimizer have **no MIL operator
   and are CPU-resident in every observed system** (Orion, lil-bro, maderix) → keep on CPU,
   overlapped. Re-open only if PR #40's "bwd dX/dW verified on ANE (0.06 ms)" proves reproducible
   *and* survives the IOSurface round-trip at our shapes.
5. **Measurement methodology (new):** what is the externally-validated way to confirm ANE
   residency for a *non-CoreML, private-API* MIL workload? (powermetrics ANE-power vs per-op
   buckets — the Instruments Core ML template is blind to us.) The ~56% rests on our own
   instrumentation; close this before ranking migrations by expected residency gain.
6. **Bandwidth ceiling (new):** does spatial-packing a 32K (or compacted ~9.2K) classifier
   weight into the IOSurface input push the *next* bottleneck from compute to **bandwidth**
   (Apple: short-input / large-weight regimes are bandwidth-bound)? Measure before committing P2.
