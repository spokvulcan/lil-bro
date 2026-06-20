# ADR 0004 — research dossier (evidence backing the V2 "ANE-resident trainer")

- **Date:** 2026-06-21
- **Purpose:** the cited, adversarially-verified evidence base for
  [ADR 0004](0004-ane-resident-trainer-v2.md). Separated from the ADR so the decision
  doc stays readable and the evidence stays reproducible (per CLAUDE.md: "a frontier
  result nobody can reproduce is not a result").
- **Method:** three tracks, this session — (1) a deep-research web workflow (5 angles,
  22 sources fetched, 109 claims extracted, 25 adversarially verified by 3-vote refute
  panels → 15 confirmed / 10 killed); (2) a GitHub deep-dive over `maderix/ANE` open
  PRs/issues + the fork network via `gh`; (3) direct verification of every load-bearing
  claim against lil-bro's own `training/training_dynamic/` source. Every assertion below
  is tagged with where it was checked. Single-source / unreproduced figures are marked.

---

## 1. Bottom line (what changed vs the first draft of the ADR)

1. **"~100% ANE residency" is not a reachable target — and saying so makes the ADR
   *more* honest, not less.** Two independent from-scratch ANE trainers — lil-bro and
   **Orion** (arXiv 2603.06728) — converge on the *same* split: forward + activation-
   gradient (`dx`) backward on the ANE; **weight-gradient `dW`, cross-entropy/loss,
   classifier-backward, embedding gather, and the optimizer on CPU.** No observed system
   has moved `dW` or the optimizer onto the ANE, and MIL has **no** loss / gradient /
   optimizer operator. So there is an **irreducible CPU floor**. The real goal is "drive
   the CPU residual and the CPU↔ANE round-trips down to that floor," not "100%."
2. **Mega-kernel fusion is *not* the residency lever for lil-bro** — on two independent
   grounds: the "deep 16–64-op fused graphs hit 94% util" claim was **refuted 0-3** in
   verification, and fusion à la upstream #24 *requires* `const()` weights → recompile
   per step, which lil-bro's compile-once design deliberately avoids (§5).
3. **The biggest *available* throughput lever is function-parameter IOSurfaces**
   (upstream PR #22, measured **76.9 vs 110 ms/step, −30%**): it deletes the
   `slice_by_size→reshape→transpose` weight-unpack tax lil-bro pays on every kernel
   (§5, §7).
4. **P1/P2 (classifier, RMSNorm) are not greenfield** — proven ANE kernels already exist
   for the *static* trainer (`ane_classifier.h`, `ane_rmsnorm_bwd.h`); the work is re-
   expressing them in the *dynamic* MIL generator. But their fp16 **execution**
   correctness at 32K width is **unverified** (documented ANE fp16 `layer_norm` overflow
   precedent), and the dynamic trainer's **vocab compaction** (32K→~9.2K active,
   `training/README.md:199`) already shrinks the CPU classifier — so the ANE port has to
   beat an already-optimized baseline, not the naive 32K one.
5. **The recompile problem that dominates the upstream literature does not apply to
   lil-bro.** lil-bro is compile-once / runtime-weight (no `exec()` restart, no 119-
   compile limit); the double-buffering (#23), ACCUM-for-compile (#33/#24), and Orion
   "delta-compilation" levers all solve a problem lil-bro already doesn't have (§5).

---

## 2. Prior & parallel art

### Orion — the decisive datapoint ⭐ (arXiv 2603.06728, Kumaresan, 6 Mar 2026; repo `mechramc/Orion`)
Trains **Stories110M from scratch on an M4 Max ANE**. §4.5 verbatim: *"Backward pass
(dx): ANE executes ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd per layer"*; forward *"ANE
executes fwdAttn (RMSNorm → QKV → SDPA → Wo) and fwdFFN."* Table 10: 1,000 steps,
22.4 min, **0/1,000 NaN**. **Table 5 (division of labor) keeps `dW` (cblas_sgemm + GCD),
NLL loss+grad, classifier backward, embedding lookup, and Adam on CPU** — the *same*
split as lil-bro, reached independently. Two more Orion results matter:
- **"Delta compilation"**: instead of recompiling on weight change, *unload → patch
  weight BLOBFILEs on disk → reload*, cutting recompile **4,200 ms → 494 ms/step (8.5×)**
  and removing the ~119-compile/process `exec()`-restart. (lil-bro solves the same
  problem a *different* way — spatial-packed runtime weights, §5 — so this is corroborating
  background, not a lever to adopt.)
- **Round-trips are the tax**: Orion's CPU baseline (283 tok/s) *beat* its ANE inference
  path (170 tok/s) "due to IOSurface overhead" — independent confirmation of the ADR's
  core premise that host↔ANE interaction, not ANE compute, is what's left to fight.
- *Caveat:* single-author, non-peer-reviewed, self-reported; no third-party reproduction.
  Credible because it independently agrees with lil-bro's architecture **and** lil-bro's
  own code reproduces its key mechanisms (verified §4). Treat absolute speedups as
  plausible-per-primary-source.

### Apple `ml-ane-transformers` + "Deploying Transformers on the ANE" (2022) — inference only, but the layout bible
- **Inference-only.** Two reads of the README found zero mention of training/backward/
  gradients; purpose verbatim: *"reference … if you are considering **deploying**."* Apple
  routes *training* to the GPU (WWDC24 "Train your ML models on Apple GPUs"; MLX/M5 uses
  GPU Neural Accelerators, not the ANE). ⇒ the classifier/CE/optimizer migration is
  genuinely novel territory; only maderix + Orion are precedent, and both keep those on CPU.
- **The hard layout rule** (verified 3-0, hardware-fundamental): ANE's conducive format is
  **4D channels-first `(B, C, 1, S)`**; the *last axis is not packed — must be contiguous
  and 64-byte aligned*; a **singleton last axis is padded to 64 B = 32× memory in fp16**
  (64× in int8). The contracted/small dim must **not** land last; sequence does. Orion
  confirms it as a *fixed* requirement for the reverse-engineered training path
  (`[1,768,1,1]` padded to `[1,768,1,16]`). lil-bro obeys it (`mil_dynamic.h` emits
  `[1,C,1,S]`). *Older chips use 32-byte, not 64-byte, alignment.*
- **`nn.Linear` → `nn.Conv2d` (1×1 conv)**: the ANE's native primitive is convolution,
  not matmul; Apple swaps all Linear→Conv2d. One technical source quantifies *1×1 conv =
  3× the throughput of the ANE native matmul path*, **but** `smpanaro/more-ane-transformers`
  found the swap "doesn't seem to help" at GPT-2 scale when bandwidth-bound. lil-bro
  currently emits **`matmul`** (`mil_dynamic.h:31, 184, 197`) → conv substitution is an
  **unexploited but unproven** datapath lever.

### maderix Substack "Inside the M4 ANE" (Parts 2–3) — the fp16-backward result
Verbatim Part 3: *"Loss would plateau at ~5.5 … Every ANE-computed gradient was
effectively zero … Only the embedding gradient (computed on CPU in fp32) showed learning
… Multiply the loss gradient by 256 right after cross-entropy … Divide it back out before
the Adam update."* Conclusion: *"FP16 is fine for forward, dangerous for backward."* This
is reproduced in lil-bro's shipping code (§4) and is the empirical basis for the fp32-
island discipline. Standard technique (NVIDIA mixed precision, arXiv 1710.03740).

### Refuted / not transferable
- **Draw Things "ANE doesn't do dynamic weights"** (engineering.drawthings.ai) — every
  load-bearing claim refuted **0-3** (abandoned-CoreML, 22 TFLOPs dynamic matmul, "private
  ANE rejected because weights baked in"). Their pessimism about runtime weights is
  *contradicted* by lil-bro + Orion, which both run dynamic weights in production.
- **`apple/ml-cross-entropy` (cut-cross-entropy)** — a real memory-saving CE technique, but
  GPU/Triton-oriented; relevant only as a design reference for a fused-CE island, not an
  ANE path.

### Fork network (`gh` survey of `maderix/ANE`, 947 forks)
- **`solatticus/ANE`** (+14 ahead) — the only fork with real *training-side* work: **LoRA
  on a zero-recompile dynamic-weight pipeline**, 256× loss-scaling, the `exec()`-restart
  workaround, iOS execution, and Ghidra RE of the private framework. Citable parallel art
  for dynamic-weight training.
- **`m0at/ANEtransformers`** (+10) — inference + an **ANE quantization probe** (does the
  ANE do native int8 or dequant-to-fp16?); verdict leans "dispatch overhead dominates at
  S=1." Inference/quant reference.
- All other surveyed forks (`0xSojalSec` 26★, `studioburnside/ane-neural-network-trainer`,
  `sgwd/ANE-M4-Training`, `vipuldivyanshu92`) are **pure mirrors** (0 commits ahead).

---

## 3. ANE hardware / software constraints (verified)

| Constraint | Status | Evidence | Bearing on lil-bro |
|---|---|---|---|
| Layout `(B,C,1,S)` fp16, last axis 64-B aligned, 32× penalty for singleton last axis | **PROVEN** (3-0) | Apple ANE-transformers; Orion | Already obeyed (`mil_dynamic.h`); constrains any new kernel's shapes |
| ANE native op is **conv**, not matmul; Linear→1×1 Conv2d is the datapath map | **PROVEN** (3-0); speedup scale-dependent | Apple; smpanaro | lil-bro emits matmul → conv swap unexploited, **not guaranteed faster** |
| MIL has **no** loss / cross-entropy / gradient / optimizer op | **PROVEN** (3-0) | coremltools iOS15 ops dir | CE + optimizer have **no** drop-in MIL primitive — must hand-build from fwd primitives or stay CPU |
| MIL has **no per-op accumulation/precision control** (only `T={fp16,fp32,int32}`) | **PROVEN** (3-0) | coremltools | fp32 "islands" must be built by **explicit `cast` ops in the graph**, not an attribute |
| MIL **does** expose `reduce_log_sum_exp` (stable LSE), `reduce_sum/mean/sum_square`, `rsqrt`, `l2_norm`, `layer_norm` | **PROVEN expressible** (3-0) | coremltools iOS15 `reduction.py`, `normalization.py` | CE + RMSNorm are *expressible* as a MIL graph — **expressibility ≠ correct fp16 execution** |
| ANE fp16 `layer_norm` / wide softmax can **overflow / be wrong on-device** | **LIVE RISK** | hollance unsupported-layers; Apple Dev forum "ANE-Optimized Layer Norm Fails on ANE" | P1/P2 fp16 correctness is the real unknown → R0 gate decides |
| Native fused **SDPA silently ignores the causal mask** | **PROVEN** (Orion 3-0 + maderix) | Orion Table 3 #6; maderix | lil-bro sidesteps via **mask-as-additive-bias** inside the kernel (`mil_dynamic.h:188-190`) → attention stays fully on ANE |
| fp16 **backward gradients underflow to ~0** without loss-scaling | **PROVEN** (3-0) | maderix Part 3; lil-bro code | lil-bro has `loss_scale` (§4); any op migrated into the bwd chain inherits this |
| Short-seq decode is **bandwidth-bound** (latency flat across seq 32/64/128) | **PROVEN** (2-1) | Apple ANE-transformers; Apple Llama-on-CoreML | *Not* lil-bro's regime (large-batch full-seq training is compute/ANE-bound, ~56% of step) — **but** packing a 32K classifier weight could push the *next* ceiling to bandwidth |
| "**512-channel** hard constraint" | **HARNESS-CONDITIONAL, not a silicon law** | maderix #42 vs #28 contradict on the *same* M3 Ultra; the 64-B last-axis rule does **not** explain it (channels are non-last) | Do **not** bake "512" in as a law; re-measure conv-channel configs per chip at R0 |
| Max tensor dim **65536** | **PROVEN** | maderix #26 chunks vocab 151936/16 | lil-bro vocab 32K (and 32K classifier channels) are **under** the limit — single-shot OK dimensionally |
| ~119 compiles/process; ANE compiler leaks → `exec()` restart | **PROVEN for the *baked-weight* path** | maderix #23/#42; solatticus | **Does not apply to lil-bro's dynamic path** (compile-once, §5) |
| Per-eval dispatch/XPC overhead ~70–160 µs | **PROVEN** | maderix #24/#40; m0at | Argues for **fewer, bigger** kernels — but via runtime-weight-safe fusion, not const() mega-kernels |
| Second ANE die on Ultra (UltraFusion) reachable? | **UNVERIFIED** | maderix #42 (single-die ~8.77 TFLOPS) | Don't assume 2× headroom on Ultra parts |

---

## 4. lil-bro's actual op placement *today* (verified against `training/training_dynamic/`)

| Op | Where | Evidence (this session) |
|---|---|---|
| QKV / Wo / FFN(W1,W3,SiLU,W2) **forward** | **ANE** | `gen_sdpa_fwd_dynamic`, `gen_ffn_fused_dynamic` `mil_dynamic.h:56,223` |
| RoPE fwd, GQA tiling, scores, **causal mask (as bias)**, softmax, ×V | **ANE** | `mil_dynamic.h:131-197` |
| Attention/FFN **dx backward** | **ANE** | `ffnBwd*`, `sdpaBwd1/2`, `qBwd/kvBwd` kernels `mil_dynamic.h:13-19` |
| RMSNorm fwd+bwd | **CPU** (vDSP) | `cpu_ops.h:7-54`; called `train.m:481…1621` |
| SiLU **backward** | **CPU** (vDSP) | `train.m:1379` (fwd is ANE-fused) |
| RoPE **backward** | **CPU** | `cpu_ops.h:480` (trunk); `:401` is MTP-only |
| Classifier logits GEMM + cross-entropy | **CPU** (`cblas_sgemm` + NLL) | `train.m:1296`; `cpu_ops.h:151` |
| **`dW`** weight-gradients (all layers) | **CPU**, async GCD-overlapped | `dw_cblas` serial queue `train.m:1227`; `cblas_sgemm` `:1322-1585`; awaited `:1644/1666` |
| Optimizer (Muon Newton–Schulz / AdamW) | **CPU** | `muon_update`/`adam_update` `train.m:878-880,1814-1823`; `cpu_ops.h:56-147` |
| Embedding lookup + backward | **CPU** | `embed_lookup`/`embed_backward` `train.m:641,1645`; `cpu_ops.h:259-273` |
| fp16 loss-scaling (256) + grad-clip (1.0) | **present** | `train.m:898-899,1318,1667` (README frames as `256×NLAYERS`) |
| Vocab compaction 32K→~9.2K active | **present** (CPU classifier shrink) | `training/README.md:199` |

**The CPU floor** = `dW` + optimizer + cross-entropy/loss + embedding gather. This is
exactly Orion's CPU set. None of it has a MIL operator; none of it has been moved to the
ANE in any observed system. lil-bro already **overlaps `dW` with the next step's ANE
forward** (GCD async) — the right mitigation given it can't be moved.

---

## 5. The recompile non-problem, and the non-levers

lil-bro's dynamic trainer **packs weights + activations into one runtime IOSurface input
and slices them apart inside the kernel** (`mil_dynamic.h:18-34, 65-94`), compiling each
kernel **once** (`compile_kern_mil_w`, `mil_dynamic.h:42,55`). Grep for
`recompile|exec(|119|compileModel|MLModel` across the dynamic trainer returns **empty**.
`training/README.md:38` confirms: *"compile 10 kernels once at startup, no recompilation."*
The *static* pipeline (`train_large*.m`) is the one that *"recompile[s] every 10 steps via
`exec()` restart"* (`training/README.md:24`).

⇒ **Non-levers for lil-bro** (they fix a problem lil-bro already solved):
- async double-buffered compile (upstream #23, "eliminates 88% compile bottleneck"),
- ACCUM-for-compile-amortization (#33/#24, `−28% wall` via fewer `exec()` restarts),
- Orion "delta compilation" (4,200→494 ms/step).

⇒ **Real levers** to keep activations resident *within* the compile-once regime:
- **Function-parameter IOSurfaces** (PR #22): one IOSurface per weight as a `func main`
  parameter, deleting the 3 unpack ops/weight (≈12 ops/attention kernel) → **−30%
  (76.9 vs 110 ms/step)**. lil-bro pays exactly this tax today.
- **Host-memcpy kernel chaining** — lil-bro **already has it** as `io_copy` (`io.h:63`),
  the same mechanism as PR #22's `ane_bridge_copy_io` (a CPU `memcpy` between mapped
  surfaces; removes the ANE→CPU-tensor→ANE conversion, not a true on-device DMA).
- **True on-device chaining (`_ANEChainingRequest`) is BLOCKED** for in-memory MIL
  (PR #40: `prepareChainingWithModel:` needs disk Espresso IR, crashes on
  `_ANEInMemoryModel`). ⇒ the fusion ceiling is set by host-memcpy + bigger single MIL
  programs, **not** native chaining.

**Runtime-weight op constraints** (PR #47, imperatormk — silent-correctness traps to
encode as asserts): matmul inner dim must be a multiple of **32**; IOSurface slot sizes
strictly ascending (in) / descending (out); grouped/depthwise conv with runtime weights
fails; some `reshape/transpose/reduce_*` fail with runtime weights. *Nuance:* lil-bro
**does** reshape/transpose its spatially-packed runtime weights and trains fine, so these
constraints are **mechanism-specific** (separate weight inputs vs spatial packing) and
**must be re-validated per kernel** before adopting PR #22's function-parameter form.

---

## 6. Feasibility-tagged migration verdicts (the core output)

- **PROVEN FEASIBLE** (do): runtime weight update without recompile (already shipped);
  full forward + `dx` backward on ANE (already shipped); 256× loss-scaling fp32 island
  for backward (already shipped); `(B,C,1,S)` layout (already obeyed); matmul-as-1×1-conv
  as the *correct datapath map* (unexploited, speedup unproven at scale).
- **PLAUSIBLE-BUT-UNVERIFIED** (gate at R0 before trusting): **classifier-forward + CE**
  and **RMSNorm fwd/bwd** as a MIL graph on the ANE. The reductions exist
  (`reduce_log_sum_exp`, `rsqrt`, `l2_norm`); proven static-trainer kernels exist
  (`ane_classifier.h`, `ane_rmsnorm_bwd.h`); **but** fp16 on-device correctness of a
  wide softmax/LSE and RMSNorm is unverified, with `layer_norm`-overflow precedent. And
  P1 must beat the **compacted ~9.2K** CPU classifier, not the naive 32K one — and the
  per-batch-dynamic active-token count fights fixed-shape MIL.
- **BLOCKED / CPU floor** (don't spend kernel risk): **`dW` weight-gradients** and the
  **optimizer** (Muon NS / AdamW). No MIL operator; CPU-resident in *every* observed
  system (Orion + lil-bro + maderix). The win here is **overlap** (already done), not
  migration. Embedding gather likewise stays CPU (gather is ANE-awkward; cost tiny).
- **REFUTED** (remove from the plan): "deep fused kernels → 94% util" (0-3); "32K softmax
  33.8× faster on ANE" (0-3); "ANE training only 5–9% of peak" (0-3). Mega-kernel fusion
  is **not** the residency lever.

---

## 7. Upstream PR / issue ledger (`maderix/ANE`) — levers vs noise

| # | State | What | Number | Use for lil-bro |
|---|---|---|---|---|
| #22 | closed (no reason) | function-param IOSurfaces vs spatial packing | **76.9 vs 110 ms/step, −30%** | **TOP lever** — port + re-validate on our config |
| #19 | **merged** | bridge + offload classifier/softmax/rmsnorm-bwd to ANE (static) | — | proven precedent for P1/P2 kernels |
| #40 | open | private-API research: chaining BLOCKED, E5 custom-MIL works, **bwd dX/dW verified on ANE 0.06 ms** | — | decision-shaping: chaining dead, bwd-matmul-on-ANE possible |
| #47 | open | runtime-weight training loop + the silent-zero constraint list | ~3 it/s M1 | **constraints to encode as asserts** |
| #39 | open | cache-optimized embedding gather | **~12×, bit-exact** | cheap CPU-floor reduction if embedding is hot |
| #24 | open | mega-kernel fusion (needs `const()` weights) | 3.0× fwd @768 | **not applicable** (forces recompile) |
| #23 | open | async double-buffer compile | 55→0 ms stall | **non-lever** (no recompile here) |
| #33 | open | ACCUM/MAX_COMPILE env + IOSurface pool | −28% wall @ACCUM=20 | **non-lever** (compile amortization) |
| #18 | closed | weights-as-tensors (origin of dynamic weights) | ~100→3.4 ms weight sync | already lil-bro's design |
| #21 | closed | CPU-side tuning | 107→92 ms/step | closed *because* "real gains come from moving work onto the ANE" |
| #26 | closed | inference w/o CoreML — **ANE conv compiled+ran but numerically wrong**, fell back to CPU BLAS | — | **proof the R0 gate is necessary** |
| #46 | open | "20× error in benchmarks" — power never measured; util% use wrong M4 denominator | — | **discount all upstream util%/efficiency multipliers** |

---

## 8. The measurement gap (open)

Research angle 5 (how to *measure* ANE residency) produced **zero surviving claims**. The
~56% figure is lil-bro's **own** per-step `timing:` instrumentation, externally
unvalidated. Compounding it: lil-bro drives the ANE through **direct private APIs
(`_ANEIOSurfaceObject`/ANEServices), not CoreML** (`io.h:125,152`), so the Xcode
Instruments **Core ML / Neural-Engine template will not see it**, and `asitop`/`mactop`
read aggregate `powermetrics` ANE *power* (not per-op residency). Tools surfaced but
unproven for this use: `powermetrics --samplers ane_power`, `hollance/neural-engine`
`is-model-using-ane.md`, `CoreMLProfiler`. **Action:** treat per-step buckets as the
operative metric (they're causal and reproducible) but cross-check against
`powermetrics` ANE-power so "ANE-bound" rests on two independent signals.

---

## 9. Sources

**Primary research / papers**
- Orion: ANE training — arXiv 2603.06728 (html/v1); repo `github.com/mechramc/Orion`
- Apple, "Deploying Transformers on the Apple Neural Engine" — machinelearning.apple.com/research/neural-engine-transformers
- Apple `ml-ane-transformers`; Apple "On-Device Llama 3.1 with Core ML"
- coremltools MIL op defs (iOS15 `reduction.py`, `normalization.py`, `elementwise_unary`) — apple.github.io/coremltools
- NVIDIA Mixed-Precision Training — arXiv 1710.03740; `apple/ml-cross-entropy`
- Muon — kellerjordan.github.io/posts/muon

**Reverse-engineering / community**
- maderix Substack "Inside the M4 ANE" Parts 2–3; `maderix/ANE` (PRs #18–#51, issues #3/#24/#42/#46/#47/#49)
- `solatticus/ANE` (LoRA on dynamic weights); `m0at/ANEtransformers` (quant probe)
- `hollance/neural-engine` (incl. `is-model-using-ane.md`); `smpanaro/more-ane-transformers`
- Draw Things "Making the ANE Work" (claims **refuted** in verification); `tlkh/asitop`; `fguzman82/CoreMLProfiler`; eclecticlight "hunt for the M1's Neural Engine"

**Internal (verified this session)** — `training/training_dynamic/`: `mil_dynamic.h`,
`train.m`, `cpu_ops.h`, `config.h`, `io.h`, `mhc.h`; `training/ane_classifier.h`,
`training/ane_rmsnorm_bwd.h`; `README.md`, `training/README.md`.

**Verification caveats:** Orion + maderix writeups are single-author, non-peer-reviewed,
self-reported — credible by mutual-independent agreement + lil-bro code reproduction, but
absolute speedups are plausible-not-established. Apple's layout article is 2022/inference-
framed; its layout/bandwidth claims are hardware-fundamental and re-confirmed by 2025–26
work. The 596M dynamic-weights datapoint is upstream-single-source (lil-bro's verified
ceiling is 110M on one ANE).
