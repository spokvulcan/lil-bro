# ADR 0001 — DeepSeek-V4 architecture, adapted for a small dense model on the ANE

- **Status:** Accepted — ratified via `/grill-with-docs` on 2026-06-19 (decisions table below)
- **Date:** 2026-06-19
- **Context docs:** [PRD](../PRD.md) · [ROADMAP](../../ROADMAP.md) (Phase 3 names mHC + CSA/HCA) · [training/README](../../training/README.md)
- **Primary spec:** `~/Downloads/DeepSeek_V4.pdf` ("DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"), read in full this session.
- **Research input:** deep-research run `wf_d4fafd82-a72` (22 sources, 25 claims verified, 20 confirmed / 5 killed). Cited below as **[R]**.

> **Amended 2026-06-20 — ANE-only.** No MLX/torch twin, no fp64 gradient oracle, no
> GPU baseline. Correctness is verified **behaviorally** (R0 overfit on the ANE → loss
> ~0), with each mechanism's effect read on **held-out validation** loss. Every
> reference below to the "MLX twin / fp64 oracle / R1 grad-diff gate / prototype" is
> superseded: the gate is R0 overfit, and the mHC "prototype" is now an **ANE-side
> spike** on the CPU path. (`lilbro/mlx_ref` + the grad-diff tests remain on disk,
> retained but no longer the gate.) Accepted trade-off: with no gradient oracle, a
> subtly wrong backward that still trains will not be caught.

## Provenance convention

Because one research finding collided with my first PDF read, every load-bearing
claim is tagged: **[PDF §x]** = read directly from `DeepSeek_V4.pdf` this session ·
**[R]** = deep-research finding (source URL inline) · **[INFERENCE]** = reasoned
extrapolation, not yet measured. Keep these separate — an inference is not a fact
until a number backs it.

---

## Context

`lil-bro` trains a small **dense** TinyStories transformer **from scratch on the
Apple Neural Engine** to measure whether DeepSeek-V4 ideas improve the
small-dense efficiency frontier (PRD). The current trainer
(`training_dynamic/`) is a plain pre-norm transformer: RMSNorm → MHA/GQA →
SwiGLU, AdamW or Muon (`--opt`), with attention decomposed as **Q@Kᵀ on ANE →
mask+softmax on CPU → scores@V on ANE** (`training/README.md`). MTP is not yet
implemented on the ANE (Tier B lands it).

We want to extend the architecture "as close as possible to real DeepSeek-V4,
but optimized for the dense little model." DeepSeek-V4 is **V3 + three
architectural upgrades + a large infra/precision/post-training half** [PDF §2].
The infra/precision half (TileLang, MegaMoE/EP, FP4/FP8, contextual parallelism,
on-disk KV, OPD/GRM/reasoning modes) is irrelevant to a dense model on **fp16**
ANE silicon. This ADR decides which of the **architectural** ideas to port, and
specifies the first one (mHC) in build-ready detail.

### Hard constraints

- **fp16 compute** on the ANE; no FP8/FP4. Precision-sensitive ops need fp16-safe
  formulations.
- **Fixed-shape MIL graphs** — new dimensions recompile (cached); new ops need new
  MIL kernels. Dynamic shapes / gather-scatter are the *hard* ANE ops.
- **CPU is already in the attention loop** (mask+softmax) — cheap CPU-side ops
  (softmax, Sinkhorn on tiny matrices) can land there first, then move to ANE.
- **Gate discipline (ANE-only, behavioral):** a component's ablation number is
  trusted **only after** it passes the **R0 overfit** gate on the ANE (full loop →
  loss ~0), with its effect read on **held-out validation** loss. Headline metric =
  tokens-to-target. No gradient oracle — a subtly wrong backward that still trains is
  not caught (accepted ANE-only trade-off).

---

## Decision — keep / drop / adapt for a dense model on the ANE

| Tier | Component | Verdict | Rationale |
|---|---|---|---|
| **A — stabilizers first** | Q/KV RMSNorm · attention sink · SwiGLU clamping · partial RoPE (last 64 dims) · finish Muon→V4 Newton-Schulz coefficients | **Adapt now** | Move toward V4 with no new heavy kernels; sink+softmax already live on CPU. Each is a clean one-variable R2 ablation. [PDF §2.3.3, §2.4, §4.2.3] |
| **B — flagship** | **mHC** (this ADR) · land the **ANE MTP** path (depth 1) | **Adapt, mHC first** | mHC is *the* dense-relevant V4 architecture idea (already Phase 3). MTP is cheap and helps small models. [PDF §2.2, §2.1] |
| **C — long-context only** | CSA (lightning indexer + top-k) · HCA | **Defer** | Per ROADMAP, "needs long context (seq ≥ 2–8K) to mean anything"; we train at seq=256. Top-k/gather is the most ANE-hostile op (dynamic shapes) **[R]**. Zero payoff at current rungs. [PDF §2.3] |
| **D — out of scope** | MoE + DeepSeekMoE machinery (Sqrt-Softplus affinity, hash routing, aux-loss-free balancing, Anticipatory Routing) · FP4/8 QAT · post-training (SFT/GRPO/OPD/GRM) | **Drop** | It is a *dense* model on fp16 silicon. MoE is a separate research axis, not "closest to V4 dense." [PDF §2.1, §4.2.3, §5] |

**Reference anchors to stay close to** [PDF §4.2.1, p24–25, DeepSeek-V4-Flash]:
`n_hc=4`, Sinkhorn `t_max=20`, partial RoPE last 64 dims, grouped output `g=8`,
sliding window `n_win=128`. (Attention-side anchors apply when Tier C is undeferred.)

**mHC is the first Phase-3 issue.** Tier A items precede it as quick R2 ablations
that de-risk training; Tier C waits behind a future long-context rung.

---

## Ratified decisions (grilling — 2026-06-19)

Taken in a `/grill-with-docs` session, leading with a recommended answer per the
user's request ("what would *you* choose for V4 parity"). All seven were accepted.

| Decision | Choice | One-line rationale |
|---|---|---|
| **Parity** | Architectural-mechanism parity | Only definition both measurable and honest at 20M. See [CONTEXT.md](../../CONTEXT.md). |
| **A / C maps** | **Sigmoid** (`A=σ`, `C=2σ`), per V4 PDF | Faithful to V4; correctness is checked behaviorally (overfit), so the softmax reference code isn't needed as an oracle. |
| **Sinkhorn backward** | **Unrolled** through `t_max` | At 4×4 the cost is trivial, so implicit-diff's O(1)-memory win is moot; unrolled is *more* exact at small `t_max` and is what V4 did. Implicit kept as fallback. |
| **`t_max`** | **20** + doubly-stochasticity convergence check; **log-domain** Sinkhorn | Parity value; convergence risk favors more iters not fewer; cost negligible. The ANE-side spike verifies `B` row/col sums ≈1 in fp16. |
| **`n_hc`** | **4**, exposed as a `Config` knob | Parity value; 4× of a tiny residual is still tiny; trivial to ablate `{2,4}` later. |
| **Placement** | **CPU-first** for the whole mHC block; ANE by profile | Mirrors the existing mask+softmax decomposition; the iterative 4×4 Sinkhorn fights fixed-shape MIL; lowest new-kernel risk to reach an overfit-green gate. |
| **Sequencing** | **Tier A on ANE ∥ mHC Sinkhorn spike on the ANE CPU path** | The spike is isolated (non-blocking) and answers the riskiest fp16 unknown early while Tier A delivers quick ANE wins. |

The build-ready spec below reflects these choices; its earlier "recommendation"
phrasing now reads as ratified.

---

## mHC specification (build-ready)

### Forward — DeepSeek-V4 formulation (authoritative for this project) [PDF §2.2, eqs 1–8]

Keep the residual stream **expanded** to `n_hc` parallel streams: `X_l ∈ ℝ^{n_hc×d}`
(vs. the usual `x ∈ ℝ^d`). Each sub-layer `F_l` (attention block or FFN block) is
wrapped by three learned maps instead of a plain residual add:

```
  X_{l+1} = B_l · X_l  +  C_l · F_l(A_l · X_l)            (eq 1)

  A_l ∈ ℝ^{1×n_hc}     input map   (collapse streams → one d-vector layer input)
  B_l ∈ ℝ^{n_hc×n_hc}  residual map (mix streams; doubly-stochastic)
  C_l ∈ ℝ^{n_hc×1}     output map  (broadcast layer output back to streams)
```

Dimensions: `A_l X_l ∈ ℝ^{1×d}` is the actual layer input; `F_l(·) ∈ ℝ^d`;
`C_l F_l(·) ∈ ℝ^{n_hc×d}`; `B_l X_l ∈ ℝ^{n_hc×d}`. Entry to the network broadcasts
the token embedding into `n_hc` streams; exit collapses them before the final
norm + head.

The three maps are generated **dynamically (input-dependent) + statically** from a
normalized flattening of the stream state [PDF §2.2, eqs 3–7]:

```
  X̂_l = RMSNorm(vec(X_l))                  ∈ ℝ^{1×(n_hc·d)}
  Ã_l = α^pre_l ·(X̂_l W^pre_l)  + S^pre_l   ;   A_l = σ(Ã_l)
  B̃_l = α^res_l ·Mat(X̂_l W^res_l) + S^res_l ;   B_l = SinkhornKnopp(B̃_l, t_max)
  C̃_l = α^post_l·(X̂_l W^post_l)ᵀ + S^post_l ;   C_l = 2·σ(C̃_l)
```

`W^pre,W^post ∈ ℝ^{(n_hc·d)×n_hc}`, `W^res ∈ ℝ^{(n_hc·d)×n_hc²}`; `S^*` are static
biases; `α^*` are scalar gates initialized small (so the layer starts ≈ a plain
residual). `σ` = sigmoid (bounds A,C non-negative & bounded → avoids signal
cancellation).

### Why it helps — the load-bearing property [R, 3-0]

`B_l` is projected onto the **Birkhoff polytope** (doubly-stochastic matrices) by
Sinkhorn-Knopp. A doubly-stochastic matrix has **spectral norm ≤ 1** (non-expansive,
`‖B_l X‖₂ ≤ ‖X‖₂`) and is **closed under multiplication**, so the composite residual
map across layers stays bounded — the stated mechanism for mitigating gradient
explosion. Empirically mHC cuts HC's peak gain magnitude (~3000) by three orders of
magnitude (~1.6) [R: arXiv 2512.24880 §5.4]. **Caveat [R, 3-0]:** this guarantee
holds *only to the extent Sinkhorn actually converged* — at small fixed `t_max` the
projection may not be doubly-stochastic and the norm bound is uncontrolled
(arXiv 2606.07574). ⇒ treat `t_max` as a hyperparameter to validate, not assume.

### Backward — two ANE-friendly options [R]

Both reduce to **matmul + elementwise + reduction**, with **no gather/scatter and
static shapes** for the `n_hc×n_hc` projection — i.e. ANE-expressible [R, INFERENCE
on the ANE-specific part].

1. **Unrolled (paper-faithful).** Differentiate through all `t_max` Sinkhorn
   iterations. Cost (compute + activation memory) grows **linearly in `t_max`**.
   Theoretically sound: derivatives of unrolled Sinkhorn converge **geometrically**
   to the true entropic-OT derivative (Pauwels & Vaiter, SIAM J. Opt. 2023). This is
   what DeepSeek did [R: arXiv 2512.24880 §4.3.1 "traverses the entire iteration"].
2. **Implicit-function-theorem (alternative).** Differentiate the Sinkhorn fixed
   point analytically — backward memory **O(t_max)→O(1)**, often more stable
   (Eisenberger et al. CVPR 2022; default in OTT-JAX). **Exact only near the fixed
   point**, so its advantage weakens at small `t_max`.

For `n_hc=4, t_max≈10–20` the unrolled cost is tiny (4×4 matrices), so **start
unrolled** (paper-faithful, simplest) and keep implicit-diff as a fallback if fp16
gradients misbehave.

### Low-precision Sinkhorn — the fp16 lever [R, 3-0]

Run Sinkhorn in the **log domain (log-sum-exp)**. This is the field-standard
numerical-stability fix (POT `sinkhorn_log`, GeomLoss, Peyré-Cuturi); the community
mHC repo uses it verbatim (`logsumexp + elementwise add + exp`). Its only downside —
numpy logsumexp being non-parallel — **does not apply to the ANE's parallel
reductions** [R, INFERENCE]. Optional second lever: **epsilon-scaling** (anneal the
entropic temperature) makes iteration count logarithmic, letting a smaller `t_max`
reach near-doubly-stochastic.

> ⚠️ Every fp16/ANE relevance note here is **[INFERENCE]** — none of the OT/Sinkhorn
> sources measured fp16 or the ANE. The strongest *evidence-backed* fp16 lever is
> log-domain Sinkhorn; whether its **gradient** is stable in fp16 at our `t_max` is
> **unmeasured** → see Open Questions / the mandatory prototype.

### Hyperparameters

| Param | V4 paper [PDF §4.2.1] | Community repo [R] | For our ~20M dense model |
|---|---|---|---|
| `n_hc` (streams) | 4 | 4 | **4** (start) |
| `t_max` (Sinkhorn) | 20 | 10 | **validate {10,20}** — sources disagree |
| `τ` (Sinkhorn temp) | — | 0.05 | 0.05 (start) |
| `α` gate init | small | 0.01 | small, so layer ≈ identity at init |

**Gap [R]:** the paper only validates `n_hc=4 / t_max=20` down to **3B**; nothing is
validated at sub-3B or in fp16. `t_max` and `τ` are open at our scale.

### Placement in the trainer

mHC replaces the **two residual adds** per block (`x += attn(norm(x))`,
`x += ffn(norm(x))`) with the A/B/C mixing over the `n_hc`-wide stream. New work:
(1) expand to `n_hc` streams at the embedding, reduce at the final norm; (2) per
sub-layer: `RMSNorm(vec(X))` + three low-rank matmuls + sigmoid + **log-Sinkhorn on
a 4×4 matrix**; (3) the A/B/C applications are tiny (`n_hc=4`) matmuls. The 4×4
log-Sinkhorn is a natural **CPU-first** landing (alongside the existing mask+softmax)
before any ANE MIL kernel — mirroring the trainer's current decomposition.

### Reference implementation [R]

`github.com/tokenbender/mHC-manifold-constrained-hyper-connections` — community
PyTorch **prototype** (clarity-first, *not* official DeepSeek code). Useful for the
`sinkhorn_log(logits, num_iters, tau)` projection and overall wiring. **Divergences
from our V4 target:** it bounds `H_pre/H_post` with **softmax over the stream dim**
(V4 PDF uses sigmoid), defaults `t_max=10` (V4 uses 20), and is a static (no custom
`autograd.Function`) unrolled-autograd path. Small-model config to mirror:
`n_layer=6, n_embd=288, n_head=6, hc_streams=4, sinkhorn_iters=10, tau=0.05,
alpha=0.01`.

---

## The V4-PDF ⇄ standalone-mHC-paper discrepancy (recorded, not resolved)

The deep-research run **refuted (0-3)** the sigmoid + RMSNorm-low-rank description
*as a description of the standalone mHC paper (arXiv 2512.24880)* — that paper and
its reference repo use **softmax** pre/post maps and `t_max=10`. But the
**DeepSeek-V4 PDF** (our actual build target, read directly this session) **does**
specify `A=σ`, `C=2σ`, RMSNorm-low-rank dynamic generation, and `t_max=20` in
§2.2/§4.2.1. Both are "mHC"; they describe the same Sinkhorn-projected
doubly-stochastic core with different input/output-map bounding.

**Resolution for this project:** the **DeepSeek-V4 PDF is authoritative** (the goal
is "closest to V4"). The standalone paper/repo is reference code only. The
sigmoid-vs-softmax pre/post choice and `t_max ∈ {10,20}` are **decision points to
settle empirically**, not contradictions to litigate.

---

## Consequences

**Positive:** a principled, gradient-stable residual upgrade that is dense-compatible
and (per the math) ANE-expressible without the hard ops; a clean first Phase-3
ablation; reuses the existing CPU-in-the-loop decomposition for the 4×4 Sinkhorn.

**Negative / cost:** `n_hc×` wider residual stream → `n_hc×` activation memory on the
residual path; new forward + backward + (eventually) MIL kernels; per-sub-layer
dynamic-param generation adds compute.

**Risks (→ Open Questions):** fp16 Sinkhorn *gradient* stability is **unmeasured**;
`t_max` convergence at small scale is **unvalidated**; the dynamic-generation detail
should be re-read against the PDF before coding.

### Decision: the Sinkhorn spike is now **mandatory**

Because the fp16/ANE stability of Sinkhorn's backward is inference, not evidence
[R, INFERENCE], and the unrolled-vs-implicit and `t_max` choices are unsettled:
**spike log-domain Sinkhorn forward+backward in isolation on the ANE CPU path
first** — verify fp16 doubly-stochasticity + a behavioral overfit of an isolated mHC
block — *before* wiring it into the full mHC residual. This was "optional" in the
pre-research plan; the evidence promotes it to a gate.

---

## Open questions (post-grilling — for the spike, not the design)

The design is settled (decisions table above). What remains is **measured**, not decided:

1. **Does `B` converge to doubly-stochastic in fp16 at `t_max=20`?** The non-expansiveness
   guarantee depends on it. The spike checks row/col sums ≈1; if not, raise `t_max`
   or add epsilon-scaling. (Backward stays unrolled; escalate to implicit only if fp16
   gradients misbehave *here*.)
2. **Re-read PDF §2.2 eqs 3–7 before coding the map generation** — the RMSNorm+low-rank
   dynamic/static split is in the V4 PDF only (web sources could not confirm it). Verify
   against the PDF, not memory.
3. **MIL op-support check before moving any mHC piece off CPU** — confirm the chosen hot
   op (likely the map-generation matmuls) compiles as static-shape matmul/elementwise/
   reduction with no gather-scatter. Until then, everything stays on CPU.

*Resolved in grilling:* parity (architectural-mechanism), A/C bounding (sigmoid),
backward (unrolled), `t_max` (20), `n_hc` (4), placement (CPU-first), sequencing (parallel).

---

## Next steps (workflow)

1. ✅ `/grill-with-docs` — **done 2026-06-19**; decisions ratified above; `CONTEXT.md` created.
2. ✅ `/to-prd` → `/to-issues` — **done 2026-06-20**; PRD spokvulcan/lil-bro#2 + 9 ANE-only issues (#3–#11).
3. **ANE-side Sinkhorn spike (mandatory, next)** — log-Sinkhorn fwd/bwd (unrolled, `t_max=20`,
   sigmoid `A`/`C`) on the ANE CPU path; verify fp16 doubly-stochasticity + a behavioral
   overfit; then the MIL op-support check (issue #5).
4. **In parallel** — land Tier A stabilizers (Q/KV RMSNorm, attention sink, SwiGLU
   clamping, partial RoPE, Muon→V4 NS coefficients) as clean R2 ablations on the ANE.
5. `/implement` per issue (fresh context) → **R0 overfit green** → R2 ablation
   (iso-loss, ≥3-point LR sweep, vary exactly one component).

## References

- **DeepSeek-V4 PDF** — §2.1 (inherited V3: MoE, MTP), §2.2 (mHC, eqs 1–8),
  §2.3 (CSA/HCA + §2.3.3 attention details), §2.4 (Muon), §4.2.1 (V4-Flash config),
  §4.2.3 (SwiGLU clamping, Anticipatory Routing).
- **[R] mHC** — arXiv 2512.24880 (Xie et al.); repo tokenbender/mHC-…; follow-ups
  arXiv 2606.07574 (Birkhoff acceleration), arXiv 2601.05732 (mHC-lite).
- **[R] Sinkhorn backward** — Pauwels & Vaiter, SIAM J. Opt. 2023 (arXiv 2207.12717);
  Eisenberger et al. CVPR 2022 (arXiv 2205.06688); OTT-JAX discussion #125.
- **[R] Low-precision Sinkhorn** — POT `sinkhorn_log` (pythonot.github.io); GeomLoss
  epsilon-scaling; Peyré & Cuturi (arXiv 1803.00567).
- Full verified report: deep-research run `wf_d4fafd82-a72`.
