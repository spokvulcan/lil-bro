# mHC — manifold-constrained hyper-connections (issue #11, flagship)

**Status: R0 overfit gate GREEN with mHC enabled (N_HC ∈ {2,4} both collapse below
the baseline floor); the Sinkhorn residual map stays doubly-stochastic; the full
per-position map generation + stream coupling forward *and* backward is FD-verified.
N_HC=1 (default) compiles the whole mechanism out — byte-identical to the plain
trainer.**

## What changed (DeepSeek-V4 §2.2 hyper-connections)

The residual stream is expanded to `N_HC` parallel streams `X ∈ ℝ^{N_HC×d}` per
position. Each transformer sub-layer `F` (attention, FFN — two per layer) is wrapped
in three maps generated *per position* from `X̂ = RMSNorm(vec X)`:

```
u      = A·X                         # collapse N_HC streams → one d-vector (F's input)
X_out  = B·X  +  C ⊗ F(u)            # B mixes streams, C broadcasts F's output back

A = σ(α_pre ·(X̂ W^pre ) + S^pre )                  ∈ (0,1)^{N_HC}      input map
B = Sinkhorn(α_res·Mat(X̂ W^res) + S^res, t_max=20)  doubly-stochastic   residual map
C = 2σ(α_post·(X̂ W^post) + S^post)                 ∈ (0,2)^{N_HC}      output map
```

`A` is a sigmoid and `C` a doubled sigmoid (**not** a softmax) — per the issue. `B`
is projected onto the doubly-stochastic manifold by Sinkhorn-Knopp, reusing the #5
spike's verified log-domain code verbatim (τ = 0.5, t_max = 20, unrolled forward,
reverse-mode backward). `W^pre,W^post ∈ ℝ^{(N_HC·d)×N_HC}`, `W^res ∈ ℝ^{(N_HC·d)×N_HC²}`;
`S^*` static biases; `α^*` scalar gates. The **entry** broadcasts the embedded hidden
into all `N_HC` streams; the **exit** sums the streams back to one hidden before the
shared final RMSNorm + compact-vocab head.

### Init ≈ plain residual

Gates start small (`α=0.1`), biases zero: `A=σ(0)=0.5`, `C=2σ(0)=1`, `B=Sinkhorn(0)=1/N`
(uniform). With the entry broadcast making all streams equal, `u ∝ h0` (the per-position
RMSNorm right after is scale-invariant) and `B·X = h0`, so each sub-layer sees the same
input and residual as the plain transformer. Evidence: **R0 step 0 is identical to the
baseline (4.6449)** at N_HC=2 and N_HC=4 — the mechanism then learns to differentiate
streams.

### Placement: CPU-first (per ADR 0001)

The truncation-free but doubly-stochastic Sinkhorn and the per-position dynamic maps
fight the fixed-shape MIL kernels, so — as with the MTP path (#6) — mHC runs on **CPU**
(`mhc.h`) while the sub-layer cores (`F` = attention, FFN) stay on the **ANE**, unchanged.
The wrap feeds each ANE sub-layer the collapsed input `u = A·X` and consumes its bare
output: the fused FFN kernel is given a **zero residual base** so it returns
`res_alpha·ffn` (the attention path already exposes `res_alpha·o_out`), and the mHC
recombine supplies the residual coupling. The whole module is `#if N_HC > 1`; at
`N_HC = 1` (every current rung) it compiles away and the trainer is byte-identical —
verified: R0 = 0.0150, unchanged, and no "mHC enabled" banner.

Orchestration lives in `train.m`: `forward_hidden` carries the wide stream `X[N_HC,DIM,SEQ]`
through entry → (premap → sub-layer → recombine) × 2 per layer → exit; the backward
mirrors it, replacing the two plain residual passthroughs with `mhc_recombine_bwd` /
`mhc_premap_bwd` at the FFN→attn and attn→input seams, and broadcasting/summing the
stream gradient at the exit/entry. mHC params (`W^pre/W^res/W^post` + `S^*` + `α^*`)
use AdamW on both `--opt` paths (the main 2D-Muon rule doesn't extend to these tiny
dynamic generators; AdamW is lower-risk for the gate), scaled/clipped/normed/zeroed
alongside the main grads. Not yet checkpoint-persisted (resume re-inits, like MTP).

## Verification

**R0 overfit (M3 Max ANE), byte-256 synthetic batch pinned at pos 0**
(`--scratch --overfit --data r0_synthetic.bin --accum 1 --wd 0 --lr 2e-3 --warmup 10`):

| build | step 0 | best loss (800 steps) | B doubly-stochasticity (max\|Σ−1\|) |
|---|---|---|---|
| `N_HC=1` (off, identity) | 4.6449 | **0.0150** (= baseline) | — (mHC compiled out) |
| `N_HC=2` | 4.6449 | **0.0032** | 1.19e-07 |
| `N_HC=4` | 4.6449 | **0.0035** | 2.38e-07 |

Both mHC widths collapse the combined loss monotonically *below* the baseline overfit
floor (the extra capacity overfits one batch slightly more easily), with no NaNs — the
acceptance gate. It can only happen if the per-position map generation, the Sinkhorn
coupling, the `N_HC`-stream expand/collapse, and the gradient back through all of them
(maps, `A·X` collapse, `B·X + C·F` recombine, entry/exit) wire together correctly. The
runtime doubly-stochasticity probe confirms every `B` stays on the manifold (`~1e-7`,
fp32 CPU — tighter than the fp16 `t_max=20` bound #5 established).

**Backward — finite differences** (`test_mhc_module.c`, the `mhc.h` split API:
`premap → linear-F → recombine` and `recombine_bwd → F_bwd → premap_bwd`, multi-position
non-square dims to catch the `[N_HC,DIM,SEQ]` layout):

| precision | max \|analytic − numerical\| | doubly-stoch |
|---|---|---|
| double (`-DMHC_F=double`) | **1.77e-09** (worst: `S^res`, the Sinkhorn FD floor) | 2.2e-16 |
| float (trainer) | 4.20e-03 (fp32 Sinkhorn-gradient floor) | 1.2e-07 |

The double run is tight enough to prove the transcription is exact; gradients w.r.t.
`X` and **every** map parameter (`W^pre/W^res/W^post`, `S^pre/S^res/S^post`,
`α^pre/α^res/α^post`) and the sub-layer's own weight are covered.

```bash
cc -O2 -DMHC_F=double -o /tmp/t test_mhc_module.c -lm && /tmp/t   # double: ~1.8e-9
cc -O2               -o /tmp/t test_mhc_module.c -lm && /tmp/t   # float : 4.2e-3
cd training/training_dynamic && make MODEL=gen_r0_overfit EXTRA=-DN_HC=2
./train --scratch --overfit --data r0_synthetic.bin --steps 800 --accum 1 --wd 0 --lr 2e-3 --warmup 10
```

The held-out-val ablation (does mHC reduce tokens-to-target on a real rung, and at what
`N_HC`?) is the follow-up measurement; R0-green + doubly-stochastic B + the FD check are
the correctness gate.
