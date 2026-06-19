# Attention sink (issue #8)

**Status: R0 overfit gate GREEN on the ANE with the knob on; the softmax-with-sink
forward + backward (incl. the per-head sink gradient) verified by finite
differences, GQA-aware.**

## What changed (V4 attention sink)

A learnable per-head logit `sink[layer, head]` is added as one extra, value-less
entry in the attention softmax denominator. For query `q`, head `h`:

```
α_hq(j) = exp(s_hj) / ( Σ_{k≤q} exp(s_hk) + exp(sink_h) )      (causal)
attn_out_hq = Σ_{k≤q} α_hq(k) · V_hk
```

i.e. the sink behaves like an always-present key whose value is **0**, so it only
drains probability mass (`Σ_k α < 1` by `exp(sink)/Z'`). Initialised to `sink=0`
→ `exp(0)=1`, so the sink is **active from step 0** (one unit of mass, no warm-up).

### Why this is a CPU bypass, not an ANE-kernel edit

The forward softmax runs **on the ANE** (`mil_dynamic.h gen_sdpa_fwd_dynamic`),
and its backward is the fixed two-pass `sdpaBwd1`/`sdpaBwd2` kernel pair — neither
has a slot for an extra denominator term. The sink changes `Z → Z + exp(sink)`,
which rescales every probability and adds a `dL/dsink` path; that cannot be
expressed through the existing kernels. So, `#if ATTN_SINK`, the attention core is
recomputed on CPU from the kernel's own Q/K/V outputs:

- **Forward** (`cpu_ops.h attn_cpu_forward`, called in `train.m forward_hidden`
  under `#if ATTN_CPU`): overwrites `attn_out` with the sink softmax. Q/K/V (incl.
  partial RoPE) still come from the ANE kernel — only the softmax+`·V` is redone
  on CPU. (This same CPU core also carries QK-norm, issue #7; the sink is selected
  by passing its `sink_h` pointer, with the QK-norm gains NULL.)
- **Backward** (`cpu_ops.h attn_cpu_backward`, in `train.m` under `#if ATTN_CPU`):
  replaces the `sdpaBwd1/2` + GQA-reduce block. From `da = dL/d(attn_out)` it
  produces GQA-reduced `dQ[Q_DIM]`, `dK[KV_DIM]`, `dV[KV_DIM]` and accumulates the
  per-head `dL/dsink`. The standard `rope_backward_inplace` then runs as before.

`sink` is one fp32 scalar per (layer, head), optimised with AdamW (no weight
decay, like the norm/bias params), scaled/clipped/zeroed alongside the other
gradients, and persisted in the checkpoint (`#if ATTN_SINK`, appended). With the
knob off, every path is byte-identical to the original (verified: R0 off-run
below reproduces the baseline `0.0156` exactly).

The sink gradient derivation: with `p_j` the (sink-renormalised) probabilities and
`p_sink = exp(sink)/Z'`, the softmax-with-sink VJP gives
`g = Σ_j p_j·dp_j`, `dscore_j = p_j·(dp_j − g)·scale`, and
`dL/dsink = −p_sink·g` (the sink has `dp_sink = 0` since its value is 0).

## Verification

**R0 overfit (M3 Max ANE), byte-256 synthetic batch pinned at pos 0**
(`--overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10`):

| build | step 200 | step 390 | best |
|---|---|---|---|
| `attn_sink=0` (baseline) | 0.0538 | **0.0156** | — |
| `attn_sink=1` | 0.0432 | **0.0110** | **0.0105** |

Both start at the identical `loss=4.6449` (same init) and collapse monotonically.
The off-run reproduces the documented baseline floor (`0.0156`) bit-for-bit,
proving the knob-off path is inert. The on-run is **active** at R0 (unlike the
SwiGLU clamp, which is inert at that scale): the smooth collapse to `0.0105`
requires the CPU sink forward *and* its backward — including `dQ/dK/dV` and
`dsink` — to be self-consistent, or the loss would plateau/diverge. It converges
slightly *below* baseline because the learnable per-head sink is an extra DOF.

**Active-sink backward — finite differences** (`test_attn_cpu.c`, GQA 4/2,
hd=8, seq=5): analytic vs central-difference over `Q`, `K`, `V`, **and `sink`**:
max `|analytic − numerical| = 2.37e-04` (sink-only case) — the decisive check
that the per-head sink gradient (which R0's behavioral collapse alone cannot
isolate) is correct. The same test covers the QK-norm and combined cases.

```bash
cc -O2 -o /tmp/t training/training_dynamic/test_attn_cpu.c -lm && /tmp/t
cd training/training_dynamic && make MODEL=gen_r0_overfit EXTRA=-DATTN_SINK=1
./train --scratch --overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10
# baseline (knob off): make MODEL=gen_r0_overfit && ./train --scratch --overfit ... → 0.0156
```

The held-out-val ablation (does the sink reduce tokens-to-target on a real rung?)
is the follow-up measurement; R0-green + the FD check are the correctness gate.
