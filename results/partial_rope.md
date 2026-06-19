# Partial RoPE — last `rope_rotary_dims` dims (issue #10)

**Status: R0 overfit gate GREEN on the ANE.** Identity at every current ladder
rung (head_dim ≤ 64); behaviorally exercised and green at head_dim = 128.

## What changed

Rotate only the last `ROPE_ROTARY_EFF = min(HD, rope_rotary_dims)` dims of each
head; leave the leading `HD − ROPE_ROTARY_EFF` dims unrotated (NoPE), per V4's
"last 64 dims" scheme (§2.3.3 anchor). Implemented entirely in the RoPE cos/sin
blobs — the unrotated dims get an identity rotation baked in (`cos=1, sin=0`), so
the forward MIL kernel is **unchanged**:

- `mil_dynamic.h` `get_rope_cos_blob` / `get_rope_sin_blob`: identity entries for
  pairs covering the leading `ROPE_NOROT` dims.
- `cpu_ops.h` `rope_backward_inplace`: skip those same pairs (identity rotation →
  identity gradient), so forward and backward stay consistent.

Interleaved-RoPE interpretation: dim `d` belongs to pair `d/2`; "last `eff` dims"
= the pairs covering dims `[HD−eff, HD)`, which keep their natural frequency
`1/θ^(2i/HD)`. The leading pairs pass through unrotated.

## Scale caveat (per the issue)

V4 uses last-64 because its head dim is large. Our rungs have `head_dim` 16
(`r0_overfit`), 32 (`r2_small`), 64 (`r3_110m`) — all ≤ 64 = default
`rope_rotary_dims` — so `eff = head_dim` and the knob is a **no-op (identity)**.
It only becomes a measurable ablation at `head_dim > 64` (e.g. the Qwen3-0.6B
config, `head_dim = 128`). This is a null result by construction at small head
dims, not "no improvement".

## Verification (M3 Max ANE)

| config | head_dim | eff / norot | step 0 | step 390 |
|---|---|---|---|---|
| `gen_r0_overfit` | 16 | 16 / 0 (identity) | 4.6449 | **0.0156** (bit-identical to baseline) |
| `head_dim=128` smoke | 128 | 64 / 64 (**partial**) | 4.6299 | **0.0000** |

The head_dim=128 run is the load-bearing check: it overfits to ~0 with 64 of 128
dims rotated, so the partial forward (blobs) and partial backward (CPU) are
mutually consistent — an inconsistent pair would corrupt the gradient and the
batch would not collapse. The measurable head_dim>64 ablation on held-out
validation is the follow-up measurement (Qwen3-0.6B rung).
