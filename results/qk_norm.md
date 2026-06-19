# Q/KV RMSNorm (issue #7)

**Status: R0 overfit gate GREEN on the ANE with the knob on; the RMSNorm-of-Q/K
forward + backward (incl. the learnable per-dim gains and the GQA key reduction)
verified by finite differences.**

## What changed (V4 stabilizer)

RMSNorm is applied to each attention head's query and key vector — over
`head_dim`, with a learnable per-dim gain — **just before the scores** (i.e. on
the post-RoPE Q/K). This bounds the attention logits (V4's stated reason for
dropping QK-Clip). For head `h`, position `t`:

```
q̂_h,t = RMSNorm(q_h,t) ⊙ g_q     k̂_h,t = RMSNorm(k_h,t) ⊙ g_k
score_h(t,j) = (q̂_h,t · k̂_h,j) / sqrt(head_dim)
```

`g_q`, `g_k` are `[head_dim]` gains, one pair **per layer** (shared across heads),
initialised to 1.0 — so the norm is **active from step 0**. RMSNorm uses the
model `NORM_EPS` (1e-5). The key norm is computed on the `kv_heads` keys before
GQA tiling (one norm per kv-head, then shared by the `gqa_ratio` query heads).

### Why it shares the attention-sink CPU bypass

QK-norm rewrites the Q/K that enter the score `q·k`, but the ANE forward softmax
kernel and its fixed `sdpaBwd1/2` backward have no notion of it — exactly the same
wall the attention sink hit (#8). So both knobs share **one** CPU attention core,
`cpu_ops.h attn_cpu_forward/backward`, gated by `ATTN_CPU = ATTN_SINK || QK_NORM`
(`config.h`). The kernel still produces Q/K/V (post-RoPE); the CPU core normalises
Q/K (`#if QK_NORM`), optionally adds the sink (`#if ATTN_SINK`), and does
softmax+`·V`. With both knobs off, `ATTN_CPU` is 0 and the original all-ANE path
is byte-identical.

Backward (`attn_cpu_backward`): from `da = dL/d(attn_out)` it forms the gradient
in *normalised* space (`dQ̂`, `dK̂`), then applies the RMSNorm VJP back to the
post-RoPE Q/K and accumulates the gain gradients:

```
dL/dx_i = g_i·dy_i/r − x_i·c/(HD·r³),   c = Σ_d g_d·dy_d·x_d,   r = rms(x)
dL/dg_d += dy_d·x_d/r
```

`dQ`/`dK` come out w.r.t. the post-RoPE Q/K, so `rope_backward_inplace` runs
unchanged afterwards. `g_q`, `g_k` are per-layer `[head_dim]` params trained with
AdamW (no weight decay, like the other norm gains), scaled/clipped/zeroed with the
rest of the grads and appended to the checkpoint (`#if QK_NORM`).

## Verification

**R0 overfit (M3 Max ANE), byte-256 synthetic batch pinned at pos 0**
(`--overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10`),
minimum loss over the run:

| build | min loss |
|---|---|
| baseline (`qk_norm=0`, all-ANE attention) | 0.0150 |
| **`qk_norm=1`** | **0.0202** |
| `attn_sink=1` (regression check, unified core) | 0.0105 |
| `qk_norm=1 attn_sink=1` (combined path) | 0.0154 |

QK-norm is **active** at R0 (it rescales Q/K from step 0), so the collapse
exercises the CPU forward *and* its backward — including the RMSNorm VJP and the
gain gradients — or the loss would not converge. The sink-only build still hits
0.0105 (identical to before the unification → the shared core didn't regress #8),
and the combined build converges too, confirming the two knobs compose.

**Backward — finite differences** (`test_attn_cpu.c`, GQA 4/2, hd=8, seq=5),
analytic vs central-difference over `Q`, `K`, `V`, `sink`, `g_q`, `g_k`:

| case | max \|analytic − numerical\| |
|---|---|
| qk_norm only | 5.75e-04 |
| sink only | 2.37e-04 |
| sink + qk_norm | 5.83e-04 |
| neither (identity) | 3.94e-04 |

The `g_q`/`g_k` gain gradients (which R0's behavioral collapse cannot isolate) are
covered by the qk_norm and combined cases.

```bash
cc -O2 -o /tmp/t training/training_dynamic/test_attn_cpu.c -lm && /tmp/t
cd training/training_dynamic && make MODEL=gen_r0_overfit EXTRA=-DQK_NORM=1
./train --scratch --overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10
```

The held-out-val ablation (does QK-norm reduce tokens-to-target / stabilise a
higher-LR run on a real rung?) is the follow-up measurement; R0-green + the FD
check are the correctness gate.
