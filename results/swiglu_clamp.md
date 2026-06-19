# SwiGLU clamping (issue #9)

**Status: R0 overfit gate GREEN on the ANE with the knob on; backward verified by
finite differences across both clamp boundaries.**

## What changed (V4 §4.2.3)

V4: "clamped the linear component of SwiGLU to [−10, 10], while capping the upper
bound of the gate component at 10." In our `W2(SiLU(W1·x) ⊙ (W3·x))`:

- **gate** = `SiLU(h1)` → capped at 10: `siluc = min(SiLU(h1), 10)`
- **linear** = `h3 = W3·x` → clamped: `h3c = clamp(h3, −10, 10)`
- `gate = siluc · h3c`

Forward (`mil_dynamic.h` `gen_ffn_fused_dynamic`, `#if SWIGLU_CLAMP`): MIL
`minimum`/`maximum` ops on the fused FFN path (compiles on the ANE). Backward
(`train.m` SiLU block, `#if SWIGLU_CLAMP`): the clamp VJP, zero gradient in the
clamped regions —
```
dh3 = dsilu · min(silu,10)     · [|h3| < 10]
dh1 = dsilu · clamp(h3,-10,10) · [silu < 10] · silu'(h1)
```
With the knob off, both paths are byte-identical to the original.

## Verification

**R0 overfit (M3 Max ANE), `swiglu_clamp=1`:** the MIL clamp ops compile (10
kernels) and the batch overfits to **0.0156** — bit-identical to the knob-off
baseline, because R0 activations stay well inside ±10 (the clamp is inert at that
scale, so R0 proves wiring + inert correctness, not the active clamp).

**Active-clamp backward — finite differences** (`test_swiglu_clamp.c`): the
analytic backward matches central-difference numerical gradients through the
clamped forward across inputs that straddle both boundaries (silu>10, |h3|>10,
both), and the analytic gradient is **exactly 0** in each clamped region. All 9
cases pass. This is the decisive check that R0 cannot give.

```bash
cc -O2 -o /tmp/t training/training_dynamic/test_swiglu_clamp.c -lm && /tmp/t
```

The held-out-val ablation (≥3-pt LR sweep on a rung whose activations exceed ±10)
is the follow-up measurement; R0-green + the FD check are the correctness gate.
