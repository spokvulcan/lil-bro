# ANE MTP path (issue #6)

**Status: R0 overfit gate GREEN with the combined (main + MTP) loss; the full MTP
forward + backward — extra transformer block, glue, and cross-depth gradient
chaining into the shared trunk/embed — verified end-to-end by finite differences.**

## What changed (V4 Multi-Token Prediction)

For `mtp_depth = k`, after the trunk produces the hidden stream `h`, `k` extra
heads each predict one more future token (DeepSeek-V3/V4 MTP, `MTP_LAMBDA = 0.3`).
Per depth `kk = 1..k`, with `Sk = SEQ − kk`:

```
hp   = h_prev[:, :Sk]                          # trunk (kk=1) or previous MTP hidden
e    = embed[ targets[kk−1 : kk−1+Sk] ]        # embedding of the shifted target
hkin = proj · concat( RMSNorm_h(hp), RMSNorm_e(e) )   # proj: [D, 2D]
hk   = block_kk(hkin)                           # a full transformer block
loss_kk = CE( head(hk), targets[kk : kk+Sk] )   # shared compact-vocab head
```

Combined loss = `main_CE + MTP_LAMBDA · mean_kk(loss_kk)`. Each block is identical
to a main layer (pre-norm attention + SwiGLU FFN, residual scale `res_alpha`); the
head reuses the shared `rms_final` + compact-vocab classifier, and the input
embedding `e` reuses the shared `embed` table. Gradients flow back into both
shared tables and into the trunk (depth-1 `hp = trunk[:S1]`), with depth `kk`'s
hidden feeding depth `kk+1`.

### Placement: CPU-first (per ADR 0001)

The MTP block operates on `Sk < SEQ` positions, which fights the fixed-shape MIL
kernels — exactly the case ADR 0001 resolves with **CPU-first** ("lowest
new-kernel risk to reach an overfit-green gate"). So the MTP blocks compute on CPU
within the ANE trainer (`cpu_ops.h mtp_block_fwd/bwd`: cblas matmuls, the
FD-verified `attn_cpu` core for attention, CPU RMSNorm/RoPE/SwiGLU), while the
**trunk trains on the ANE** as before. ANE-by-profile placement of the MTP blocks
(and checkpoint persistence of the MTP params) are the follow-ups. The whole path
is `#if MTP_DEPTH > 0`; with `mtp_depth = 0` (every current rung) the trainer is
byte-identical to before — verified: base R0 = 0.0150, unchanged.

Orchestration lives in `train.m`: `mtp_forward` (after the main loss, folds the
MTP term into the reported loss) and `mtp_backward` (run before the main backward's
async dW so the shared-grad writes don't race; injects its trunk gradient into
`dy` after the main final-RMSNorm backward). MTP params (`proj` + block 2D weights
→ Muon when `--opt muon`, else AdamW; the four RMSNorm gains → AdamW, no wd) are
scaled/clipped/normed/zeroed alongside the main grads.

## Verification

**R0 overfit (M3 Max ANE), byte-256 synthetic batch pinned at pos 0**
(`--overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10`):

| build | step 0 (combined) | step 100 | min loss |
|---|---|---|---|
| baseline (`mtp_depth=0`) | 4.6449 | 0.1877 | 0.0150 |
| `mtp_depth=1` | 6.0261 | 0.1173 | **0.0137** |
| `mtp_depth=2` | 6.0297 | 0.1318 | **0.0105** |

The step-0 jump (`4.6449 → 6.026 ≈ 4.6449 + MTP_LAMBDA·log(CV)`) shows the MTP term
is added at the right magnitude and the targets/shapes are right (an off-by-one in
the shift or a wrong loss weight would not land on `0.3·log(101)`). The **combined**
loss then collapses toward 0 — both the main and the auxiliary heads overfit the
single batch — which is the acceptance gate: it can only happen if the MTP
forward, the per-depth heads, the cross-depth chaining, and the trunk/embed
gradient all wire together correctly. No NaNs.

**Backward — finite differences** (`test_mtp.c`, a self-contained tiny config
D=8/HD=4/GQA 2:1/MTP_DEPTH=2): analytic vs central-difference for the gradient of
the combined MTP loss w.r.t. **every** MTP parameter (`rms_h`, `rms_e`, `proj`,
`Wq/Wk/Wv/Wo`, `W1/W2/W3`, `rms_att`, `rms_ffn` across both depths) **and** the
shared `trunk`, `embed`, `rms_final`: max `|analytic − numerical| = 8.23e-05`. This
covers the pieces R0's behavioral collapse alone cannot isolate (the per-tensor
gradients and the depth-2→depth-1→trunk chain).

```bash
cc -O2 -o /tmp/t training/training_dynamic/test_mtp.c -framework Accelerate -DACCELERATE_NEW_LAPACK && /tmp/t
cd training/training_dynamic && make MODEL=gen_r0_overfit EXTRA=-DMTP_DEPTH=1
./train --scratch --overfit --data r0_synthetic.bin --steps 400 --accum 1 --wd 0 --lr 2e-3 --warmup 10
```

The held-out-val ablation (does MTP reduce tokens-to-target on a real rung?) is the
follow-up measurement; R0-green + the FD check are the correctness gate.
