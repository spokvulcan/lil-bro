# Autodiff spike — MPSGraph `gradientForPrimaryTensor` on the chess trunk (issue #23 / ADR 0006 build-step 0)

**Status: GREEN — hybrid autodiff is viable; slice 2 proceeds with the autodiff trunk.**
Standalone probe: `training/training_dynamic/probe_autodiff.m`.

```bash
cd training/training_dynamic
make probe_autodiff
./probe_autodiff --fd     # autodiff vs finite-difference ground truth (the gate)
./probe_autodiff --g0     # behavioral overfit on the autodiff backward
./probe_autodiff --grad-diff   # autodiff vs the hand-written CPU backward (diagnostic)
```

The repo had **zero prior MPSGraph autodiff usage** — every graph was hand-written
forward. This spike confirms `gradientForPrimaryTensor:withTensors:` handles the full
trunk op mix (rmsnorm + scaled masked-softmax attention + SwiGLU + residuals + final
rmsnorm) in fp32, two independent ways.

## Setup

The probe builds the chess trunk as ONE MPSGraph using the **separate canonical
weights** (`Wq/Wk/Wv/Wo/W1/W2/W3/rms_att/rms_ffn/rms_final`) — the same layout the
hand-written `chess_trunk_backward` accumulates into. It defines a scalar source

```
loss = Σ (x_out ⊙ dx_final)
```

and calls `gradientForPrimaryTensor:withTensors:` to obtain `d(loss)/d(·)` for every
weight and for `x_in`. This is the **vector-Jacobian product** `Jᵀ·dx_final` — exactly
what a hand-written backward computes — so the autodiff result is directly comparable
to `chess_trunk_backward`'s `G.W*` / `dy_out`. The forward reuses the debugged
`mg_matmul`/`mg_rmsnorm`/`mg_attention` op helpers from `chess_net.h`.

## Findings

### (1) Autodiff matches finite-difference ground truth (the gate)

`--fd` central-differences `d(Σ x_out·dx_final)/d(W_elem)` on a 64-element random
sample per tensor via the CPU forward (deterministic fp32), and reports cosine over
the non-trivial-magnitude elements (near-zero grads are uninformative FD noise):

| tensor | cos(FD, autodiff) | cos(FD, cpu_bwd) |
|---|---|---|
| L0.Wq | 0.99866 | **-0.2517** |
| L0.Wk | 0.99822 | 0.5191 |
| L0.Wo | 0.99999 | 0.5711 |
| L0.W2 | 0.99998 | 0.2894 |
| L1.Wq | 0.99958 | **-0.1476** |
| L1.Wk | 0.99936 | 0.99936 |
| L1.Wo | 0.99999 | 0.99999 |
| L1.W2 | 0.99999 | 0.99999 |

**worst cos(FD, autodiff) = 0.9982** (the fp32 central-difference ceiling; `Wo`/`W2`
reach 0.9999 with the same method, confirming the autodiff is exact and the 0.998 on
`Wq`/`Wk` is FD roundoff on near-zero grad elements). **PASS.**

### (2) Behavioral overfit (the safety net)

`--g0` swaps ONLY the trunk forward+backward (ANE+CPU → MPSGraph autodiff) in the G0
loop; the heads/loss/posenc/embed/AdamW stay on the CPU floor. Both cross-entropies
collapse:

```
step 0     loss_pol=2.78239  loss_val=1.05448
step 199   loss_pol=0.00007  loss_val=0.00002   =>  PASS (G0-green on autodiff)
```

A wrong backward does not overfit to ~0; combined with (1), the autodiff backward is
correct. Forward agreement with the CPU trunk is cos = 1.000000 (max_abs_err 2.3e-4).

### (3) PRE-EXISTING BUG found in the hand-written `chess_trunk_backward` (independent of this spike)

The `--grad-diff` mode (autodiff vs the CPU backward) initially looked like an
autodiff *failure*: the CPU and autodiff disagreed on `Wq` and on all of layer 0. The
`--fd` mode resolves **which one is wrong**: the **CPU backward**. `cos(FD, cpu)` is
≤ 0.6 across the entire first layer and `L1.Wq` (as low as **-0.25** for `L0.Wq`,
**-0.15** for `L1.Wq`), while `cos(FD, autodiff)` is ≥ 0.998 everywhere.

The corruption originates in the attention backward's `dQ` path
(`attn_cpu_backward_batched` in `chess_net.h`) and propagates to everything downstream
in the reverse pass: all of L0's grads and `Wq` in both layers. `L1.Wk`/`Wo`/`W2` —
the parts of the last layer computed *before* the attention backward — are correct.

This bug is **masked by AdamW + G0-overfit-on-one-position** (a somewhat-wrong
gradient still drives one batch's loss to ~0), and it was never caught because the
only existing backward check (`train_chess --selfcheck`) compares ANE-vs-CPU — **both
paths share `attn_cpu_backward_batched`**, so a shared bug reads as cos 0.999993. This
is exactly the "silently-wrong backward" failure mode the gate ladder guards against,
and it is highly relevant to the G2-doesn't-climb mystery (G2 was trained on this
backward). Tracked separately — not in scope for slice 0.

### (4) MPSGraph first-execution note

The autodiff graph's **first** execution produces garbage grads; a single priming
forward makes it correct and deterministic (stable across re-runs). The G0 loop is
unaffected — its per-step `ad_forward → ad_backward` provides the prime naturally
(proven by the loss collapse from step 0). Slice 2's learner should ensure one priming
forward before relying on backward output (or investigate via
`MPSGraphExecutable` caching).

## Verdict + recipe for slice 2

- **Hybrid autodiff is VIABLE.** Slice 2 builds the GPU learner trunk backward on
  `gradientForPrimaryTensor:withTensors:`; the hand-written trunk backward fallback
  (ADR 0006 decision 3) is **not** needed.
- **Do NOT use `chess_trunk_backward` as a grad-diff reference for slice 2** — it is
  buggy. Use finite-difference (this probe's `--fd` mode is the template) and the G0
  overfit as the two gates, per the repo's "never trust an unverified path" law.
- **Fix or route around the CPU `attn_cpu_backward_batched` dQ bug** before slice 1's
  diagnosis pass reads grad-norms off the CPU learner (else the diagnosis measures a
  broken backward). Recommended: a new issue to FD-verify `attn_cpu_backward_batched`
  end-to-end (the heads have `test_heads_cpu.c`; the trunk attention backward has no
  FD gate today).
- **Prime the autodiff graph** with one forward before reading backward output.
