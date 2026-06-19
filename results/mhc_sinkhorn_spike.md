# mHC Sinkhorn spike — fp16 known-good recipe (issue #5)

**Status: GREEN.** The hardest mHC numeric is de-risked before the full residual
integration (#11). Standalone CPU spike:
`training/training_dynamic/mhc_sinkhorn_spike.c`.

```bash
cc -O2 -o /tmp/spike training/training_dynamic/mhc_sinkhorn_spike.c -lm && /tmp/spike
```

Unrolled **log-domain** Sinkhorn-Knopp over a 4×4 matrix (`n_hc=4`), `t_max=20`,
forward + backward, with an fp16 doubly-stochasticity check. No MLX/torch, no
gradient oracle — correctness is the doubly-stochasticity property, a
finite-difference self-check of the unrolled backward, and a behavioral overfit.

## Findings

**(1) fp16 doubly-stochasticity depends critically on the Sinkhorn temperature τ**
(worst |rowsum−1| / |colsum−1| over 200 random 4×4 logit matrices, `t_max=20`):

| τ | worst row err | worst col err | verdict |
|---|---|---|---|
| 1.0 | 1.0e-3 | 3.1e-4 | **doubly-stochastic** |
| 0.5 | 1.2e-3 | 4.6e-4 | **doubly-stochastic** |
| 0.1 | 7.6e-2 | 1.6e-3 | rows NOT converged |
| 0.05 (community default) | 1.3e-1 | 1.7e-3 | rows NOT converged |

**(1b) At the stiff τ=0.05, raising `t_max` alone does not rescue fp16:** worst row
err is 1.7e-1 (t=20) → 5.2e-2 (40) → 2.1e-2 (80) → 1.5e-2 (160) — still above the
1e-2 bar at 8× the iterations. fp16's ~3-decimal precision can't resolve the dual
potentials when the log-kernel is scaled by 1/0.05 = 20×. (The row≫col asymmetry
is because the iteration ends on a column update.)

**(2) Unrolled backward is correct:** max |analytic − central-difference| = **7.9e-11**
(loss = Σ W⊙B), i.e. the hand-written reverse pass through all 20 iterations
matches numerical gradients.

**(3) Behavioral overfit:** an isolated mHC mixing block `Y = Sinkhorn(B̃)·X` fitted
to a reachable `B*·X` target collapses **loss 1.52 → 8.7e-6 (174626×)** under plain
gradient descent through the Sinkhorn — the unrolled backward trains.

## Known-good recipe for #11

- **Unrolled log-domain Sinkhorn, `n_hc=4`, `t_max=20`** — backward correct and
  trainable; keep it CPU-first (alongside mask+softmax) per ADR 0001.
- **Use τ ≈ 0.5, not the community 0.05.** τ ≥ 0.5 gives clean fp16
  doubly-stochasticity (row/col sums within ~1.2e-3 of 1) at `t_max=20`; τ ≤ 0.1
  does not converge in fp16 at any practical `t_max`. This resolves ADR 0001 open
  question #1 ("does B converge to doubly-stochastic in fp16 at t_max=20?") —
  **yes, but only at τ ≥ 0.5.**
- If a sharper τ is later needed for expressivity: hold the dual potentials
  `(f,g)` in fp32 (B can stay fp16), or add epsilon-scaling — but the spike shows
  τ=0.5 already passes, so start there.
