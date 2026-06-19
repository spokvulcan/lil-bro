# Muon → DeepSeek-V4 Newton-Schulz (issue #4)

**Status: R0 overfit gate GREEN on the ANE for both Muon variants.** The ANE
trainer's Muon now defaults to the DeepSeek-V4 hybrid Newton-Schulz; the prior
Keller-Jordan path is kept selectable for a clean one-variable comparison.

## What changed (`training/training_dynamic/cpu_ops.h`)

DeepSeek-V4 §2.4, Algorithm 1 + eq 28 (read from the V4 PDF this session):

- **Hybrid Newton-Schulz, 10 iterations, two stages.** First 8 steps use
  `(a,b,c) = (3.4445, −4.7750, 2.0315)` for rapid convergence; the final 2 steps
  switch to `(2, −1.5, 0.5)` to settle the singular values precisely at 1. (Prior:
  5 single-stage iterations on the first triple.)
- **RMS update rescale.** `O = O′·√(max(n,m))·γ` with `γ = 0.18` (the V4-Flash
  target update RMS, §4.2.1), so the orthogonalized update reuses the AdamW LR.
  (Prior: aspect-ratio scale `max(1, n/m)^0.5`.)
- **Decoupled weight decay** on Muon matrices: `w = w·(1 − lr·wd) − lr·O`
  (V4 applies weight decay to Muon params; the prior ANE path applied none).

Selected at runtime by `--muon-variant {v4|prior}` (default `v4`); mirrors the
existing `--opt` precedent. AdamW still owns the embedding, LM head, RMSNorm
weights (and, later, the mHC static/gating params), per V4 §2.4.

## R0 overfit gate (M3 Max, `gen_r0_overfit`, byte vocab)

```bash
cd training/training_dynamic && make MODEL=gen_r0_overfit
./train --scratch --overfit --opt muon --data r0_synthetic.bin \
        --steps 400 --accum 1 --wd 0 --lr 2e-2 --warmup 10            # default v4
./train --scratch --overfit --opt muon --muon-variant prior ...      # prior
```

| variant | step 0 | step 100 | step 390 |
|---|---|---|---|
| **v4 (default)** | 4.6449 | 0.0003 | **0.0000** |
| prior | 4.6449 | 0.0004 | **0.0000** |

Both collapse decisively → the full ANE loop (fwd/bwd/optimizer/write-back)
trains correctly under each Muon. The V4 update drives the residual stream to a
larger magnitude (x∈[−47,47] vs [−28,28] at step 390), consistent with the
0.18-RMS-targeted (larger) step.

## Held-out validation ablation (the measurement — to run on a real rung)

R0 green is the correctness gate; the headline number is prior-vs-V4 Muon on
held-out validation loss at a real rung (`r2_small`, train shard `data00` / val
`data01`), as a ≥3-point LR sweep with everything else held fixed:

```bash
make MODEL=gen_r2_small
for lr in 1e-3 3e-3 1e-2; do
  ./train --scratch --opt muon --muon-variant v4    --lr $lr \
          --val-data ../tinystories_data01.bin --val-every 100 ...
  ./train --scratch --opt muon --muon-variant prior --lr $lr ...   # one-variable Δ
done
```
