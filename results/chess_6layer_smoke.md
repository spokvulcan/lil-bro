# Chess 6-layer G0/G2 smoke

**Date:** 2026-06-22; **Machine:** Apple **M3 Max**; **Base commit:** `f4847f4`;
**Path:** `--mps-graph` GPU learner (MPSGraph fp32 hybrid-autodiff trunk backward);
**Status:** **G0 green, selfcheck green, 30-iter G2 numerically stable but G2 still red.**

This was a pre-#31 capacity/stability smoke, not a formal #31 closure read. #29/#30 remain formal
G2-green ablations unless that scope is explicitly changed.

## Local diff used

- `training/training_dynamic/models/chess_g0.h`: chess smoke depth changed from `NLAYERS=2` to
  `NLAYERS=6`.
- `training/training_dynamic/chess/chess_net.h`: MPS buffer cache changed from the fixed
  two-layer-era `128` entries to `MPS_BUF_CACHE_MAX = 64 + 24 * NLAYERS`.

The cache change was required before the G2 smoke could run. The first exact G2 command aborted
immediately after iter 0 with:

```text
[mps] buffer cache overflow (128 bufs)
make: *** [g2] Abort trap: 6
```

With 6 layers, the run combines forward/eval graph buffers and learner autodiff feed/result buffers;
the fixed 128-entry pointer cache was an infrastructure limit, not a model instability or a training
knob. After scaling the cache with `NLAYERS`, the same G2 command completed.

## GPU-G0

Command:

```bash
cd /Users/owl/projects/lil-bro/training/training_dynamic
make probe_autodiff
./probe_autodiff --g0 --steps 300 --lr 0.001 --thresh 0.05
```

Result:

```text
# DIM=256 HIDDEN=512 HEADS=8 HD=32 SEQ=96 NLAYERS=6  (steps=300 lr=0.001)
step 0     loss_pol=2.72214  loss_val=1.18871
step 50    loss_pol=0.00010  loss_val=0.00004
step 100   loss_pol=0.00006  loss_val=0.00002
step 150   loss_pol=0.00005  loss_val=0.00002
step 200   loss_pol=0.00004  loss_val=0.00002
step 250   loss_pol=0.00004  loss_val=0.00001
step 299   loss_pol=0.00003  loss_val=0.00001

## [G0] final: loss_pol=0.00003 loss_val=0.00001  (thresh 0.050)  =>  PASS
```

The requested 600-step fallback was not run because 300 steps passed.

## Selfcheck

Command:

```bash
cd /Users/owl/projects/lil-bro/training/training_dynamic
make train_selfplay
./train_selfplay --selfcheck --mps-graph
```

Result:

```text
# chess self-play (selfcheck) - DIM=256 HIDDEN=512 L=6 SEQ=96
batched-vs-single trunk: min cos = 1.000000
readout: max|prior diff| = 1.86e-08   max|value diff| = 2.98e-08
priors sum-to-1 max err = 1.90e-07 ; values in [-1,1]: yes
=> SELFCHECK OK
```

## G2 stability smoke

Command:

```bash
cd /Users/owl/projects/lil-bro/training/training_dynamic
make g2 G2ARGS='--B 64 --sims 16 --considered 16 \
  --dir-alpha 0.3 --dir-frac 0.25 --temp 1.0 --temp-moves 8 --max-plies 20 \
  --replay 80000 --lbatch 96 --lsteps 60 --iters 30 \
  --opt muon --lr 0.005 --loss-scale 256.0 --clip 1.0 --wd 0.0 --vw 1.5 \
  --td-lambda 0.5 --eval-games 40 --eval-every 10 --eval-sims 8 \
  --eval-considered 8 --eval-max-plies 50 --bench-games 320 --seed 42 \
  --ckpt /tmp/lilbro_6l_nstep_muon_smoke.ckpt --curriculum --curriculum-plies 8 \
  --adjudicate --mps-graph'
```

`make g2` exited nonzero because the measured G2 thresholds were not met. The binary completed and
printed a measured verdict.

| iter | loss_pol | loss_val | vs random | vs greedy |
|---:|---:|---:|---:|---:|
| 0  | 0.0000 | 0.0000 | 0.550 | 0.350 |
| 10 | 2.1475 | 0.2723 | 0.575 | 0.487 |
| 20 | 1.9310 | 0.2451 | 0.637 | 0.400 |
| 30 | 2.1455 | 0.2411 | 0.675 | 0.475 |

Verdict:

```text
vs random : start 0.550 -> end 0.675  (max 0.675)  climb=yes beats-random[>=0.85]=NO
vs greedy : start 0.350 -> end 0.475  (max 0.487)  climb=yes beats-greedy[>0.50]=NO
=> G2 NOT YET
checkpoint: /tmp/lilbro_6l_nstep_muon_smoke.ckpt
```

Gradient diagnostics:

```text
1800/1800 grad lines finite
nonzero-NaN grad lines: 0
per-line count: nan=0/3976960
```

## Read

- **Capacity/stability:** 6 layers are stable on the GPU learner for this 30-iter smoke after removing
  the fixed MPS buffer-cache capacity limit. G0 is green, selfcheck is green, the run crosses the
  warmup boundary at iter 20, and all 1800 learner steps have finite gradients.
- **Learning quality:** the cheap eval curve climbs against both fixed opponents, but it does not meet
  the G2 thresholds and should not be reported as G2-green.
- **#31:** this supports spending a formal 200-game eval read if the next question is purely
  "6-layer depth stability/curve at lower noise." It does **not** close #31 or remove the #29/#30
  blocker semantics by itself.
- **#35:** no NaN, divergence, or post-warmup instability was observed, so this smoke does not trigger
  the mHC fallback on stability grounds.
