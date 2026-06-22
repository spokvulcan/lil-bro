# Chess Muon — Slice 5 one-variable read (issue #28)

**Date:** 2026-06-22 · **Machine:** Apple **M3 Max** · **Status:** implementation + measurement
complete; **G0 green**, **G2 still not green**.

## What changed

The chess learner now uses the V4 optimizer split:

- canonical 2D trunk matrices (`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `W3`) -> **Muon**
  (`cpu_ops.h` V4 hybrid Newton-Schulz, fp64 internally);
- RMSNorm gains, embeddings, policy/value heads, and non-trunk parameters -> **AdamW**.

This run keeps the slice-4 GPU learner setup fixed and changes only the optimizer split:
`--mps-graph`, `g2_diag` shape, seed 42, 1800 learner steps, `eval_games=200`.

## Gates

| gate | result | evidence |
|---|---:|---|
| optimizer dispatch test | pass | `make test_chess_optimizer` |
| G0 overfit | pass | `make g0`: `loss_pol=0.00001`, `loss_val=0.00000` |
| grad finite count | pass | 1800/1800 grad lines, nonzero-NaN count = 0 |
| G2 | **red** | vs-random climbs but does not beat 0.85; vs-greedy stays flat |

## G2 curve with Muon

| iter | loss_pol | loss_val | vs random | vs greedy |
|---:|---:|---:|---:|---:|
| 0  | 0.0000 | 0.0000 | 0.545 | 0.403 |
| 5  | 2.1654 | 0.5840 | 0.655 | 0.435 |
| 10 | 2.1322 | 0.5215 | 0.640 | 0.432 |
| 15 | 2.3241 | 0.5469 | 0.655 | 0.422 |
| 20 | 2.4677 | 0.4393 | 0.632 | 0.415 |
| 25 | 2.4390 | 0.5053 | 0.667 | 0.425 |
| 30 | 2.5803 | 0.5173 | 0.657 | 0.403 |

Verdict from the binary:

```text
vs random : start 0.545 -> end 0.657  (max 0.667)  climb=yes beats-random[>=0.85]=NO
vs greedy : start 0.403 -> end 0.403  (max 0.435)  climb=NO beats-greedy[>0.50]=NO
=> G2 NOT YET
```

## One-variable comparison vs slice 4

Slice-4 baseline from `results/chess_g2_diagnosis.md`:

| metric | AdamW GPU baseline | Muon split | read |
|---|---:|---:|---|
| final vs-random | 0.677 | 0.657 | -0.020 |
| max vs-random | 0.677 | 0.667 | -0.010 |
| final vs-greedy | 0.407 | 0.403 | -0.004 |
| max vs-greedy | 0.453 | 0.435 | -0.018 |
| final loss_val | 0.97 | 0.5173 | value head leaves the ln(3) floor |

Verified: Muon changes the optimization dynamics substantially. The value loss no longer stays near
the `ln(3)` WDL-uniform floor, but the win-rate gate does not clear and the final/max eval scores
do not beat the slice-4 baseline. The remaining blocker is still downstream learning quality, not
gradient corruption.

## Reproduce

```bash
cd training/training_dynamic
./train_selfplay --g2 --B 64 --sims 16 --considered 16 \
  --dir-alpha 0.3 --dir-frac 0.25 --temp 1.0 --temp-moves 8 --max-plies 20 \
  --replay 80000 --lbatch 96 --lsteps 60 --iters 30 \
  --opt muon --lr 0.005 --loss-scale 256.0 --clip 1.0 --wd 0.0 --vw 1.5 \
  --td-lambda 1.0 --eval-games 200 --eval-every 5 --eval-sims 8 \
  --eval-considered 8 --eval-max-plies 50 --bench-games 320 --seed 42 \
  --ckpt /tmp/lilbro_muon_g2_diag.ckpt --curriculum --curriculum-plies 8 \
  --adjudicate --mps-graph 2>/tmp/lilbro_muon_g2_diag_grad.log
```
