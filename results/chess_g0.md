# Chess policy/value heads — build-step 2, gate **G0** (ADR 0005, issue #16)

**Date:** 2026-06-21 · **Machine:** Apple **M3 Max**, macOS 26.5 · **Status:** **G0-GREEN.**
**Trainer:** [`training/training_dynamic/train_chess.m`](../training/training_dynamic/train_chess.m) ·
**Heads:** [`chess/chess_heads.h`](../training/training_dynamic/chess/chess_heads.h) ·
**FD gate:** [`chess/test_heads_cpu.c`](../training/training_dynamic/chess/test_heads_cpu.c)

Build-step 2 turns the dense transformer trunk into a chess **policy/value net on the ANE**
(ADR 0005 decisions 5/10/13) and proves the new heads + their backward + the fp16 trunk path
are correct by overfitting one position — the **G0** rung of the gate ladder.

---

## TL;DR — the verdict: **both losses → ~0 on the ANE. GO to build-step 3 (G1 search).**

- **G0 (the gate):** overfit one fixed position (startpos → one-hot target policy + one-hot
  target value). Policy CE falls **2.78 → 2e-5**, value CE **1.05 → 1e-5** in ~50 steps.
  Initial losses match the entropy floors (ln 20 legal moves ≈ 3.0; ln 3 ≈ 1.10), so the
  legal mask and WDL softmax are wired correctly — not a degenerate pass.
- **The fp16 path is real, not zeros** (the discipline — measured, not assumed):
  - ANE matmul vs cblas: **cos 0.999999** (the `gen_dyn_matmul_mil` primitive).
  - Full 2-layer trunk **forward** ANE vs CPU: **cos 0.999999**.
  - Full 2-layer trunk **backward** ANE vs CPU (the dx-matmuls G0 trusts): **cos 0.999949**.
- **Finite-difference gate** on every new backward (policy / value / posenc / combined AZ dx):
  max |analytic − numeric| ≤ **1e-3** (fp32, pure C). The posenc decomposes **exactly** as
  `rank_emb[rank] + file_emb[file]` (additivity 6e-8).

## Reproduce

```bash
cd training/training_dynamic
make test_heads   # FD analytic-vs-numeric gate on the new heads' backward (pure C, fast)
make g0           # substrate cos checks (ASSERTED), then the G0 overfit gate -> PASS
```

`make g0` is `./train_chess --overfit`; it aborts if the ANE substrate cos drops below 0.99,
then asserts both cross-entropies fall below the threshold (exit 0 = G0-green). Reproducible
from a fixed seed (`srand48(42)`) + the engine's start position.

## What runs where (the CPU/ANE split — ADR 0004)

- **ANE (fp16):** every trunk matmul — QKV / Wo / W1·W3 / W2 forward **and** the dx backward
  matmuls — via `gen_dyn_matmul_mil(ic,oc,seq)`, the exact primitive `train.m`'s
  `woFwd`/`qBwd`/`wotBwd`/`ffnBwd` kernels are built from (mil_dynamic.h 218/676/681/425). So
  G0 exercises the real fp16 forward+backward, just un-fused.
- **CPU (fp32):** RMSNorm, attention softmax (`attn_cpu_*`, **RoPE OFF**), SiLU, dW (cblas),
  the embedding + 2D rank+file posenc, the policy/value heads, the AZ loss, AdamW — the
  irreducible CPU floor ([[ane-resident-training-cpu-floor]]). Heads on CPU mirror the LM
  classifier head (decision 5) and dodge the WDL=3-not-mult-of-32 ANE constraint.

## Design notes / scope (v1)

- **2D posenc replaces 1D RoPE** (decision 10): learned `rank[8]` + `file[8]` summed at input
  for board squares; a `misc[32]` table for the state tokens + padding. RoPE is off by
  construction (the chess path never calls the RoPE kernel) and `ROPE_ROTARY_DIMS=0` belt-and-
  suspenders. Unit-checked for rank/file additivity.
- **Attention is causal** here (reuses the FD-verified `attn_cpu` core); bidirectional / 2D-RoPE
  attention is a deferred ablation (ADR 0005 "Open"). Causal is sufficient to overfit one
  position, so it does not affect the G0 correctness gate.
- **Un-fused matmul-primitive trunk** (separate evals + CPU attention) is chosen for
  correctness and per-kernel verifiability; the fused `sdpaFwd`/`ffnFused` kernels are a later
  throughput step (G2+). Neither changes the gate.
- **V4 knobs OFF** (decision 11): plain transformer; AdamW for the gate (lower-risk than Muon
  for overfit, as on LM-R0). Net: DIM=256, HEADS=8, HD=32, HIDDEN=512, 2 layers, SEQ=96.

## Gate-ladder status

perft ✅ (build-step 1) → **G0 ✅ (this)** → G1 (Gumbel-MCTS on known tactics) → G2 (self-play
win-rate climb) → G3 (self-anchored Elo + Stockfish calibration).
