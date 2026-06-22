# Chess n-step / TD(λ) value densification — ADR 0006 build-step 7 (PRD #22, stories 22–24)

**Date:** 2026-06-22 · **Machine:** Apple **M3 Max**, macOS 26.5.1 · **Path:** `--mps-graph`
GPU learner (MPSGraph fp32 hybrid-autodiff trunk backward) · **Branch:** `feat/chess-nstep-value`
(off `main` post-PR #37) · **Status:** **loss_val descends off the ln(3) floor — the H2 fix works.**
G2 win-rate gate (0.85 vs random) still not met in 30 iters (needs Muon/depth/more iters — sequenced next).

The pre-committed H2 fallback (ADR 0006 decision 4 / build-step 7). The diagnosis
(`results/chess_g2_diagnosis.md`) isolated the G2 blocker to **H2 (value-sparsity)**: `loss_val` was
pinned to the ln(3)=1.099 uniform-WDL floor for all 30 iters while `loss_pol` descended. n-step
densifies the terminal-only `z` with the net's own value estimates over the next plies.

## What changed

- **Schema** (`chess/replay.h`): `ReplaySample` gains `float z_nstep` (the TD(λ) value target in
  [-1,1], stm-perspective). `z` (terminal outcome) is kept for the anchor + existing tests.
- **Per-ply leaf value** (`chess/selfplay.c`): the `Game` struct gains a parallel `leaf_v[]` buffer;
  `res[k].root_value` (mcts.h:113 — evaluator value at the root, side-to-move perspective) is recorded
  per ply alongside the existing `side[]` fill.
- **Relabel** (`chess/selfplay.c relabel_value_targets`, declared in `selfplay.h`): at game end each
  ply's `z_nstep` is the TD(λ) return — `G_t = (1-λ) Σ_{n=1}^{last-t} λ^{n-1} b_{t+n} + λ^{last-t} z_t`
  (γ=1, zero intermediate rewards), where `b_{t+n}` is `leaf_v[t+n]` perspective-flipped to `side[t]`
  and `z_t = (side[t]==fstm)? fv : -fv`. λ=1.0 ⇒ `z_nstep == z` (pure Monte Carlo = the legacy label);
  λ=0.0 ⇒ 1-step TD; the last ply always anchors on the terminal.
- **Value-loss target** (`train_selfplay.m`): reads `z_nstep` and builds a **soft WDL** target
  (`tw=max(v,0)`, `tl=max(-v,0)`, `td=1-tw-tl`) — reduces to the prior one-hot at {-1,0,+1}, so
  backward-compatible. The existing cross-entropy loss + gradient (`chess_heads.h:182,189`) handle soft
  targets unchanged.
- **Knob** (`lilbro/chess/config.py` + C twin `SPConfig`): `td_lambda` (default **1.0** ⇒ zero behavior
  change; ablate down). `--td-lambda` on the CLI.

## Verification (gates, all green on the final code)

- **Pure-C seam** (`chess/test_selfplay.c`): TD(λ) math asserted at λ=1.0 (==z), λ=0.0 (1-step TD),
  λ=0.5 (the blend), ±1 boundary, with hand-computed oracle values (no net/ANE/GPU). Integration:
  a real B=1 game populates `z_nstep`, == z at λ=1.0 (backward-compat), and **72/80 plies differ from
  z at λ=0.5** (relabel active at λ<1).
- `make g0` → loss_pol/loss_val → 1e-5 (PASS). `--selfcheck --mps-graph` → cos 1.000000.
  `make test_engine test_heads test_trunk_attn test_mcts test_replay test_selfplay` → ALL PASSED
  (value-head backward still FD-green). `./chess/perft` → 593631134 exact. `pytest` → 100.

## Step 0 — confirming ablation (config-only, one variable each vs the `g2_diag` baseline)

Both re-ran `g2_diag` (eval cheap at 40 games; `loss_val` is the read, eval doesn't train) with one
knob moved. **Neither lifted `loss_val` off ln(3):**

| run | loss_val range (ln(3)=1.099) | note |
|---|---|---|
| baseline (vw=1.5, max_plies=20) | [0.91, 1.10] | the diagnosis floor |
| Run A: max_plies 20→**80** | [0.90, 1.16] | richer positions helped **policy**/win-rate (vs-random peak 0.775) but value stayed uniform |
| Run B: vw 1.5→**5.0** | [0.83, 1.07] | wider variance (head pushed harder, dipped 0.83) but no descent |

G0 (loss_val→1e-5 on a fixed batch) rules out capacity/loss/grad-path: the head *can* fit consistent
targets. So "both stuck" isolates the blocker to **target consistency** (terminal `z` = a confident ±1
on genuinely-ambiguous mid-game positions) — which is exactly what n-step replaces with a local,
fittable target. Step 0 reinforced (not contradicted) the n-step case.

## The λ=0.5 G2 run — the behavioral win

`g2_diag` (`--td-lambda 0.5`, eval_games=200, eval_every=5; the diagnosis shape, one variable changed):

| iter | loss_pol | loss_val | note |
|---:|---:|---:|---|
| 1  | 2.87 | 0.82 | already off the ln(3) floor |
| 10 | 2.28 | 0.73 | baseline loss_val was ~1.00 here |
| 20 | 2.25 | 0.62 | |
| 23 | 2.41 | **0.47** | the floor of the run |
| 30 | 2.43 | 0.58 | settled ~0.52–0.65 |

- **`loss_val`: 1.10 → ~0.58 (floor 0.47), off the ln(3) floor the whole descent.** Baseline was pinned
  to [0.91, 1.10] for all 30 iters. **The value head is learning — H2 is fixed.**
- `loss_pol`: more stable than baseline (end 2.43 vs baseline 2.61 — less of the late regression the
  diagnosis attributed to the stuck value contaminating MCTS leaf eval).
- Win-rate (200 games, ±0.035): vs-random 0.545→0.627 (max 0.670) vs baseline 0.545→0.677; vs-greedy
  0.403→0.388 (max 0.430). **Comparable to baseline (within noise) — the G2 gate (≥0.85) is not met.**
  The value head learning has not yet translated to decisively more wins; that is the job of the
  sequenced next steps (Muon → depth), not more n-step tuning.

## Honest caveat — a stale-binary trap (caught before reporting)

The first λ=0.5 / λ=0.0 G2 runs showed loss_val flat at ln(3) (identical to baseline). Bit-identical
loss trajectories across λ values is only possible if the value targets are identical — so the
`td_lambda` knob was having no effect. Root cause: `./train_selfplay` was invoked directly without a
rebuild, and `make train_selfplay` is **not** a dependency of `make g0` / `make test_*` / the selfcheck,
so the binary on disk predated the n-step code (verified: `strings train_selfplay | grep td-lambda`
returned nothing; mtime 38 min stale). The pure-C gate (`make test_selfplay`) rebuilds its own binary
from `selfplay.c` and so was correct throughout — the relabel logic was never in question. After
`make train_selfplay`, `strings` shows `--td-lambda` and λ=0.5 immediately produced divergent,
descending loss_val. **Fix-up: invoke the trainer via `make g2 G2ARGS=...` (which depends on the
`train_selfplay` build target) or rebuild explicitly before any direct `./train_selfplay` run.**

## Reproducibility

```
# from training/training_dynamic/  (rebuild first: make train_selfplay)
./train_selfplay --g2 --mps-graph \
  --B 64 --sims 16 --considered 16 --max-plies 20 --iters 30 \
  --lsteps 60 --lbatch 96 --replay 80000 --lr 5e-3 --temp-moves 8 \
  --curriculum --curriculum-plies 8 --adjudicate --vw 1.5 --td-lambda 0.5 \
  --eval-games 200 --eval-sims 8 --eval-considered 8 --eval-max-plies 50 --eval-every 5 \
  --ckpt /tmp/nstep_lam05.ckpt
```

Deterministic from `seed=42`. λ=1.0 reproduces the diagnosis baseline bit-for-bit (backward-compat).

## What this tells the plan (ADR 0006)

- **H2 is genuinely the value-signal lever**, and n-step is the correct densifier: loss_val — stuck for
  the entire diagnosis run — now descends in the same 30-iter budget. The pre-committed fallback worked.
- **The G2 gate still needs the convergence-speed + capacity levers.** The value head now learns, but
  1800 updates over a 2-layer net is not enough to convert that into win-rate gains. Per ADR 0006
  sequencing: **Muon (build-step 4) next** (faster convergence), then **QK-norm/SwiGLU-clamp**, then
  **scale 2→6 layers** — on a base where the value signal is now dense.
- λ is a tuning parameter; 0.5 is a first value, not a tuned one. A small λ sweep (0.3 / 0.7) is a cheap
  follow-up but is downstream of the convergence/depth levers above.
