# Handoff — Chess RL self-play on the ANE (v1)

**Date:** 2026-06-21 · **Status:** design ratified, **zero implementation** · **Next action:** the throughput probe (build-step 0).

**Authoritative docs:** [`../adr/0005-chess-rl-self-play-on-ane.md`](../adr/0005-chess-rl-self-play-on-ane.md) (14 decisions + gate ladder + build order) · [`CONTEXT.md`](CONTEXT.md) (glossary) · [`../../CONTEXT-MAP.md`](../../CONTEXT-MAP.md) (sibling contexts) · PRD = `spokvulcan/lil-bro#14`.

## TL;DR

A method-first, search-guided (AlphaZero / Gumbel-MCTS) chess RL self-play trainer on the ANE, designed via `/grill-with-docs`. Nothing is built yet. The whole project is gated on **one measurement — the throughput probe** — which tells us whether purist-Zero self-play can climb on a single Mac in **days vs never**. Build that first; do not write kernels before it.

## The design in one screen

- **Net:** plain transformer trunk + Muon (V4 knobs off), ~13–30M. Input = one position as ~77 special tokens; **2D rank+file** position encoding (RoPE off). Shared trunk, two heads: **policy** 8×8×73 (legal-masked) + **value** 3-way WDL.
- **Learning:** **search-guided self-play** — Gumbel-MCTS (low-sim, CPU) is the policy-improvement operator; policy ← MCTS visit distribution, value ← game outcome z. **Purist-Zero** cold-start (Dirichlet + temperature; curriculum only if measured-necessary).
- **System:** single process, **one ANE client**, vectorized B games; leaf evals batched into one ANE forward; replay buffer; learner time-shares the client. Own **C bitboard engine, perft-validated**.
- **Metric:** **self-anchored Elo** (round-robin vs own checkpoints) + periodic capped-Stockfish calibration.
- **Trust:** gate ladder `perft → G0 (heads overfit→0) → G1 (search solves tactics) → G2 (loop beats baseline) → G3 (Elo climbs)`. Nothing trusted until its gate is green.

## Done

- Foundational design ratified (ADR 0005, decisions 1–14).
- Glossary (`docs/chess/CONTEXT.md`), context map (`CONTEXT-MAP.md`), PRD (`#14`).
- **No code.**

## Build order (probe-gated) — all not-done

0. **Throughput probe** ← **START HERE** (details below)
1. C engine — bitboard movegen + Gumbel-MCTS; **perft**-green
2. Heads + loss on the ANE — policy 8×8×73 + WDL value + 2D posenc + AZ loss; **G0**-green
3. Search correctness — Gumbel-MCTS on tactics; **G1**-green
4. The loop — vectorized self-play + replay + learner; **G2**-green
5. Self-anchored Elo curve + Stockfish calibration; **G3**

## Immediate next task: the throughput probe (build-step 0)

**Why first:** it is the single number that decides feasibility. Purist-Zero is the slowest path through the cold-start desert; if games/day is too low, re-scope (smaller net / curriculum / 2–5M proof-of-life) **before** writing any kernel.

**Measure → games/day** for the v1 loop (trunk ~13–30M, seq ≈ 77, Gumbel *n* sims/move, *B* parallel games, ~100 plies/game):

1. **ANE batched forward-only eval** at chess shapes — reuse the dynamic pipeline's forward (no policy/value heads needed; the trunk dominates). Sweep *B* ∈ {1, 64, 256, 1024}; report ms/eval + the batch-scaling curve.
2. **CPU self-play orchestration** — stub movegen + stub Gumbel-MCTS (*n* sims/move, tree bookkeeping, no real rules) for *B* parallel games; report ms/move-step.
3. **Combine** → games/day = f(ms/eval@*B*, *n*, ~100 plies, *B*, max(ANE-bound, CPU-bound)/step). Report a few (*B*, *n*).

**Deliverable:** a runnable probe + `results/chess_throughput_probe.md` with measured ms, the ANE-bound-vs-CPU-bound verdict, implied games/day, and an honest **days-vs-never** read. If too low for purist-Zero, recommend a re-scope — do **not** start kernel work.

**Gotchas (from the existing project):**
- **One ANE client only** (a second client = contention).
- ANE is **dispatch-bound** (~0.12 ms fixed/eval, compute ~free to n≈1024) → batch big, few evals.
- The private-API MIL workload is **invisible to the Xcode Instruments Core ML template**; use per-op timing + `powermetrics` ANE power (the two-signal rule).
- Likely outcome to watch for: the **CPU stub (MCTS + movegen) dominates** — meaning the real bottleneck, and the real engineering, is the C engine's speed, not kernels.

## Starter prompt for the next session

```
Read docs/adr/0005-chess-rl-self-play-on-ane.md, docs/chess/CONTEXT.md, and
docs/chess/HANDOFF.md. This is a new chess-RL-on-ANE sub-project: design
ratified, zero code.

Build "build-step 0: the throughput probe" — the measurement that gates the
whole project, before any kernel work. It decides whether purist-Zero
self-play can climb on this single Mac in DAYS vs NEVER.

Measure and report games/day for the v1 design (transformer trunk ~13-30M,
seq ~= 77 chess tokens, Gumbel-MCTS self-play, single-process / one ANE
client, vectorized over B games):

1. ANE batched forward-only eval at chess shapes — reuse the dynamic
   pipeline's forward (training/training_dynamic/); you do NOT need the
   policy/value heads, the trunk forward dominates. Sweep batch
   B in {1, 64, 256, 1024}; report ms/eval and the batch-scaling curve.
2. CPU self-play orchestration — a stub movegen + stub Gumbel-MCTS (n
   sims/move, tree bookkeeping, no real chess rules) for B parallel games;
   report ms per move-step.
3. Combine -> games/day = f(ms/eval@B, n sims/move, ~100 plies/game, B
   games, max(ANE-bound, CPU-bound) per step). Report across a few (B, n).

Honor the project's evidence discipline: measure don't guess; ONE ANE client
(second = contention); ANE is dispatch-bound (batch big, few evals);
cross-check "ANE-bound" against powermetrics ANE power; flag every estimate
as an estimate.

Deliverable: a runnable probe + results/chess_throughput_probe.md with the
measured ms, the ANE-bound-vs-CPU-bound verdict, the implied games/day, and
an honest days-vs-never read. If games/day looks too low for purist-Zero,
recommend a re-scope (smaller net / curriculum / 2-5M proof-of-life) — do
NOT start kernel work.

Start by exploring training/training_dynamic/ (train.m, mil_dynamic.h,
probe_dispatch.m) for the cheapest way to get a batched forward-only eval at
chess shapes.
```
