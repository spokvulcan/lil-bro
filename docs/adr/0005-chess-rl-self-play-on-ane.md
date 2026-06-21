# ADR 0005 — Chess RL self-play on the ANE: a method-first, search-guided policy/value transformer

- **Status:** Accepted (v1 design) — ratified via `/grill-with-docs`, 2026-06-21. The build is
  **probe-gated** (Build order step 0); the "Open" items are tuning/measurement, not blockers.
- **Date:** 2026-06-21
- **Context docs:** [chess CONTEXT.md](../chess/CONTEXT.md) · [CONTEXT-MAP.md](../../CONTEXT-MAP.md)
  · [ADR 0001](0001-deepseek-v4-for-dense-ane.md) (V4 tier list) · [ADR 0004](0004-ane-resident-trainer-v2.md)
  (CPU/ANE split, CPU floor, fp16 numerics) · DeepSeek-V4 paper §2.
- **Supersedes nothing.** A new sibling context on the existing ANE-trainer substrate.

## Context

The operator wants a chess model that learns *by RL self-play, on the ANE*, ultimately
competing with Stockfish, reusing lil-bro's ANE achievements. The repo today is a supervised
next-token LM trainer (`lilbro/configs/schema.py`); there is no value head, policy head,
search, self-play, game environment, or RL anywhere (grepped, 0 hits). So above the kernel
layer this is a from-scratch build; the *reuse* is the substrate — ANE fwd/bwd transformer
kernels, the compile-once dynamic pipeline, Muon/AdamW, fp16+loss-scaling.

## Decisions

1. **Method-first, not strength-first.** Deliverable = a self-play RL loop that *provably
   learns chess on the ANE*. "Compete with Stockfish" is a **yardstick** (climbing Elo, beating
   *capped* Stockfish), **not** a pass/fail bar. Rejected: strength-first, which the evidence
   says means supervised Stockfish distillation (DeepMind 2024, ~2895 Elo, search-free) —
   discarding the RL self-play that is the point.
2. **Search-guided (AlphaZero-style), not search-free.** MCTS (Gumbel-AlphaZero,
   low-simulation) is the policy-improvement operator; without it, policy-gradient self-play
   from random init is the known-weak path for chess. MCTS runs on **CPU**, calling the **ANE**
   for batched leaf evals — the same CPU/ANE split as the trainer (ADR 0004).
3. **Transformer, not conv-ResNet.** Reuse lil-bro's transformer kernels rather than build a
   new conv fwd/bwd suite — even though the ANE's native primitive is conv (ADR 0004) and
   AlphaZero is canonically conv. "Reuse our achievements" points at the transformer because
   the achievements are transformer-shaped.
4. **V4-inspired = the small-dense-relevant V4 ideas only.** Muon (winner) + cheap stabilizers
   + a deliberate mHC/MTP fork. **Not** CSA/HCA — V4's million-token mechanism, irrelevant at
   chess's ~77-token context (ADR 0001 Tier C).
5. **Input + heads.** Input: one chess **position** as ~77 special tokens (64 squares + state).
   Trunk: shared transformer. Heads: **policy** = per-square 8×8×73 (4672) logits, legal-masked
   *before* softmax via an additive −∞ bias (the same trick as the causal mask) — reuses the
   classifier-head pattern; **value** = 3-way Win/Draw/Loss softmax from a pooled/value-token
   representation (new head). Multi-head on a shared trunk — the same pattern a future language
   head will use.
6. **Scope + North Star.** v1 is **chess-play only**. The end goal — a model that *explains* and
   *teaches* at chosen Elo levels ("chess tutor") — is a later phase needing language data + a
   language head; the chess net is its backbone. v1 keeps the door open for ~free
   (shared-trunk-multi-head + growable vocab) and pays nothing more.
7. **Headline metric = self-anchored Elo.** Round-robin vs the net's *own past checkpoints*;
   *periodic* capped-Stockfish games for absolute calibration only. Stockfish stays an
   eval-only CPU tool, not a continuous dependency. Proof of learning = the curve's monotonic climb.
8. **Cold-start = purist Zero.** Random init, full games from the standard start, Dirichlet root
   noise + move-temperature, **no external labels / imitation**. Curriculum (sampling-only:
   random openings / backward curriculum) is held as a *measured* fallback, added only if the
   cold-start desert proves too long.
9. **Engine = our own C, perft-gated.** Hand-written bitboard movegen + Gumbel-MCTS, zero
   dependency, MIT-clean (the "add the op, don't add a dependency" ethos), validated by **perft**
   (a first-class correctness gate). `python-chess` is allowed for *eval-side* tooling only
   (driving Stockfish, sanity checks), never the self-play hot loop (too slow → starves the ANE).
10. **Positional encoding = 2D.** v1: learned **rank + file** embeddings summed at input, **RoPE
    off** — captures board geometry with no new attention kernel (1D sequence RoPE scrambles 2D
    adjacency). **2D-RoPE** (an SDPA fwd/bwd kernel change) is deferred as a later ablation.
11. **V4 knobs deferred — plain Muon baseline.** v1 = plain transformer + **Muon** only. mHC /
    MTP / qk-norm / attn-sink / swiglu-clamp stay **default-off ablation knobs** (LM-ladder
    evidence: mHC redundant-with-Muon + 10–22× CPU cost; MTP unproven *and* redundant with MCTS;
    stabilizers hurt at seq256). Revisit later with self-anchored Elo as the metric.
12. **Scale = small.** v1 net ~13–30M (r2_small/r2_mid shape, d=256–384, 6–8L), chosen for
    games/day (the binding constraint). The exact point is set by the throughput probe; a tiny
    ~2–5M proof-of-life is the optional fastest G2 smoke-test. Scale up only after the loop is proven.
13. **Loss = standard AlphaZero.** Policy cross-entropy to the (temperature-shaped) MCTS
    visit-count distribution + value 3-way WDL cross-entropy to the game outcome *z* + small L2.
    Reuse the trainer's fp16 **loss-scaling (256×) + grad-clip** — the new heads' fp16 softmax
    reductions are the same precision-risk class G0 guards.
14. **Self-play opponent = always-latest (AlphaZero).** The net plays its current self; no
    best-checkpoint gating, no league. The past checkpoints kept for the self-anchored Elo curve
    are **eval-only**, not opponents. Gating/league deferred — added only if self-play shows
    strategic collapse / cycling.

## Gate ladder (nothing downstream is trusted until its gate is green — LM-R0 discipline)

- **perft** — engine movegen node-counts match known values (rules correctness).
- **G0** — overfit one tiny (position → target-policy, target-value) batch → both losses ~0 on
  the ANE (the new 8×8×73 policy + WDL value heads + their fp16 backward are correct).
- **G1** — Gumbel-MCTS picks the forced move on known tactics (mate-in-1/2) given a correct value
  (CPU search correct, independent of learning).
- **G2** — full self-play produces a *monotonically climbing* win-rate vs a fixed random-mover,
  then a weak fixed baseline (the loop learns).
- **G3** — self-anchored Elo climbs across runs; periodic Stockfish calibration (the headline).

## Build order (v1) — probe-gated

0. **Throughput probe** *(first; gates everything)* — batched forward-only ANE eval at chess
   shapes + stub C movegen + stub MCTS → measured **games/day**. Sets net size / sims / parallel
   games. *(Measure, don't guess — "measure ANE utilization yourself".)*
1. **C engine** — bitboard movegen + Gumbel-MCTS; **perft**-green.
2. **Heads + loss on the ANE** — policy 8×8×73 (legal-mask additive bias) + WDL value + 2D posenc
   + the AZ loss; **G0**-green.
3. **Search correctness** — Gumbel-MCTS on known tactics; **G1**-green.
4. **The loop** — vectorized self-play, replay buffer, learner time-shares the one ANE client;
   **G2**-green.
5. **Self-anchored Elo** curve + periodic Stockfish calibration; **G3** — the headline.

## Open (tuning / measurement, not blockers)

- The probe's **games/day** number → fixes net size, sims/move, parallel-game count.
- 2D positional encoding: learned rank+file (v1) vs **2D-RoPE** (if the bias earns a kernel).
- **Curriculum fallback trigger** — how long a desert is "too long" before adding sampling accelerants.
- **Replay buffer** window + sampling (standard sliding window; tune from the probe).
- **Code layout** *(proposal)* — `lilbro/chess/` orchestration (self-play driver, replay, Elo,
  eval) + the C engine + new ANE kernels (heads, WDL loss, legal-mask, 2D posenc) under
  `training/training_dynamic/`. Move `docs/chess/CONTEXT.md` next to it once fixed.
