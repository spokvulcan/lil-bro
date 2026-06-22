# ADR 0006 — G2 learning-quality: GPU iteration path + diagnosis-first plan

- **Status:** Accepted — ratified via `/grill-with-docs`, 2026-06-22.
- **Date:** 2026-06-22
- **Context docs:** [chess CONTEXT.md](../chess/CONTEXT.md) · [CONTEXT-MAP.md](../../CONTEXT-MAP.md) ·
  [ADR 0005](0005-chess-rl-self-play-on-ane.md) (chess-RL design, partially superseded) ·
  [ADR 0001](0001-deepseek-v4-for-dense-ane.md) (V4 tier list; ANE-only amendment, partially superseded) ·
  [ADR 0004](0004-ane-resident-trainer-v2.md) (CPU/ANE split).
- **Partially supersedes** ADR 0005 decision 11 (substrate framing — the learner moves to a GPU
  iteration path; the V4 stabilizers are recast as ablations-on-a-stable-base, not default-off
  bug-fixes) and decision 13 (the AZ loss stays as baseline; n-step value densification is the
  pre-committed fallback if diagnosis shows value-sparsity is the blocker). **Partially
  supersedes** the ADR 0001 "ANE-only" amendment (the GPU iteration path is a non-ANE substrate
  for the learner; the ANE forward path is retained as a non-default compile option — the
  "ANE can train" thesis stays documented and buildable).

## Context

PR #21 landed the MPSGraph whole-trunk-forward for generation/eval (5.3–6x speedup,
~129k games/hour), but **G2 is not green**: the win-rate doesn't climb vs random/greedy over
30 iters × 60 learner-steps. Two coupled problems:

1. **A backward numerical instability** — `rmsnorm_bwd`'s `vvrsqrtf` overflows on near-zero
   activations (`cpu_ops.h:24`), producing ~5–10% NaN-gradient steps that a skip-step bandages
   (`train_selfplay.m:210-220`). The code's own comment (`train_selfplay.m:207-209`) names this
   as the working hypothesis.
2. **A sparse RL signal** — the value target `z` is terminal-only (`selfplay.c:252`); the policy
   target (MCTS visits) is already per-move dense.

The measured G2 result (40-game evals, ±0.08 noise, 2σ≈0.16; "0.688→0.562" is ~1.6σ; "max"
inflated by selection over 6 evals) **cannot distinguish** four causes: **H1** (NaN corruption
degrading even the "clean" 90% of updates — subtle, not just the skipped 10%), **H2**
(value-sparsity for a 2-layer net in 1800 updates), **H3** (a label/loss bug — e.g. the z-sign
convention or the policy/value grad blend), **H4** (eval noise + too-few-iters). Committing to
a fix direction before separating them risks fixing the wrong thing.

The operator requires the **GPU/MPSGraph path** for all iteration (~10x faster than the
ANE/CPU learner path), V4 architecture fidelity, modern RL, and ambitious scaling.

## Decisions (ratified in `/grill-with-docs`, 2026-06-22)

1. **Diagnose first.** Before any architecture change, run a 5-instrument diagnosis pass on the
   current `--mps-graph` cblas learner (the path G2 was measured on), 30 iters + one
   200-eval-game run:
   - per-step grad-norm (max|g| per param group) + finite-count + NaN-count, logged to stderr;
   - policy loss and value loss reported **separately** per step (already computed at
     `train_selfplay.m:163-183`, not split in output) — value-stuck + policy-descending → H2;
     both-stuck → H1/H3;
   - clean-grads-only variant (the skip-step already gives ~95% clean updates) — does G2 still
     not climb? separates H1-subtle from H2/H3/H4;
   - `eval_games=200` for the final eval (kills H4: ±0.035 vs ±0.08);
   - label-bug sign check: z-sign convention (`selfplay.c:252`) vs the value head's
     stm-perspective (`chess_net.h:1018`) + the policy/value grad blend weight `vw`.
   The logging is path-agnostic and carries to the GPU learner unchanged. Uses the `diagnose`
   skill (reproduce → minimise → hypothesise → instrument → fix → regression-test).

2. **GPU iteration path; ANE retained as non-default.** Port the learner
   (forward+backward+heads+loss) from cblas-CPU to MPSGraph fp32 GPU. Keep the ANE forward path
   as a **non-default compile option** — the "ANE can train" thesis (ADR 0001/0004/0005) stays
   documented and buildable. The GPU port **subsumes the NaN fix by construction** (fp32 rsqrt,
   Apple-tested kernels) — the CPU `rmsnorm_bwd` `vvrsqrtf` overflow is left behind on the
   legacy cblas path. This recasts QK-norm + SwiGLU clamp as **V4-fidelity ablations on a
   stable base**, not bug-fixes. Consistent with today's architecture: the learner already
   bypasses the ANE under `--mps-graph` (`train_selfplay.m:404`).

3. **Hybrid-autodiff GPU backward.** MPSGraph `gradientWithSourceTensor` for the standard trunk
   (rmsnorm + matmul + scaled masked-softmax + SwiGLU — all standard ops); hand-written for the
   chess-specific heads (legal-mask policy softmax, WDL value). The repo has **zero prior
   MPSGraph autodiff usage** (every graph is hand-written forward), so **build-step 0 is a 1-day
   autodiff spike** confirming the API handles the trunk op mix in fp32, grad-diff'd against the
   CPU `chess_trunk_backward` reference (the `train_chess.m:163-164` pattern) to fp32 cosine,
   **G0 overfit as the safety net**. **Fallback:** if the spike fails, hand-write the trunk
   backward op-by-op from the CPU reference. Rationale: speed-to-first-iteration (the operator's
   priority) with the G0 gate as the verifiability net — the repo's "never trust an unverified
   path" law is honored by grad-diff + G0, not by hand-writing.

4. **Defer the RL change; n-step pre-committed fallback.** Do not change the RL signal until the
   GPU port + diagnosis settle whether RL is even the blocker. The policy target is already
   per-move dense (MCTS visits); the sparse part is the value target (terminal z). **Pre-committed
   fallback:** if diagnosis shows value-sparsity (H2) is the blocker, densify via
   **n-step/TD(λ) value targets** — re-label `z` using the net's own value estimates during
   generation (replay-schema + relabel change, no generation change). **GRPO per-move**
   (V4 §5.1.1) and **dense shaping rewards** (the operator's "legal-move = small good" intuition,
   framed as potential-based shaping à la Ng et al. 1999, which preserves the optimal policy)
   are **ablations on a G2-green baseline**, not upfront commits. Avoids feature-creep on an
   un-diagnosed failure; preserves the per-move ask as the pre-committed fallback.

5. **V4 architecture sequencing: stabilizers + Muon, then depth; mHC as fallback.**
   - (a) GPU backward **G0-green** (decision 3).
   - (b) Wire **Muon** for 2D trunk weights, **AdamW** for embeddings/heads (the V4 split —
     executing ADR 0005 decision 11, which specified Muon but never wired it in; the LM impl is
     the template, `cpu_ops.h:67`). One-variable swap, landed **after** the GPU backward is
     G0-green (Muon-on-an-unverified-backward = two unknowns).
   - (c) **QK-norm + SwiGLU clamp** as Tier-A one-variable ablations on the G2-green baseline
     (ADR 0001 Tier A).
   - (d) **Scale 2→6 layers** with Muon + stabilizers.
   - (e) **mHC only if the deeper stack doesn't stabilize** — fallback, not a first move.
     Evidence: CONTEXT-MAP records the LM finding "mHC is redundant-with-Muon and CPU-expensive";
     the LM stabilized depth with Muon alone. On GPU the CPU-cost argument weakens (Sinkhorn
     runs on GPU) but the redundancy-with-Muon argument holds. mHC (V4 §2.2: A/B/C maps +
     log-domain Sinkhorn-Knopp, `n_hc=4`, `t_max=20`) stays a deferred fallback per
     ADR 0005 decision 11.

## Gate ladder (unchanged from ADR 0005; the GPU port must clear each gate)

- **perft** — engine movegen (unaffected).
- **G0** — heads/loss overfit-one-batch → ~0, **now on the GPU backward** (the autodiff-spike gate).
- **G1** — Gumbel-MCTS solves tactics (unaffected — CPU search).
- **G2** — self-play win-rate climbs vs fixed baselines, **now on the GPU learner** (the gate
  this ADR is about). Eval at `eval_games≥200` to read below the noise floor.
- **G3** — self-anchored Elo climbs.

## Build order (probe-gated; step 0 is the autodiff spike)

0. **Autodiff spike** *(first; gates the GPU backward method)* — MPSGraph
   `gradientWithSourceTensor` on the trunk op mix in fp32; grad-diff vs CPU
   `chess_trunk_backward`; G0 overfit. Fallback: hand-written trunk backward.
1. **Diagnosis pass** (decision 1) — 5 instruments on the current cblas learner; reads H1–H4.
   **Runs in parallel with step 0** (independent; the logging carries to the GPU learner).
2. **GPU learner forward+backward+heads+loss** (decisions 2–3) — MPSGraph fp32 end-to-end;
   **G0-green** on GPU.
3. **G2 on GPU** — re-run 30-iter + 200-eval with the GPU learner + diagnosis logging. If green
   → the NaN was the blocker; proceed to V4 ablations. If not → the diagnosis tells us which of
   H2/H3/H4 remains.
4. **Muon** (decision 5b) — one-variable swap; re-run G2.
5. **QK-norm + SwiGLU clamp** ablations (decision 5c) — one-variable each, R2-style.
6. **Scale 2→6 layers** (decision 5d) — with Muon + stabilizers.
7. **(Fallback) n-step value densification** (decision 4) — if diagnosis showed H2 and G2 is
   still not green after steps 2–6.
8. **(Fallback) mHC** (decision 5e) — if depth doesn't stabilize with Muon + stabilizers.

## Open (tuning / measurement, not blockers)

- The autodiff spike's pass/fail → fixes the GPU backward method (hybrid vs hand-written trunk).
- The diagnosis's H1–H4 read → fixes whether step 7 (n-step) is needed and whether H3 (a
  label/loss bug) is in fact present.
- `eval_games` for the ongoing curve (200 for the gate read; possibly lower for the per-iter
  curve with smoothing).
- Muon hyperparameters at chess scale (the LM validated Muon at LM shapes; chess shapes differ).
