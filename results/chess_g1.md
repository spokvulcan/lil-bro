# Chess Gumbel-MCTS search correctness — build-step 3, gate **G1** (ADR 0005, issue #17)

**Date:** 2026-06-21 · **Machine:** Apple **M3 Max**, macOS 26.5.1 · **Status:** **G1-GREEN.**
**Search:** [`chess/mcts.c`](../training/training_dynamic/chess/mcts.c) ·
**API + evaluator interface:** [`chess/mcts.h`](../training/training_dynamic/chess/mcts.h) ·
**Gate:** [`chess/test_mcts.c`](../training/training_dynamic/chess/test_mcts.c)

Build-step 3 implements the **policy-improvement operator** — REAL **Gumbel-AlphaZero**
MCTS on the CPU (Danihelka et al., 2022) over the #15 engine's legal moves (ADR 0005
decisions 2/9) — and proves **search correctness independently of learning** (the **G1**
rung): on a mate-in-1/-2 suite the search returns the forced move *given a correct
(oracle) value*. The leaf evaluator is a **pluggable interface** (a stub oracle here);
the real ANE net (#16) drops into the same contract at build-step 4 (#18).

---

## TL;DR — the verdict: **forced move on all 9 mate positions @ 512 sims, robust across 40 seeds. GO to build-step 4 (#18, the self-play loop).**

- **G1 (the gate):** a 9-position suite — **5 mate-in-1 + 4 mate-in-2**, including two
  **material distractors** — solved at a stated **512-sim** budget, `max_considered=64`,
  `seed=42`. Each answer is verified by the **engine itself** (a mate-in-1 answer must
  deliver checkmate; a mate-in-2 answer must be a genuine forced-mate key: after it,
  *every* opponent reply allows an immediate White mate). Every position + key was also
  cross-checked with python-chess **offline** (eval-side tooling only — never the hot
  path). `make g1` → **9/9 → G1-GREEN**, exit 0.
- **It is the search, not a clever oracle (the distractors):**
  - **m1e** `r5rk/6pp/7N/8/8/8/8/6QK w` — the search plays the *quiet* `Nf7#` over
    `Nxg8` which grabs a **rook**.
  - **m2d** `6k1/5ppp/8/8/8/7q/5PPP/R5K1 w` — the search plays `Ra8!` (mate in 2) over
    `gxh3` which wins a whole **queen**. (Grabbing the material leads to *no* forced
    mate — verified — so a greedy search fails here.)
- **MEASURED, not asserted (the project's whole point):** a budget sweep + a **40-seed**
  robustness scan show the suite is solved for **every** seed at **≥ 384 sims** (mate-in-1
  at ≤ 128; mate-in-2 needs the key to survive Sequential Halving *and* complete its
  solver proof). 512 is that threshold plus margin. One m2 search = **0.54 ms / 382
  nodes**; the whole gate runs in ~10 ms.
- **Deterministic:** seeded splitmix64 Gumbel noise → two `--g1` runs are byte-identical.

## What the search is (REAL Gumbel-AlphaZero, not the probe stub)

`probe_chess.m`'s `mcts_one_sim` models only the per-sim *CPU cost shape* (fake
select/expand/backup). This is the algorithm:

- **Gumbel root-action selection + Sequential Halving** (`mcts_considered_set`,
  `mcts_seq_halving`): the considered set is the top-`min(m,n_legal)` actions by
  `log P(a) + Gumbel(a)`; SH spends the budget over `ceil(log2 m)` halving phases and the
  final pick is `argmax_a [ g(a) + log P(a) + sigma(q̂(a)) ]`, `sigma(q) = (c_visit +
  max_b N_b)·c_scale·q` (paper defaults 50/1).
- **Non-root deterministic completed-Q selection:** `argmax_a [ pi'(a) - N(a)/(1+ΣN) ]`,
  `pi'(a) = softmax(log P(a) + sigma(completedQ(a)))` with the mixed value `v_mix` filling
  unvisited actions — the policy-improvement operator the paper proves.
- **Negamax value backup** with a per-ply sign flip (default `gamma=1.0` = pure,
  undiscounted Gumbel-AZ; the solver proof below handles forced-mate distance, so the
  gate is gamma-independent — `gamma<1` stays an optional #18 play-quality knob).
- **MCTS-Solver proof overlay** (Winands et al., 2008) — see the diagnosis below.

## The diagnosis (why a plain low-sim search failed, and the fix)

A first cut (real Gumbel-AZ + a material+terminal oracle) **failed m2b and m2d at every
budget up to 1024** — and *more* sims made m2d *worse* (Q(`Ra8`) = −0.48 at 256 → −0.64 at
1024). Instrumenting the root statistics showed the cause: **averaging-backup dilution.**
Below the node where White has the mate (after `Ra8 Qc8`, White to play `Rxc8#`), the
search explores `Rxc8`'s ~15 inferior siblings — lines where White is *down a queen* — and
their negative values drag the forced-mate key's *mean* Q negative. Vanilla AZ escapes this
via a trained policy prior (plays `Rxc8` first) + an accurate value net; the uninformative
oracle has neither, so budget cannot fix it.

The fix is the standard remedy for exact tactics in MCTS — a **proof overlay**, engine-
derived and evaluator-independent, composing with Gumbel + SH unchanged:

1. A non-root node from which the side to move has an immediate checkmate is a **proven
   win** (value +1, not expanded) — removing exactly the dilution.
2. **Proof propagation:** a node is a proven win if *any* child is a proven loss-for-the-
   opponent (shortest mate), a proven loss only if *all* children are proven wins-for-the-
   opponent (longest resistance). Once a root move is proven, it is **locked** — selected
   regardless of the Gumbel noise, which is what makes mate-finding **seed-independent**.

This is sound by construction (a proof rests only on engine checkmates), and the gate's
independent engine verification would catch any false proof — it does not (9/9, 40/40 seeds).

## Per-component gates (the REAL functions `mcts_search` uses)

- **Oracle / terminal value:** checkmate→−1, stalemate→0, material sign + uniform priors.
- **Sequential Halving schedule:** phases `== ceil(log2 m)`, sizes halve, budget spent.
- **Gumbel considered set:** size `min(m,n_legal)`, deterministic, includes the top-prior.
- **Value-only search (overlay inactive):** on a position with a hanging queen and *no*
  mate (`r3k2r/ppp2ppp/8/3q4/4P3/8/PPP2PPP/R3K2R w`, 22 legal moves), the proof overlay
  never fires, so `exd5` is found by Gumbel root selection + Sequential Halving +
  completed-Q descent + averaging backup **alone** — isolating the core machinery that
  the mate-in-1 rung (which the solver locks at the root) would otherwise mask.
- **Value-backup sign:** a forced win backs up to **Q > +0.9** at the root; on m2d
  `Q(Ra8) > Q(gxh3) > 0` (the mate outranks winning a queen — both invert if the sign is
  wrong).
- **Improved-policy readout:** `mcts_visit_policy` and `mcts_improved_policy` are dense
  4672-vectors (`chess_move_to_index`), sum to 1 over legal moves, zero on illegal indices,
  argmax == the mating move. **#18 trains the policy head toward the improved policy** (the
  Gumbel completed-Q target, well-behaved at low sims; raw visit counts are also exposed).

## Reproduce

```bash
cd training/training_dynamic
make test_mcts   # all per-component gates + 8-seed robustness + the G1 suite
make g1          # the G1 mate suite only -> "G1-GREEN", exit 0
./chess/test_mcts --sweep   # the budget sweep that fixes 512
```

Pure C, zero deps, `-Wall -Wextra` clean. Reproducible from `seed=42` + the engine.

## What's a stub vs. the real contract for #18

- **Oracle (stub):** `chess_oracle_evaluator()` — material (`0.9·tanh(diff/5)`, so a
  proven ±1 always dominates a material edge) + uniform priors. Pure heuristic, like the
  net will be. Mate detection is the **search's** job, not the oracle's.
- **Pluggable evaluator (`ChessEvaluator`):** `(position) → (priors over legal moves,
  value in [-1,1] from the side-to-move)` — exactly what #16's forward produces (legal-
  masked policy softmax + WDL→W−L value). The net plugs in **unchanged** at #18; batched
  leaf eval across B parallel games is a #18 orchestration concern layered above this
  single-position contract.

## Gate-ladder status

perft ✅ (build-step 1) → G0 ✅ (build-step 2) → **G1 ✅ (this)** → G2 (self-play win-rate
climb) → G3 (self-anchored Elo + Stockfish calibration). Next: **#18 — the self-play loop**
(vectorized self-play, replay buffer, the learner time-shares the one ANE client; the #16
net replaces the oracle behind `ChessEvaluator`).
