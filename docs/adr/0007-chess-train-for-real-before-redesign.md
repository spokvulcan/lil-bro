# ADR 0007 — Chess G2: the loop was never trained and its teacher is blind — train it properly before any redesign

- **Status:** Accepted — ratified via `/grill-with-docs`, 2026-06-22.
- **Date:** 2026-06-22
- **Context docs:** [chess CONTEXT.md](../chess/CONTEXT.md) ·
  [ADR 0006](0006-g2-learning-quality-gpu-path.md) (diagnosis-first GPU path — **executed**; this
  ADR is its sequel) · [ADR 0005](0005-chess-rl-self-play-on-ane.md) · results:
  [g2 diagnosis](../../results/chess_g2_diagnosis.md) · [n-step](../../results/chess_nstep.md) ·
  [6-layer smoke](../../results/chess_6layer_smoke.md) · [G1](../../results/chess_g1.md) ·
  [speedup](../../results/chess_speedup.md).
- **Extends ADR 0006.** ADR 0006's plan (GPU learner, diagnosis → H2, n-step value densification,
  Muon, 2 → 6 layers) was **executed in full** and G2 is **still red**. The handoff asked "why
  doesn't it learn?" This ADR records the answer found via `/grill-with-docs` and the next plan. It
  does **not** supersede 0006's substrate decisions (GPU `--mps-graph` path, ANE retained as
  non-default, Muon, TD(λ)) — all retained.

## Context — what the grilling found (every claim verified this session)

ADR 0006 closed the *numerical* story (0/1800 non-finite grads) and the *value-sparsity* story
(TD(λ): `loss_val` 1.10 → 0.18). G2 stayed red, framed as "learning quality / slow convergence."
The grilling re-derived from primary sources and found that framing was wrong on two counts:

1. **The loop was never actually trained.** Every recorded chess run uses `iters=30` (≈1800
   updates, ~7 min on M3 Max) — verified by grep across `results/`; no longer run exists anywhere.
   "Not learning" is, on the evidence, **"never trained past a smoke test."**

2. **The loop *is* climbing, not flat.** vs-random 0.545 → 0.740, vs-greedy 0.403 → 0.470
   (`chess_nstep.md`); draw-rate vs-random 91% → 64.5% (`chess_g2_diagnosis.md`). A flat curve would
   mean "broken"; this is **slow** learning on a tiny budget.

3. **6 layers came out *worse* than 2 at the same budget** (vs-greedy 0.448 vs 0.470) — the textbook
   signature of an **under-trained** larger model, corroborating (1).

4. **The generation teacher is tactically blind — the dominant finding.** Training data is generated
   at `--sims 16 --considered 16`. Tracing `mcts_seq_halving(m=16, N=16)` (`mcts.c:127`): 4 phases,
   but **phase 0 alone (16 candidates × 1 visit) exhausts the 16-sim budget**, so Sequential Halving
   — the entire point of Gumbel — **never advances past phase 0** (`mcts.c:405-411`). Generation
   search is a **flat 1-ply re-ranking of 16 moves**. From the G1 gate (`chess_g1.md:33`), mate-in-1
   needs ≤128 sims and **mate-in-2 needs ≥384**. The policy-improvement operator that *is* the
   AlphaZero learning signal is no stronger than the net it teaches — which explains both the slow
   climb **and** the "can't convert a won game to mate / draws saturate" symptom (a 1-ply search
   cannot see a forced mate).

5. **The G2 measurement is itself crippled.** Eval runs at `--eval-sims 8 --eval-max-plies 50` — a
   1-ply search with a 50-ply cap. Even a net that learned to convert could not demonstrate it.
   "vs-greedy 0.47" conflates **net** weakness with **measurement** weakness.

6. **The headline metric does not exist.** Self-anchored Elo (the G3 headline; the only metric that
   can measure unbounded climb) is unbuilt — `replay.h:4` records the "no league" decision; the only
   eval is `opp_random`/`opp_greedy`, saturating fixed bots. The operator's stated goal — *fastest
   and infinite learning* — is currently unmeasurable.

## Decisions (ratified in `/grill-with-docs`, 2026-06-22)

1. **Fix in place; do not rewrite.** The loop climbs, is numerically clean, and has perft + G0 + G1
   green. A from-scratch rewrite would discard three verified gates and a working substrate to
   re-solve solved problems. *Rejected:* full rewrite; a targeted MLX port of the hot loop is held
   as an escape hatch **only** if substrate friction becomes the iteration bottleneck.

2. **Train it for real with full-enough games — the first move, not a redesign.** The smoke budget
   and the 20-ply generation horizon are *why* it looks stuck. This is "stop running a toy," not a
   method change.

3. **Strengthen the teacher: generation `sims 16 → 128`** (keep `considered 16`) so Sequential
   Halving runs ~3 real phases and mate-in-1 becomes visible. Single highest-leverage lever for
   *both* faster learning and the conversion failure. *Rejected for Run 1:* playout-cap
   randomisation (the better efficiency answer — deferred to **after** Run 1 confirms the teacher
   lever); moderate `sims 64 / considered 8` (the narrow set risks dropping the best move while the
   prior is still near-uniform).

4. **Strengthen the measurement: eval `sims 8 → 128`, `max-plies 50 → 200`.** Non-negotiable —
   without it the gate cannot read conversion. The fixed bots remain a **cheap absolute floor**, not
   the headline.

5. **Build self-anchored Elo as the headline slope, in parallel with Run 1.** A checkpoint
   tournament + Elo, with a periodic capped-Stockfish anchor to catch non-transitive drift. This is
   the measurement substrate for the entire "infinite learning" goal — a fixed greedy bot is a
   ~1000-Elo wall that cannot measure further climb. Three-tier stack: **floor** (random/greedy) ·
   **slope** (self-anchored Elo) · **anchor** (capped Stockfish, periodic). Matches the CONTEXT.md
   G2/G3 design, now actually implemented.

6. **"Infinite learning" = an evidence-gated growth ladder, not a property of the loop.** A fixed
   net plateaus at its capacity ceiling. Grow the net (net2net widen/deepen, or distil into a bigger
   student) **only when a capacity ceiling is measured** — self-anchored Elo flat at fixed size
   despite more compute. The cycle *train-to-ceiling → grow → train-to-new-ceiling* is the
   operationalisation; "infinite" is the limit of the ladder. *Rejected:* front-loading growth
   machinery now (violates evidence-before-assertion — no ceiling measured yet); a single
   high-ceiling fixed net (a hard ceiling still exists, and bigger nets learn slower early).

7. **Budget allocation: teacher-first, medium games, high reuse — the readable-run constraint.** At
   2–4 h, full richness (~80×/game = 10× plies × 8× sims) buys only ~15–30 iterations (≈ the smoke
   count) — too few to read a slope. *Derived:* 38,350 games/hr (`chess_speedup.md`) ÷ 80 ≈ 480
   games/hr → ~1k–1.9k games in 2–4 h. So Run 1 keeps `sims=128` but caps games at **~80 plies** (4×,
   still reaches the endgames where conversion is learned) and **cranks replay reuse** (`lsteps`) so
   each expensive game drives many updates. ~32×/game → ~40–80 readable iterations. **The read is
   the slope, not the endpoint.**

## Run 1 — the first real run (actionable spec)

Single `--mps-graph` run; keep Muon + TD(λ)=0.5 + existing stabilizers; `seed=42` (deterministic).

- **Generation:** `--sims 128 --considered 16`, `--max-plies ~80`, `B=64`, curriculum + adjudicate on.
- **Eval (strengthened):** `--eval-sims 128 --eval-considered 16 --eval-max-plies 200`,
  `--eval-games 200` (low-noise), `--eval-every` set to give ≥8 curve points.
- **Learner (high reuse):** raise `lsteps` (≈60 → 150–200) and/or `lbatch` so each expensive game
  drives many updates; replay window sized to ~20–30 iters of history.
- **Budget:** target 2–4 h wall-clock; let `iters` be whatever fits (~40–80).
- **Instrumentation:** per-iter curve (vs-random + vs-greedy at strengthened eval) **and** grad-norms
  to stderr. Fit and report the **slope** with its noise band, not just first/last points.

**In parallel (decision 5):** build the self-anchored Elo harness — snapshot a checkpoint every N
iters, round-robin the latest K, compute Elo. Capped-Stockfish anchor is a later add, not a Run-1
blocker.

## Decision rule after Run 1 (pre-committed, to avoid post-hoc knob-tuning)

- **Slope clearly steeper / higher than the smoke baseline** → budget + teacher *was* the story.
  Escalate budget (overnight → multi-day) and keep climbing; promote self-anchored Elo to headline.
- **Slope still ~flat at strengthened eval over ~40–80 iters** → this is now a *real* signal (not the
  15-iter ambiguity). Next lever, in order: value-head readout (mean-pool is lossy), then playout-cap
  randomisation, then net size — one variable at a time, each against this run as the control.
- **Self-anchored Elo flat at fixed size despite more compute** → a measured **capacity ceiling** →
  first growth-ladder rung (decision 6).

## Gate ladder (unchanged; this ADR changes how G2 is *measured* and *fed*)

perft ✅ · G0 ✅ · G1 ✅ · **G2** — now read at strengthened eval (`sims≥128`, `max-plies≥200`) so it
can detect conversion · **G3** — self-anchored Elo, now to be built.

## Deferred — explicitly *not* Run 1 (recorded so they aren't silently dropped)

- Playout-cap randomisation (the real sample-efficiency answer — after Run 1 confirms the teacher lever).
- Dense shaping / GRPO per-move (ADR 0006 decision 4 — ablations on a green baseline).
- mHC (ADR 0006 decision 5e / step 8 — no stability problem exists; not triggered).
- Full 200-ply generation games — trimmed to ~80 for Run-1 readability; revisit at larger budget.
- Growth machinery (net2net / distillation) — gated on a measured capacity ceiling.

## Open (tuning / measurement, not blockers)

- Exact `sims` (128 is the floor where mate-in-1 appears; mate-in-2 wants ≥384 — a later sweep).
- Exact `max-plies` / `lsteps` split that maximises readable iterations at 2–4 h (measure the
  gen/eval/learn wall-clock split on the first launch and rebalance).
- Self-anchored Elo K (tournament size) and snapshot cadence vs. its games cost.

## Implementation status — landed 2026-06-22 (Run 1 + Elo harness wired & gated)

All pure-C gates green; `--selfcheck` cos 1.000000; the full `--elo` pipeline runs end-to-end.

- **Run 1 preset** — `lilbro/chess/config.py` `LADDER["run1"]`: gen `sims=128 considered=16`
  `max_plies=80`, `lsteps=150`, eval `sims=128 considered=16 max_plies=200`, Muon + TD(λ)=0.5,
  `elo_every=8`. Launch:
  ```
  cd training/training_dynamic && make train_selfplay
  python3 -m lilbro.chess.run run1 --mode g2          # (--dry-run prints the argv)
  ```
  `iters`/`eval_every` are a STARTING point — add `--profile`, measure the per-iter
  gen/learn/eval wall-clock split, and rebalance to the 2–4 h readable-slope budget (the Open
  items above). The read is the slope, not the endpoint.
- **Self-anchored Elo** — `chess/elo.{c,h}` (Bradley-Terry MLE, gated by `chess/test_elo.c`);
  `match_net_vs_net` in `chess/selfplay.c` (net-vs-net, color-swapped paired random openings,
  gated by a decisiveness + sign + determinism test in `chess/test_selfplay.c`); `--elo-every`
  snapshots `<ckpt>.eloNNN` inside the `--g2` loop; an `--elo` mode round-robins those snapshots
  into the curve. Snapshot 0 (random init) anchors at 0 Elo, so a learning loop climbs from 0.
  Read the curve after a run:
  ```
  python3 -m lilbro.chess.run run1 --mode elo
  # or: ./train_selfplay --elo --mps-graph --ckpt <ckpt> \
  #        --eval-sims 128 --eval-considered 16 --eval-max-plies 200 --elo-games 32
  ```
  The random/greedy floor still runs every `eval_every`; the capped-Stockfish absolute anchor
  remains deferred (decision 5).
- **New gates:** `make test_elo` (BT solver vs a hand-computed oracle) and the
  `match_net_vs_net` decisiveness/sign check inside `make test_selfplay`.
- **Not yet done (next session):** the Run-1 training run itself (launch + read the slope +
  the `--elo` curve), then rebalance per the wall-clock measurement.
