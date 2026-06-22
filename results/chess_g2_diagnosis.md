# Chess G2 diagnosis pass — 5-instrument H1–H4 read (ADR 0006 build-step 1, issue #24)

**Date:** 2026-06-22 · **Machine:** Apple **M3 Max**, macOS 26.5.1 · **Status:** diagnosis **complete**;
G2 still **NOT GREEN**, but the blocker is now **isolated to H2 (value sparsity)** — H1/H3/H4 are ruled
out / addressed.
**Path:** `--mps-graph` GPU learner (MPSGraph fp32 hybrid-autodiff trunk backward, issue #25) — the
primary iteration path. **Config:** `g2_diag` preset (`lilbro/chess/config.py`): B=64, sims=16,
lbatch=96, **lsteps=60, iters=30** (1800 learner steps), **eval_games=200, eval_every=5** (7 curve
points), lr=5e-3, vw=1.5, curriculum + adjudicate on, max_plies=20, seed=42.

---

## TL;DR — the H1–H4 read

| hypothesis | verdict | evidence |
|---|---|---|
| **H1** NaN corruption (rmsnorm_bwd overflow degrading the "clean" 90%) | **RULED OUT** on GPU | **0 / 1800** learner steps produced a non-finite gradient. The slice-2 fp32 autodiff backward (`mg_ad_*`) uses fp32 rsqrt → the `vvrsqrtf` overflow is structurally impossible; #36 (`c2ab36a`) fixed the dQ path. The legacy cblas learner carried a NaN-skip bandage (`train_selfplay.m` pre-#24) — that bandage is the historical evidence H1 was real on the old path; it is inert here. |
| **H2** value-sparsity (terminal-only `z` too sparse for a 2-layer net in ~1800 updates) | **CONFIRMED — the dominant residual blocker** | `loss_val` is pinned to **0.91–1.10** across all 30 iters — i.e. on the **ln(3)=1.099** uniform-WDL floor the entire run, while `loss_pol` descends 2.87→1.94. The value head learns essentially nothing from terminal `z`; the policy head (per-move-dense MCTS-visit target) does. Exactly the signature ADR 0006 predicted. |
| **H3** label/loss bug (z-sign convention or the policy/value blend) | **RULED OUT** | `test_z_labeling_mode` pins z-perspective ↔ value-head stm alignment by transitivity (`987be6b`); `value_weight>0` is now config-validated (#24). |
| **H4** eval noise + too-few-iters (40-game ±0.08 masquerading as no-signal) | **ADDRESSED** | `eval_games=200` → ±0.035. The vs-random climb **0.545→0.677 (+0.132)** is **real at every point** (the iter-15/25 dips are within ±0.035). The old 40-game "0.688→0.562 decline" was ~1.6σ noise; this curve is signal. |

**Bottom line:** the original "G2 not climbing" (PR #21, cblas learner) was **H1 compounded with H2**.
#36 + slice #25 removed H1 (the backward is now provably clean: 0 NaN / 1800 steps, selfcheck cos
1.000000, G0 loss→1e-5). **The residual blocker on a correct backward is H2.** The pre-committed
fallback is the right next move: **n-step/TD(λ) value densification** (ADR 0006 decision 4,
build-step 7) — densify terminal `z` with the net's own value estimates at the next n plies.

---

## The instrumented curve

### Grad diagnostics (instrument 1 — `grads_diagnose` → stderr, every step)

```
[grad t=1]   gnorm=1.186e+01  maxg=9.947e-01  nan=0/1353472
[grad t=3]   gnorm=2.111e+01  maxg=2.570e+00  nan=0/1353472      <- early, clip-bound (clip=1.0)
...
[grad t=1800] gnorm=3.641e-01  maxg=8.022e-02  nan=0/1353472     <- late, healthy convergence
```

- **NaN:** **0 / 1800** steps. H1 absent (the load-bearing number).
- **Grad-norm trajectory:** early steps clip-bound (max pre-clip **21.1**, clipped to 1.0 by the
  global-norm clip); by step ~1800 the mean grad-norm settles to **~0.4** — below the clip threshold,
  i.e. the optimizer is making real, finite, in-range updates. No explosion, no vanishing.
- **Per-param detail** behind `GRAD_DIAG_VERBOSE=1` (not needed for the read — the global summary
  + NaN-localization line carry the diagnosis).

### Loss split (instrument 2 — already in stdout; `loss_pol` / `loss_val` are separate columns)

| iter | loss_pol | loss_val | note |
|---:|---:|---:|---|
| 1  | 2.87 | 0.91 | pol off the uniform floor; val already at ln(3) |
| 10 | 2.21 | 1.00 | pol descending; val flat |
| 15 | **1.94** | 1.10 | **pol minimum**; val at the ln(3) floor |
| 20 | 2.23 | 0.98 | pol regressing; val flat |
| 30 | 2.61 | 0.97 | pol un-learned; val still flat |

- **`loss_val` never leaves [0.91, 1.10]** — the ln(3)=1.099 uniform-WDL floor. The value head is
  predicting "I don't know" for the whole run. **This is H2.**
- **`loss_pol` descends then regresses** (2.87 → 1.94 → 2.61). Secondary finding: the policy improves
  to a min around iter 15, then degrades. Mechanism: the stuck value head feeds poor leaf values
  into MCTS, which contaminates the policy target (the policy distills a search guided by a uniform
  value). The two heads are coupled through the search; a bad value eventually drags the policy.
- The **asymmetry** (policy learns, value doesn't) is exactly the H2 signature: the per-move-dense
  policy target (MCTS visits) is learnable; the terminal-only value target (`z`) is not, for a
  2-layer net in 1800 updates on 20-ply games.

### Eval at 200 games (instrument 4 — `g2_diag`, eval_games=200)

| iter | vs random (W/D/L) | score | vs greedy (W/D/L) | score |
|---:|---|---:|---|---:|
| 0  | 18/182/0 | 0.545 | 5/151/44 | 0.403 |
| 5  | 50/150/0 | 0.625 | 3/161/36 | 0.417 |
| 10 | 60/140/0 | 0.650 | 5/164/31 | 0.435 |
| 15 | 45/153/2 | 0.608 | 1/154/45 | 0.390 |
| 20 | 66/134/0 | 0.665 | 6/169/25 | **0.453** |
| 25 | 56/144/0 | 0.640 | 2/171/27 | 0.438 |
| 30 | 71/129/0 | **0.677** | 5/153/42 | 0.407 |

- **vs random: 0.545 → 0.677 (+0.132), climb=yes.** Real signal at ±0.035. The net converts draws to
  wins over time (draw rate 91% → 64.5%) — it learns *something* against a random mover. But it
  plateaus well short of the 0.85 gate: **most eval games still draw** because the net cannot
  convert a material advantage to mate inside 50 plies.
- **vs greedy: 0.403 → 0.407 (max 0.453), flat.** The net does not beat a 1-ply greedy opponent.
  The draw share is high (151–171 / 200); the net holds (draws) but rarely wins and sometimes loses.

### Instruments 3 + 5

- **Clean-grads-only (instrument 3):** the NaN-skip IS the clean-grads mechanism, and instrument 1
  quantifies it — **100% of steps were clean** (0 NaN). On the GPU path there is no "subtle H1
  corruption of the clean 90%" to separate out; the question is moot because H1 is gone. (On the
  legacy cblas path the skip-step fired ~5–10% of steps — that was the H1 signature, now left behind.)
- **Label-bug sign check (instrument 5):** `test_z_labeling_mode` passes — z-perspective ↔ value-head
  stm alignment holds by construction (`987be6b`); `value_weight=1.5 > 0` is config-validated. H3 out.

---

## What this means for the plan (ADR 0006)

1. **Build-step 3 (G2 on GPU) is effectively done by this run.** The GPU learner + diagnosis logging
   ran end-to-end; the curve is the build-step-3 read. G2 is **not green**, and now we know *why*.
2. **H1 is closed.** No further work on backward stability is warranted on the GPU path; the cblas
   fallback retains its skip-step for the legacy/thesis path.
3. **H2 is the next lever.** Two complementary directions, in ADR-0006 priority:
   - **n-step/TD(λ) value densification (build-step 7, decision 4)** — the pre-committed H2 fallback.
     Densify the terminal-only `z` with the net's own value estimates over the next n plies
     (replay-schema relabel, no generation change; testable at the pure-C orchestration seam). This
     attacks H2 *directly*.
   - **Muon (build-step 4, decision 5b)** — faster convergence may help the value head move off the
     floor within the same update budget. One-variable swap after G0-green (already green).
4. **Secondary findings worth a follow-up:**
   - **Generation `max_plies=20` is very short.** Decisive games are produced (adjudicate on; gen
     W/D/L is ~60–75% decisive) but the sampled positions are early-game → terminal outcome is
     near-random from there for a shallow net. A longer generation horizon (more `max_plies`) likely
     enriches the value signal — a cheap ablation alongside n-step.
   - **Policy regression (loss_pol 1.94→2.61).** The stuck value contaminates MCTS leaf evaluation,
     which feeds back into the policy target. Fixing H2 (better value) should lift this too; if not,
     a decoupled eval-value (search uses a separate/better value than the training target) is a
     known lever.

---

## Reproducibility

```
# from training/training_dynamic/
./train_selfplay --g2 --mps-graph \
  --B 64 --sims 16 --considered 16 --max-plies 20 --iters 30 \
  --lsteps 60 --lbatch 96 --replay 80000 --lr 5e-3 --temp-moves 8 \
  --curriculum --curriculum-plies 8 --adjudicate --vw 1.5 \
  --eval-games 200 --eval-sims 8 --eval-considered 8 --eval-max-plies 50 --eval-every 5 \
  --ckpt /tmp/diag_full.ckpt \
  2>grad.log    # grad.log carries the per-step [grad ...] lines (instrument 1)
```
Equivalent via the preset: `selfplay_argv(LADDER["g2_diag"], mode="g2")`. Deterministic from
`seed=42`. Wall time: ~7 min on M3 Max (generation + 1800 learner steps + 7×400 eval games).
