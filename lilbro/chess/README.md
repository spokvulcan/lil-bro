# lilbro/chess

Eval-side orchestration for the **chess self-play RL trainer on the ANE** (build-step 4,
ADR 0005, issue #18). One run-config + a thin driver that builds and runs the C/Obj-C
`train_selfplay` binary and its **G2 gate** (the self-play loop learns: win-rate climbs vs
a fixed random-mover, then a weak 1-ply baseline).

**Eval-side only.** The hot loop — self-play generation, the sliding-window replay, and the
learner — is the ANE binary (`training/training_dynamic/train_selfplay` +
`chess/selfplay.c`). Nothing here runs in that loop, and there is **no `python-chess`
dependency** (the engine is the hand-written C in `training/training_dynamic/chess/`). The
chess net's *shape* is the hand-written `models/chess_g0.h`; chess does **not** go through
`lilbro.configs.emit_c`, so — unlike `ane_bridge/run.py` — there is no C-header step, only
runtime flags.

## Run-config

`ChessConfig` is the eval-side twin of the C `SPConfig` (`chess/selfplay.h`): one
`@dataclass(frozen=True)` naming every self-play knob, with defaults matching the binary's
`sp_defaults()`. Fields: generation (`batch`/`sims`/`considered`/`temp`/`max_plies`/
Dirichlet `dir_alpha`,`dir_frac`/`curriculum`+`curriculum_plies`/`adjudicate` — the last two
are the cold-start mitigations), replay+learner (`replay_cap`/`learner_batch`/
`learner_steps`/`iters`/`lr`/`loss_scale`/`grad_clip`/`weight_decay`/`value_weight`), and the
eval ladder (`eval_games`/`eval_every`/`eval_sims`/`eval_considered`/`eval_max_plies`).
Validation rejects footguns (`batch <= 170` — the 16384 ANE wall; `considered <= sims` and
`eval_considered <= eval_sims` — Sequential Halving needs the budget).

`ChessConfig.to_argv()` is the pure, unit-tested seam: it maps a config to the exact
`train_selfplay` flag list. JSON `save_config` / `load_config` mirror `lilbro/configs`.

The `LADDER` presets reflect the **measured** ANE cost: the per-forward time is
dispatch-bound at probe scale but **bandwidth-bound** at the loop's larger seq (measured
B=64 vs B=128 ≈ 1:2 wall, so batch is *throughput-neutral* — a bigger batch buys parallel
games, not free throughput). The gate budgets are deliberately low and prove *learning*,
not strength:

| preset | what | rough scale |
|---|---|---|
| `g2_smoke` | end-to-end shake-out — does the loop run + eval at all | seconds |
| `g2` | the gate rung — vs-random curve must climb | tens of minutes |
| `g2_full` | stronger/slower once the loop is proven to learn | hours |

## Built

- `config.py` — `ChessConfig` (frozen dataclass + validation), the `LADDER` presets, JSON
  load/save, and `to_argv()` (the config -> `train_selfplay` flags mapping).
- `run.py` — the driver: `selfplay_argv(cfg, mode=…)` (pure seam), `build()` (`make
  train_selfplay`, mtime-cached), `run(cfg, …)` (subprocess), and the
  `python -m lilbro.chess.run <preset|json> [--mode g2|selfcheck|smoke] [--dry-run]` CLI.

Tests: `tests/test_chess_config.py` — config round-trip, validation, and the `to_argv` /
`selfplay_argv` contract (pure; no ANE).

## The gate

```bash
# from a checkout, build + run the measured G2 gate (the C binary prints the win-rate curve)
python -m lilbro.chess.run g2 --mode g2
# or directly:  cd training/training_dynamic && make g2
```

G2 is a **measured** gate — green only when the printed vs-random / vs-greedy win-rate
curve climbs and clears the thresholds; never asserted from loss. The run's outcome is
recorded in `results/chess_g2.md`.
