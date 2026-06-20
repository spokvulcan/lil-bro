# V4 config sweep harness

Two-phase sweep of the DeepSeek-V4 ablation knobs on the **r2_small** rung, scored on
**correctness** (R0 overfit gate) and **fast learning** (held-out val vs step). This is
the harness behind [`results/v4_sweep.md`](../../results/v4_sweep.md) — read that for
the findings; this README is how to re-run it.

## Layout

| file | what |
|---|---|
| `cells.zsh` | the 10 main cells: `min`, the 7 V4 knobs one-at-a-time, `muon`, `max` (all-on) |
| `cells_c.zsh` / `cells_d.zsh` / `cells_e.zsh` | the LR-fairness stages (Muon sweep, AdamW sweep + grid close, architecture-at-fair-LR) |
| `run_r0.zsh` | Phase 1 — R0 overfit correctness gate (adamw, isolates architecture-backward) |
| `run_r2.zsh` | Phase 2 — R2 fast-learning run for cells that passed R0 |

Cell format (`|`-delimited): `name | r2_extra | r0_extra | opt | lr`. `r2_extra`/`r0_extra`
are the `-D` compile flags (`NONE` = none); the r0 build differs only where partial-RoPE
needs a smaller `rope_rotary_dims` at head_dim 16. `opt` (adamw|muon) and `lr` are R2-only —
R0 always uses adamw so it tests architecture correctness, not the optimizer.

## Paths

Scripts derive the repo root from their own location, so they run from anywhere. Build
artifacts, per-cell logs, and result TSVs go to a **work dir** — `/tmp/sweep` by default,
override with `SWEEP_WORK=...`. Nothing is written inside the repo except, on first run,
`training/training_dynamic/r0_synthetic.bin` (auto-generated; the byte-vocab overfit data).

Requires the project venv at `$REPO/.venv` (for the one-time R0 data generation) or set
`PYTHON=...`. Builds need Xcode CLT (`xcrun clang`); runs need the Apple Neural Engine.

## Reproduce

```bash
# Phase 1 — correctness (all cells overfit to ~0; fast, ~1-2 min total)
zsh training/sweep/run_r0.zsh

# Phase 2 — fast-learning screen (only cells that passed R0; ~15-25 min)
zsh training/sweep/run_r2.zsh

# LR-fairness stages (architectures already R0-green -> FORCE_R2=1, separate summary)
for s in c d e; do
  CELLS_FILE=training/sweep/cells_$s.zsh R2SUM=/tmp/sweep/r2${s}_summary.tsv FORCE_R2=1 \
    zsh training/sweep/run_r2.zsh
done

# consolidated leaderboard
for f in /tmp/sweep/r2*_summary.tsv; do tail -n +2 $f; done | sort -t$'\t' -k5 -g | column -t -s$'\t'
```

Default budget is a **screen**: 800 steps, accum 4, val every 50 (`R2_STEPS` / `ACCUM` /
`VAL_EVERY` / `SAMPLE_EVERY` ... override it). It reaches the steep early descent — enough
to *rank* configs, not to call converged frontiers. See the Caveats in `results/v4_sweep.md`.
