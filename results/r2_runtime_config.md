# Phase 1a foundation made real + R2 launched on the ANE

This is the ANE-side build-out the PRD's **1a** asks for — the shared config now
*drives* the ANE trainer, the trainer reports held-out **validation loss**, a
**generation** path samples from ANE checkpoints, and **Muon** is a runtime
optimizer choice verified against the numpy twin — and the first **R2** run
(d=256, 6 layers, 32K vocab, seq=256) is training on the Neural Engine.

## What landed (PRD User Stories 1, 3, 4, 5, 9)

| Piece | Where | Evidence |
|---|---|---|
| **Runtime-config CLI** | `lilbro/ane_bridge/run.py` | one config → emit header → build → run, no hand-edited `#define`s (`tests/test_ane_run.py`) |
| **Periodic val loss** | `train.m` `eval_val_loss` / `forward_hidden`; `--val-data/--val-every/--val-batches` | R2 `[val] step=0 val_loss=9.16` on the held-out `data01` shard |
| **Generation sampler** | `lilbro/ane_bridge/checkpoint.py` | loads the real R0 ckpt (magic ✓, step 400, loss 0.015) and samples via the MLX twin (`tests/test_ane_checkpoint.py`) |
| **Muon optimizer** | `train.m` `muon_update`/`newton_schulz5_f64`; `--opt adamw\|muon` | step-diff vs twin below |

### Runtime config — what "runtime" means against this hardware

The ANE compiles a **fixed-shape** MIL graph (`reshape`/`concat`/`transpose`
fail at runtime — issue #47; `config.h` bakes `DIM`, `SEQ`, …). So model
*dimensions* recompile (the header is re-emitted, `make` caches by mtime: an
unchanged shape is a no-op). Everything that does **not** change a tensor shape —
**optimizer, lr, weight decay, accumulation, warmup, clip, data shard, validation
cadence, resume/scratch** — is a runtime `./train` flag. An ablation that varies
*optimizer* or *lr* at a fixed shape therefore never recompiles: exactly the
sweep the PRD wants. `--opt` overrides the optimizer at runtime, so baseline (AdamW)
vs Muon is a one-flag change on one binary.

### The forward is defined once

Validation reuses the **same** `forward_hidden` the training step runs, so val
loss is measured by the exact code path that trains (no second, drifting forward).
Verified two ways: the R1 gradient gate is **bit-identical** after the refactor
(`results/r1_grad_diff.md` metrics unchanged), and at init val_loss ≈ train loss
(byte vocab 4.628 vs 4.645; R2 9.160 vs 9.166 — they differ only because val
averages several positions).

## Optimizer step-diff — the ANE's AdamW/Muon vs the numpy twin

R1 proves the ANE's *gradients* are right; this proves the **optimizer update**
applied to them is right. From a shared init + the ANE's own dumped gradients,
one optimizer step on the ANE is compared to `lilbro.mlx_ref`'s optimizer applied
to the *same* init+grads, comparing the **update delta** `w − init` (a small step
leaves `w ≈ init`, so comparing `w` would pass trivially).

Gate (tight — both sides run the same math on the same float32 grads, Newton-Schulz
in float64; only the final float32 weight write differs): **cosine ≥ 0.999,
rel_l2 ≤ 0.02**. Result on M3 Max — `sd_base` (MHA) and `sd_gqa2` (GQA ratio 2),
each under both optimizers, **all PASS**:

| param (sd_base) | AdamW Δ-norm | Muon Δ-norm | cosine | rel_l2 |
|---|---|---|---|---|
| `layer.0.wq` | 6.3955 | **0.5035** | 1.0000000 | 7.7e-8 |
| `layer.0.w1` | 9.0500 | **0.7032** | 1.0000000 | 8.1e-8 |
| `layer.0.w2` | 9.0504 | **0.4863** | 1.0000000 | 1.1e-7 |
| `rms_final` | 0.7999 | 0.7999 | 1.0000000 | 2.5e-7 |
| `embed` | 12.799 | 12.799 | 1.0000000 | 5.1e-8 |

The 2D weight matrices' Muon update differs sharply from AdamW (≈12× smaller,
orthogonalized + RMS-scaled) — proof Muon actually runs Newton-Schulz, not an Adam
fallback — while the norms and the tied embedding are **identical** across
optimizers, exactly as `is_muon_param` (`lilbro/mlx_ref/params.py`) prescribes.
Agreement is to float32 round-off (rel_l2 ~1e-7, five orders under the bound).

The Muon update mirrors the twin exactly: momentum 0.95, nesterov, 5 NS steps with
coefficients (3.4445, −4.7750, 2.0315), and the `max(1, rows/cols)^0.5` RMS scale.
NS runs in float64 (`cblas_dgemm`) to match numpy; the momentum buffer reuses the
per-matrix Adam `m`-slot (unused in Muon mode).

**Weight-decay caveat (verified scope).** The ANE excludes the norm vectors from
weight decay (passes `0.0f`) in *both* its AdamW and Muon branches — the common
choice. The numpy twin's `AdamW`, by contrast, currently decays every parameter
it owns (norms included). So the ANE and twin agree *exactly* only at
`weight_decay = 0`, which is what the step-diff and the R2 baseline both use; the
AdamW decoupled-decay path is therefore **not** yet gated against the twin under
`wd > 0`. Aligning the twin to also exclude norms from decay is a small follow-up
(nothing in the ladder uses `wd > 0` today).

Reproduce: `.venv/bin/python -m lilbro.ane_bridge.step_diff` →
`results/step_diff_metrics.json` (gitignored; `tests/test_ane_step_diff.py` gates it).

## R2 — running on the Neural Engine

The headline risk (32K vocab / ANE channel limits, issue #42) **did not
materialize**. R2 compiles all 10 kernels (≈430 ms) and trains:

- **Config** `r2_small`: dim=256, 6 layers, 8 heads (MHA), head_dim=32, hidden=768,
  seq=256, vocab=32000, AdamW, lr=3e-4, 4096 tokens/step (accum 16). **13.3 M params**
  (transformer 5.1 M + embed 8.2 M). Compaction: data00 uses **9205 / 32000** tokens.
- **Overfit smoke** (one pinned batch): loss 9.19 → 2.87 in 40 steps — the full
  forward+backward+optimizer loop is correct at R2 scale.
- **Real run** (data00 train / data01 val, periodic val): launched via the CLI;
  init val_loss 9.16 ≈ ln(9205); train loss 9.17 → 5.54 by step 900. ~34 ms/step.

### Status: launched, not yet gated

The ROADMAP **R2 gate** is *"val tracks the MLX twin; coherent stories"* — that
needs the run to reach the val knee, an LR sweep, and the MLX-twin val comparison.
That is a training **campaign** (the PRD's deferred 1c work), not a code change;
this session stands up the instrument and starts the baseline. Caveat for the
twin comparison: the ANE softmaxes over the **compact** vocab (9205) while the
MLX twin uses the full 32K — comparable in trend, not identical in absolute loss;
val also skips the few `data01` tokens absent from the data00 compact set.

## R2 generation — qualitative coherence

Sampling the R2 checkpoint (`ane_r2_small_ckpt.bin`, step 46400, train loss 1.48)
through the MLX twin, decoded with the 32K SentencePiece tokenizer it trained on.
The model writes fluent TinyStories prose from a cold prompt:

> **greedy** — *Once upon a time, there was a little girl named Lily. She loved to
> play outside in the sunshine. One day, she found a shiny rock and decided to play
> with it. She put it in her pocket and went outside to play…* (coherent; loops on
> "scary dog… ran away", the usual greedy degeneration.)

> **temp 0.5** — same opening, then *"Look, mommy! I found a shiny rock!" Lily said.
> Her mom smiled…* — varied and still coherent.

> **temp 0.8 / 1.0** — progressively more inventive and less consistent, but stays
> **real English words** (`prettyosa`, `healthyChase` are real-token merges, not
> garbage).

**The compact-vocab mask is load-bearing.** The ANE only trains the 9205 ids
present in data00; the other ~22.8K LM-head rows keep their init. Restricting
sampling to the active vocab (`TokenStream.active_vocab`) keeps every temperature
in-distribution. **Without** the mask (`--no-mask`), even temp 0.8 derails into
untrained tokens within a clause — *…play outside in herzec. One day… Theater
«թumpet… 控 ши probabil spielte像lu还exec Somethingmicrosoft* — the Cyrillic / CJK /
code rows that never appear in TinyStories. This is the derail `lilbro.eval.sample`
documents, shown directly.

Reproduce: `.venv/bin/python -m lilbro.eval.generate --ckpt
training/training_dynamic/ane_r2_small_ckpt.bin --temps 0 0.5 0.8 1.0` (add
`--no-mask` for the contrast).

## Next

- Let the R2 baseline reach its val knee; add the Muon and (twin-only) MTP arms →
  the R2 ablation matrix (headline tokens-to-target).
