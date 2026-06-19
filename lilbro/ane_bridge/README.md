# lilbro/ane_bridge

The bridge between the **ANE trainer** (`training/training_dynamic/train.m`) and
the Python twins. It started as the R1 gradient diff against the **torch fp64
oracle**, and now also *drives* the trainer from the shared config, scores its
held-out validation, samples from its checkpoints, and verifies its optimizer
step — i.e. the ANE-side of PRD **1a**. Where `mlx_ref` is the always-on,
hardware-independent correctness check (MLX twin vs torch), this package makes the
*actual Apple Neural Engine* the system under test.

## What R1 proves

R0 (`results/r0_overfit.md`) showed loss collapses on the ANE — the loop runs.
R1 proves the gradients are *right*, not silently wrong: from a shared init and
one fixed batch, every ANE gradient must point the same way as the fp64 oracle
and have the right magnitude. The first run **caught a real bug** — the ANE's GQA
forward and backward tiled K/V by different conventions — see
[results/r1_grad_diff.md](../../results/r1_grad_diff.md).

## Pieces

- `serialize.py` — the flat float32 wire format both sides share: tensors in
  `param_spec` order (`embed`, per-layer ×9, `rms_final`), row-major `[out,in]`,
  no header. `write_init` seeds the trainer; `read_grads` reads its dump. MTP is
  rejected (the ANE has no MTP path). `tests/test_ane_bridge.py` pins the contract.
- `compare.py` — fp16-appropriate agreement metrics: per-parameter **cosine**
  (direction) + **relative L2** (magnitude), not the fp32 element-wise tolerance.
  A real backward bug fails by orders of magnitude; fp16 rounding passes.
- `r1_gate.py` — the hardware orchestrator: emit header → `make` → write shared
  init + fixed batch → `./train --init … --dump-grads … --overfit` → read grads →
  diff vs the torch fp64 oracle → `results/r1_metrics.json` + PASS/FAIL.
- `run.py` — the **runtime-config CLI**: one `Config` (ladder name or JSON) →
  emit header → build (cached per shape) → run `./train` with flags *derived* from
  the config. `train_argv` is the pure config→argv seam (`tests/test_ane_run.py`).
  Fixed-shape MIL graphs mean dims recompile, but optimizer/lr/accum/data/val are
  runtime flags — an ablation at a fixed shape never recompiles.
- `checkpoint.py` — read a v4 ANE checkpoint (`CkptHdr` + weights, skipping Adam
  state) into shared-config params, and `generate_from_ckpt` to sample TinyStories
  text via the MLX twin (`tests/test_ane_checkpoint.py`).
- `step_diff.py` — the **optimizer** gate: one ANE optimizer step on its own
  dumped grads vs the `mlx_ref` optimizer on the same init+grads, comparing the
  update delta. Gates AdamW and **Muon** (`results/step_diff_metrics.json`,
  `tests/test_ane_step_diff.py`).

## The C end

`train.m` gained a handful of flags, kept surgical:

- `--init <path>` — load the shared numpy weights (overwrites random init, before
  the kernels are staged), so the diff compares identical models.
- `--dump-grads <path>` — after one forward+backward, dump the raw gradients at the
  accumulation unscale point (`loss_scale` cancelled, LM-head folded into the tied
  embedding, **before** clip/Adam). Exits unless `--dump-weights` is also set.
- `--dump-weights <path>` — run one optimizer step, then dump the post-step weights
  (same flat layout) and exit — the step-diff's "weights out".
- `--opt adamw|muon` — pick the optimizer at runtime (overrides the header). Muon
  is a CPU Newton-Schulz update on the 2D matrices (norms/embed stay AdamW).
- `--val-data <path> --val-every N --val-batches K` — periodic held-out val loss on
  a fixed, evenly-spaced batch set, via the same `forward_hidden` that trains.
- `--ckpt <path>` — checkpoint output path (defaults to the header's `CKPT_PATH`).

## Run

```bash
.venv/bin/python -m lilbro.ane_bridge.r1_gate        # R1 gradient gate
.venv/bin/python -m lilbro.ane_bridge.step_diff      # AdamW/Muon step gate
.venv/bin/python -m lilbro.ane_bridge.run r2_small --steps 60000 --accum 16 --val
.venv/bin/python -m pytest tests/test_ane_*.py       # all need an Apple Neural Engine for the metrics
```
