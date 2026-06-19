# lilbro — Python research instrument

The Python side of lil-bro: the shared config contract, the MLX twin (+ torch
oracle), and the eval harness. This is the software foundation under the ROADMAP's
**R0 overfit** and **R1 gradient-diff** gates — everything here runs and is
verified on-device (Apple Silicon, MLX on the Metal GPU); the ANE Objective-C
trainer (`training/training_dynamic`) is the other consumer of the same config.

## Layout

| Package | Role |
|---|---|
| `configs/` | The single shared config schema (one schema, two consumers: MLX + emitted C header). |
| `mlx_ref/` | MLX twin (oracle + GPU baseline), torch fp64 oracle, shared Muon/AdamW, MTP. |
| `eval/`    | uint16 token-stream loader (data00/data01), fixed-batch val loss, generation sampler. |
| `ablation/`| Experiment runner (deferred — needs ANE training runs). |

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install mlx numpy torch pytest
```

`mlx` is the trainer-side twin + GPU baseline; `torch` (CPU, fp64) is the
correctness oracle / tie-breaker.

## Run the gates

```bash
.venv/bin/python -m pytest                       # all tests (~3s)
.venv/bin/python -m pytest tests/test_grad_diff.py   # R1: ANE-twin grads vs torch fp64
.venv/bin/python -m pytest tests/test_overfit.py     # R0: overfit one batch -> loss ~0
```

Emit an ANE model header from a shared config:

```bash
.venv/bin/python -m lilbro.configs.emit_c r2_small > training/training_dynamic/models/gen_r2_small.h
```

## The two seams (PRD testing decisions)

- **Seam 1 — gradient diff (R1).** Shared config + fixed seed + identical init +
  one fixed batch → one forward+backward on both backends → all parameter grads
  agree within fp32-scale tolerance. Covers base + GQA + MTP. The torch fp64 twin
  is the oracle; the ANE backend joins once its runtime-config wiring lands.
- **Seam 2 — overfit one batch (R0).** Train a tiny config on one repeated batch;
  loss collapses toward ~0. Black-box, exercises the full loop incl. Muon.

## Deferred (need ANE hardware + long runs)

Parameterizing the Objective-C trainer to consume the emitted header, real ANE
training runs, the R2 ablation matrix, and the R3 energy verdict — see
[ROADMAP.md](../ROADMAP.md).
