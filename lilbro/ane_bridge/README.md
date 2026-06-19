# lilbro/ane_bridge

The bridge that lets the **ANE trainer** join the R1 gradient diff against the
**torch fp64 oracle**. Where `mlx_ref` is the always-on, hardware-independent
correctness check (MLX twin vs torch), this package makes the *actual Apple
Neural Engine* the system under test — the scientifically decisive R1 gate.

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

## The C end

`train.m` gained two flags, kept minimal and surgical:

- `--init <path>` — load the shared numpy weights (overwrites random init, before
  the kernels are staged), so the diff compares identical models.
- `--dump-grads <path>` — after exactly one forward+backward, dump the raw
  gradients at the accumulation unscale point (`loss_scale` cancelled, LM-head
  folded into the tied embedding, **before** clip/Adam) and exit.

## Run

```bash
.venv/bin/python -m lilbro.ane_bridge.r1_gate     # needs an Apple Neural Engine
.venv/bin/python -m pytest tests/test_ane_grad_diff.py tests/test_ane_bridge.py
```
