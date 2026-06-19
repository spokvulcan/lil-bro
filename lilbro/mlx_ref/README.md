# lilbro/mlx_ref

The **MLX twin** of the dense + Muon + MTP model, built from the same
`lilbro/configs` spec. MLX is **not** a trainer for the thesis — it plays two roles:

1. **Correctness oracle (c1):** its autograd gradients are ground truth for the
   R1 gate (ANE grads must match within tolerance). MLX is fp32-centric; if a diff
   is ambiguous, fall back to a PyTorch fp64 tie-breaker.
2. **GPU energy baseline (c2):** the strong Apple-optimized GPU trainer the ANE
   must beat on energy-/wall-clock-to-target.

Muon and MTP must be implemented **identically** here and in the ANE trainer, or
both the correctness diff and the energy comparison are invalid.

## Built

The model mirrors the upstream ANE architecture exactly (pre-norm decoder, no
biases, RMSNorm eps=1e-5, interleaved RoPE θ=10000, SwiGLU, GQA via tile,
`res_alpha = 1/√(2·n_layers)`, tied embedding/LM head):

- `model.py` — MLX twin (fp32, Metal GPU): correctness oracle + GPU baseline.
- `twin_torch.py` — torch fp64 reference: ground-truth grads / tie-breaker.
- `params.py` — single shared numpy init both backends load; `MTP_LAMBDA`.
- `optim.py` — shared CPU optimizers: AdamW + **Muon** (Newton-Schulz; 2D weights
  only, embedding/norms stay on AdamW).
- `train.py` — framework-agnostic training loop.

MTP (`mtp_depth > 0`) follows the DeepSeek-V3 recipe: per-depth module = projection
of `[RMSNorm(h); RMSNorm(Emb(future))]` → a reused transformer block → the shared
head; combined loss `main + MTP_LAMBDA·mean_k(loss_k)`.

GQA tiling here is **interleaved** (q-head `h` → kv-head `h % kv_heads`), matching
the ANE forward kernel's `concat(interleave=false)` (`mil_dynamic.h`) — `repeat` /
`concatenate`, *not* `repeat_interleave`. R1 caught the ANE backward disagreeing
with its own forward on this; see `results/r1_grad_diff.md`.

Gates: `tests/test_grad_diff.py` (MLX vs torch, all params within ~1e-4, base +
GQA + MTP — always-on) and `tests/test_overfit.py` (R0: loss→~0, AdamW + Muon +
MTP). The ANE joins the R1 grad diff at fp16 scale via `lilbro/ane_bridge` (real
hardware); it has no MTP path, so MTP stays twin-only.
