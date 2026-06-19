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
