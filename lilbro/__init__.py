"""lil-bro: an ANE-trained dense LM as a DeepSeek-V4 ablation testbed.

Subpackages:
  - configs:  the single shared model/run-config contract (one schema, two consumers).
  - mlx_ref:  the MLX twin (correctness oracle + GPU baseline) + a torch fp64 oracle.
  - eval:     validation/eval harness (memmap loader, val loss, generation sampler).
"""
