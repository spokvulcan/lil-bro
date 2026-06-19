"""Generation sampler — qualitative coherence check alongside the numeric loss.

Autoregressive decoding from a set of params using the MLX twin's main head.
Greedy by default; ``temperature > 0`` enables sampling. Context is cropped to
``cfg.seq`` (the trained context length).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from lilbro.configs import Config
from lilbro.mlx_ref import model as mlx_twin


def generate(params: dict, prompt, cfg: Config, n_new: int,
             temperature: float = 0.0, seed: int = 0) -> np.ndarray:
    """Return the prompt followed by ``n_new`` generated token ids (1-D uint16-range)."""
    p = {n: mx.array(a.astype(np.float32)) for n, a in params.items()}
    ids = list(np.asarray(prompt, dtype=np.int64).reshape(-1))
    key = mx.random.key(seed)
    for _ in range(n_new):
        ctx = ids[-cfg.seq:]
        x = mx.array(np.asarray(ctx, dtype=np.int32))[None]      # [1, T]
        logits = mlx_twin.forward_logits(p, x, cfg)[0, -1]       # [V]
        if temperature <= 0.0:
            nxt = int(mx.argmax(logits).item())
        else:
            key, sub = mx.random.split(key)
            nxt = int(mx.random.categorical(logits * (1.0 / temperature), key=sub).item())
        ids.append(nxt)
    return np.asarray(ids, dtype=np.int64)
