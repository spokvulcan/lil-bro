"""Generation sampler — qualitative coherence check alongside the numeric loss.

Autoregressive decoding from a set of params using the MLX twin's main head.
Greedy by default; ``temperature > 0`` enables sampling. Context is cropped to
``cfg.seq`` (the trained context length).

Sampling-mask caveat (measured on R2). The ANE trainer uses **vocab compaction**:
only the tokens present in the training shard get LM-head gradients, so for a
32K-vocab model the other ~24K rows keep their near-random init. Greedy decoding
never picks them (measured: 0/100 generated tokens were untrained), but **softmax
sampling over the full vocab draws from that untrained tail** (measured: 23-45/100
at temperature 0.8-1.0) and the story derails into off-distribution tokens. To
sample cleanly, restrict the logits to the active/compact vocab (set the untrained
ids to ``-inf`` before the softmax) — pass ``allowed_ids``. Greedy needs no mask.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from lilbro.configs import Config
from lilbro.mlx_ref import model as mlx_twin


def generate(params: dict, prompt, cfg: Config, n_new: int,
             temperature: float = 0.0, seed: int = 0, allowed_ids=None) -> np.ndarray:
    """Return the prompt followed by ``n_new`` generated token ids (1-D uint16-range).

    ``allowed_ids`` (optional): restrict generation to these vocab ids by masking
    every other logit to ``-inf`` before argmax/softmax — use the training shard's
    active/compact vocab to keep temperature sampling out of the untrained tail
    (see the module docstring). ``None`` means the full vocab.
    """
    p = {n: mx.array(a.astype(np.float32)) for n, a in params.items()}
    ids = list(np.asarray(prompt, dtype=np.int64).reshape(-1))
    key = mx.random.key(seed)

    mask = None
    if allowed_ids is not None:
        m = np.full(cfg.vocab, -np.inf, dtype=np.float32)
        m[np.asarray(list(allowed_ids), dtype=np.int64)] = 0.0
        mask = mx.array(m)

    for _ in range(n_new):
        ctx = ids[-cfg.seq:]
        x = mx.array(np.asarray(ctx, dtype=np.int32))[None]      # [1, T]
        logits = mlx_twin.forward_logits(p, x, cfg)[0, -1]       # [V]
        if mask is not None:
            logits = logits + mask
        if temperature <= 0.0:
            nxt = int(mx.argmax(logits).item())
        else:
            key, sub = mx.random.split(key)
            nxt = int(mx.random.categorical(logits * (1.0 / temperature), key=sub).item())
        ids.append(nxt)
    return np.asarray(ids, dtype=np.int64)
