"""Memory-mappable token-stream loader.

Matches the upstream on-disk format exactly: a flat **uint16** token stream with
no header (``n_tokens = filesize / 2``). A training/eval example at position
``pos`` is ``x = tokens[pos:pos+seq]``, ``y = tokens[pos+1:pos+1+seq]`` (next-token
shift), identical to ``train.m``.

Split is by shard: ``data00.bin`` = train, ``data01.bin`` = val — separate
TinyStories shards, so val has no train/val window leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_token_stream(path: str | Path, tokens) -> Path:
    """Write a uint16 token stream (the upstream format). Used by tests/tools."""
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(str(path))
    return Path(path)


class TokenStream:
    """A memmapped uint16 token shard."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def max_pos(self, seq: int) -> int:
        return len(self) - seq - 1

    def active_vocab(self) -> np.ndarray:
        """The sorted set of token ids that occur in this shard.

        This is exactly the ANE trainer's *compact vocab*: ``vocab_map_build``
        (cpu_ops.h) marks every ``data[i]`` and compacts the LM head to just those
        rows, so a model trained on this shard only ever learns these ids. It is
        therefore the right ``allowed_ids`` mask for sampling that model —
        restricting the softmax to it keeps generation out of the untrained tail
        (see ``lilbro.eval.sample``). Returned as int64.
        """
        return np.unique(np.asarray(self.tokens)).astype(np.int64)

    def batch_at(self, positions, seq: int) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) for the given start positions; y is x shifted by one token."""
        positions = np.asarray(positions, dtype=np.int64)
        x = np.stack([self.tokens[p:p + seq] for p in positions]).astype(np.int64)
        y = np.stack([self.tokens[p + 1:p + 1 + seq] for p in positions]).astype(np.int64)
        return x, y

    def random_batch(self, batch_size: int, seq: int, rng: np.random.Generator):
        pos = rng.integers(0, self.max_pos(seq), size=batch_size)
        return self.batch_at(pos, seq)

    def fixed_val_batches(self, n_batches: int, batch_size: int, seq: int, seed: int = 0):
        """A deterministic list of (x, y) batches — comparable step-to-step and
        run-to-run, so val loss is a stable signal. Positions are drawn once from
        ``seed`` and frozen."""
        rng = np.random.default_rng(seed)
        total = n_batches * batch_size
        positions = rng.integers(0, self.max_pos(seq), size=total)
        return [self.batch_at(positions[i * batch_size:(i + 1) * batch_size], seq)
                for i in range(n_batches)]
