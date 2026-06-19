"""Byte-level tokenizer for the R0/R1 correctness rungs.

The cheap correctness gates use a 256-symbol byte vocab so the model stays
genuinely tiny. The efficiency rungs (R2/R3) use the fixed 32K TinyStories
tokenizer baked into the pretokenized shards (no tokenizer needed on our side —
the shards are already token ids).
"""

from __future__ import annotations

import numpy as np

BYTE_VOCAB = 256


def encode_bytes(text: str) -> np.ndarray:
    """UTF-8 bytes -> uint16 token ids in [0, 256)."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)


def decode_bytes(ids) -> str:
    """uint16 byte ids -> string (invalid sequences replaced)."""
    arr = np.asarray(ids).astype(np.uint16)
    return bytes(int(i) & 0xFF for i in arr).decode("utf-8", errors="replace")
