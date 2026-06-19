"""Validation/eval harness — the infrastructure upstream lacks.

  - ``tokenizer`` : byte-level tokenizer for the R0/R1 correctness rungs.
  - ``data``      : memmap uint16 token-stream loader (data00 train / data01 val).
  - ``evaluate``  : val loss on a fixed batch set.
  - ``sample``    : generation sampler (imported lazily; needs MLX).
"""

from .tokenizer import BYTE_VOCAB, decode_bytes, encode_bytes
from .data import TokenStream, write_token_stream
from .evaluate import val_loss

__all__ = [
    "BYTE_VOCAB",
    "decode_bytes",
    "encode_bytes",
    "TokenStream",
    "write_token_stream",
    "val_loss",
]
