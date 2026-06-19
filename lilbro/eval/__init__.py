"""Validation/eval harness — the infrastructure upstream lacks.

  - ``tokenizer`` : byte-level tokenizer for the R0/R1 correctness rungs.
  - ``spm_bin``   : the 32K llama2.c SentencePiece tokenizer for the R2/R3 rungs.
  - ``data``      : memmap uint16 token-stream loader (data00 train / data01 val).
  - ``evaluate``  : val loss on a fixed batch set.
  - ``sample``    : generation sampler (imported lazily; needs MLX).
"""

from .tokenizer import BYTE_VOCAB, decode_bytes, encode_bytes
from .spm_bin import SpmBinTokenizer, DEFAULT_TOKENIZER
from .data import TokenStream, write_token_stream
from .evaluate import val_loss

__all__ = [
    "BYTE_VOCAB",
    "decode_bytes",
    "encode_bytes",
    "SpmBinTokenizer",
    "DEFAULT_TOKENIZER",
    "TokenStream",
    "write_token_stream",
    "val_loss",
]
